// Command run benchmarks surprisal on the js13k corpus: it compresses each game's files (as `surprisal -c` would)
// and writes a markdown table next to the other methods' sizes from results.json (bench.mjs).
//
//	go run ./benchmarks/run [-year 2025] [-level 1] [-limit 10] [-verify] [-out benchmarks/out/surprisal-c1.md]
//
// Games come from benchmarks/corpus/zips/<year>/<slug>.zip (benchmarks/corpus/fetch.sh downloads them). Zips Go
// can't read (LZMA, PPMd, Deflate64) are extracted with 7z when it is installed and skipped otherwise.
package main

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/benallfree/surprisal/surprisal"
)

// baseline is one game's row of results.json (sizes in bytes).
type baseline struct {
	Year       int    `json:"year"`
	Slug       string `json:"slug"`
	Zip        int    `json:"zip"`        // the zip as submitted
	Raw        int    `json:"raw"`        // unpacked
	ZBest      int    `json:"zBest"`      // best legal zip (every 7-Zip method, Zopfli)
	Brotli     int    `json:"brotli"`     // brotli -q11 -w24
	BrotliDict int    `json:"brotliDict"` // brotli with a dictionary trained on 2020-2024
	M5         int    `json:"m5"`         // the context-mixing model pretrained on 2020-2024
}

type row struct {
	slug          string
	files, in     int
	archive       int
	secs          float64
	base          *baseline
	verified, bad bool
}

func main() {
	dir := flag.String("dir", "benchmarks", "the benchmarks directory (corpus/zips, results.json)")
	year := flag.Int("year", 2025, "the corpus year to run (2025 is the held-out test year)")
	level := flag.Int("level", 1, "compression level, as surprisal -c<level>: 1-3 (0.5B), 4-6 (1.5B) or 7-9 (7B)")
	limit := flag.Int("limit", 0, "run only the first N games (0: all)")
	only := flag.String("games", "", "comma-separated slugs to run (default: every game of the year)")
	verify := flag.Bool("verify", false, "also decompress each archive and compare (doubles the time)")
	out := flag.String("out", "", "markdown output (default <dir>/out/surprisal-c<level>-<year>.md)")
	perGame := flag.Bool("per-game", true, "include the per-game table")
	gpu := flag.String("gpu", "auto", "auto or off")
	flag.Parse()
	if err := run(*dir, *year, *level, *limit, *only, *verify, *out, *perGame, *gpu); err != nil {
		fmt.Fprintln(os.Stderr, "run:", err)
		os.Exit(1)
	}
}

func run(dir string, year int, level int, limit int, only string, verify bool, out string, perGame bool, gpu string) error {
	m, ok := surprisal.ModelForLevel(level)
	if !ok {
		return fmt.Errorf("level %d isn't 1-9", level)
	}
	if !m.Present() {
		return fmt.Errorf("model %s isn't installed: compress anything with surprisal -c%d once to install it", m.Name, level)
	}
	zipDir := filepath.Join(dir, "corpus", "zips", fmt.Sprint(year))
	names, err := filepath.Glob(filepath.Join(zipDir, "*.zip"))
	if err != nil || len(names) == 0 {
		return fmt.Errorf("no zips in %s (run %s first)", zipDir, filepath.Join(dir, "corpus", "fetch.sh"))
	}
	sort.Strings(names)
	if only != "" {
		want := map[string]bool{}
		for _, s := range strings.Split(only, ",") {
			want[strings.TrimSpace(s)] = true
		}
		var keep []string
		for _, n := range names {
			if want[strings.TrimSuffix(filepath.Base(n), ".zip")] {
				keep = append(keep, n)
			}
		}
		names = keep
	}
	if limit > 0 && len(names) > limit {
		names = names[:limit]
	}
	bases := map[string]*baseline{}
	if b, err := os.ReadFile(filepath.Join(dir, "results.json")); err == nil {
		var all []baseline
		if err := json.Unmarshal(b, &all); err != nil {
			return fmt.Errorf("results.json: %w", err)
		}
		for i := range all {
			if all[i].Year == year {
				bases[all[i].Slug] = &all[i]
			}
		}
	}

	c, err := surprisal.Open(level, 0, gpu)
	if err != nil {
		return err
	}
	defer c.Close()
	device := c.Device()
	fmt.Fprintf(os.Stderr, "%s on %s: %d games from %s\n", m.Name, device, len(names), zipDir)

	var rows []row
	skipped := 0
	start := time.Now()
	for i, n := range names {
		slug := strings.TrimSuffix(filepath.Base(n), ".zip")
		files, err := readZip(n)
		if err != nil {
			skipped++
			fmt.Fprintf(os.Stderr, "%s [%d/%d] %s: skipped (%v)\n", time.Now().Format("15:04:05"), i+1, len(names), slug, err)
			continue
		}
		in := 0
		for _, f := range files {
			in += len(f.Data)
		}
		t0 := time.Now()
		a, err := c.CompressFiles(files, nil)
		if err != nil {
			return fmt.Errorf("%s: %w", slug, err)
		}
		r := row{slug: slug, files: len(files), in: in, archive: len(a), secs: time.Since(t0).Seconds(), base: bases[slug]}
		if verify {
			_, body, err := c.Decompress(a, nil)
			if err != nil {
				return fmt.Errorf("%s: decompress: %w", slug, err)
			}
			got, err := surprisal.DecodeFiles(body)
			r.verified, r.bad = true, err != nil || !sameFiles(got, files)
		}
		rows = append(rows, r)
		fmt.Fprintf(os.Stderr, "%s [%d/%d] %s: %d files, %d -> %d bytes, %.1f s%s\n", time.Now().Format("15:04:05"), i+1,
			len(names), slug, r.files, r.in, r.archive, r.secs, verifyNote(r))
	}
	if len(rows) == 0 {
		return errors.New("no games compressed")
	}
	md := report(m, level, device, year, rows, skipped, time.Since(start), perGame)
	if out == "" {
		out = filepath.Join(dir, "out", fmt.Sprintf("surprisal-c%d-%d.md", level, year))
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(out, []byte(md), 0o644); err != nil {
		return err
	}
	fmt.Print(md)
	fmt.Fprintf(os.Stderr, "wrote %s\n", out)
	for _, r := range rows {
		if r.bad {
			return fmt.Errorf("%s: round trip FAILED", r.slug)
		}
	}
	return nil
}

func verifyNote(r row) string {
	switch {
	case !r.verified:
		return ""
	case r.bad:
		return ", round trip FAILED"
	}
	return ", round trip ok"
}

// readZip returns a zip's regular files (directories and macOS metadata left out), with 7z for methods Go lacks.
func readZip(name string) ([]surprisal.File, error) {
	zr, err := zip.OpenReader(name)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	var fs []surprisal.File
	for _, f := range zr.File {
		if f.FileInfo().IsDir() || skipName(f.Name) {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return read7z(name)
		}
		b, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return read7z(name)
		}
		fs = append(fs, surprisal.File{Path: f.Name, Data: b})
	}
	return fs, nil
}

func skipName(n string) bool {
	return strings.HasPrefix(n, "__MACOSX/") || filepath.Base(n) == ".DS_Store"
}

func read7z(name string) ([]surprisal.File, error) {
	if _, err := exec.LookPath("7z"); err != nil {
		return nil, errors.New("compression method Go can't read, and 7z isn't installed")
	}
	tmp, err := os.MkdirTemp("", "surprisal-bench-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)
	if out, err := exec.Command("7z", "x", "-y", "-o"+tmp, name).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("7z: %v: %s", err, bytes.TrimSpace(out))
	}
	var fs []surprisal.File
	err = filepath.WalkDir(tmp, func(p string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(tmp, p)
		rel = filepath.ToSlash(rel)
		if skipName(rel) {
			return nil
		}
		b, err := os.ReadFile(p)
		fs = append(fs, surprisal.File{Path: rel, Data: b})
		return err
	})
	sort.Slice(fs, func(i, j int) bool { return fs[i].Path < fs[j].Path })
	return fs, err
}

func sameFiles(a, b []surprisal.File) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Path != b[i].Path || !bytes.Equal(a[i].Data, b[i].Data) {
			return false
		}
	}
	return true
}

func kib(n int) string { return fmt.Sprintf("%.1f", float64(n)/1024) }

func pct(a, b int) string {
	if b == 0 {
		return "–"
	}
	return fmt.Sprintf("%+.1f%%", 100*(float64(a)/float64(b)-1))
}

// report: totals for every method over the games that have a baseline, then (perGame) one row per game.
func report(m surprisal.ModelInfo, level int, device string, year int, rows []row, skipped int, took time.Duration, perGame bool) string {
	var b strings.Builder
	fmt.Fprintf(&b, "### surprisal -c%d (%s) on the js13k %d games\n\n", level, m.Name, year)
	in, arch := 0, 0
	secs := 0.0
	for _, r := range rows {
		in += r.in
		arch += r.archive
		secs += r.secs
	}
	fmt.Fprintf(&b, "%d games (%d skipped), %s KiB unpacked, on %s, %.0f s compressing (%.0f s in all).\n\n",
		len(rows), skipped, kib(in), device, secs, took.Seconds())
	// totals over the games results.json also has, so every column counts the same games
	type col struct {
		name string
		get  func(r row) int
	}
	cols := []col{
		{"submitted .zip", func(r row) int { return r.base.Zip }},
		{"best legal zip per game", func(r row) int { return r.base.ZBest }},
		{"brotli -q11", func(r row) int { return r.base.Brotli }},
		{"brotli + trained dict", func(r row) int { return r.base.BrotliDict }},
		{"context mixing, trained on 2020-24", func(r row) int { return r.base.M5 }},
		{fmt.Sprintf("**surprisal -c%d**", level), func(r row) int { return r.archive }},
	}
	var based []row
	for _, r := range rows {
		if r.base != nil {
			based = append(based, r)
		}
	}
	if len(based) > 0 {
		sum := func(c col) int {
			t := 0
			for _, r := range based {
				t += c.get(r)
			}
			return t
		}
		zb := sum(cols[1])
		fmt.Fprintf(&b, "| method (%d games) | total KiB | vs best legal zip |\n|---|---:|---:|\n", len(based))
		for _, c := range cols {
			t := sum(c)
			fmt.Fprintf(&b, "| %s | %s | %s |\n", c.name, kib(t), pct(t, zb))
		}
		ratios := make([]float64, len(based))
		wins := 0
		for i, r := range based {
			ratios[i] = float64(r.archive) / float64(r.base.ZBest)
			best := min(r.base.ZBest, r.base.BrotliDict, r.base.M5, r.base.Brotli)
			if r.archive < best {
				wins++
			}
		}
		sort.Float64s(ratios)
		fmt.Fprintf(&b, "\nMedian archive / best legal zip: %.3f. Smaller than every other method on %d of %d games.\n",
			ratios[len(ratios)/2], wins, len(based))
	} else {
		fmt.Fprintf(&b, "| method | total KiB |\n|---|---:|\n| unpacked | %s |\n| **surprisal -c%d** | %s |\n", kib(in), level, kib(arch))
	}
	if perGame {
		fmt.Fprintf(&b, "\n| game | files | unpacked | best legal zip | brotli + dict | surprisal | vs best zip | s |\n")
		fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---:|---:|---:|\n")
		for _, r := range rows {
			zb, bd := "–", "–"
			vs := "–"
			if r.base != nil {
				zb, bd, vs = fmt.Sprint(r.base.ZBest), fmt.Sprint(r.base.BrotliDict), pct(r.archive, r.base.ZBest)
			}
			fmt.Fprintf(&b, "| %s | %d | %d | %s | %s | %d | %s | %.1f |\n", r.slug, r.files, r.in, zb, bd, r.archive, vs, r.secs)
		}
	}
	return b.String()
}
