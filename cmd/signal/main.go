// Command signal compresses files with a language model; any machine with signal unpacks them exactly.
//
//	signal -c /path/to/dir/**/*.cpp     # writes archive.13k unless -o is given
//	signal -d foo.13k                   # writes ./foo/... unless -o is given
//	signal --setup best                 # once: download Qwen2.5-Coder-7B and convert it (sha256 checked)
package main

import (
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/benallfree/compress13k/signal"
)

const usage = `signal compresses files with a language model; any machine with signal unpacks them exactly.

Usage:
  signal -c [-o archive.13k] [--profile best] <files or directories>...
  signal -d [-o dir] [-f] <archive.13k>
  signal --setup good|better|best [--from <snapshot dir>]
  signal --list

Profiles: good (Qwen2.5-Coder-0.5B), better (1.5B), best (7B, the default; smallest archives, slowest).
Models live in $SIGNAL_MODELS, else nollama's models directory (%s).

Flags:
`

func main() {
	log.SetFlags(0)
	compress := flag.Bool("c", false, "compress files and directories into one archive")
	decompress := flag.Bool("d", false, "decompress an archive")
	out := flag.String("o", "", "output: the archive (-c, default archive.13k) or the directory (-d, default the archive's name)")
	force := flag.Bool("f", false, "overwrite existing files")
	profile := flag.String("profile", "best", "good, better or best")
	setup := flag.String("setup", "", "download and convert a profile's model: good, better or best")
	from := flag.String("from", "", "with --setup: convert this local copy of the official weights instead of downloading")
	list := flag.Bool("list", false, "list the models and whether each is installed")
	threads := flag.Int("threads", 0, "threads (default: all CPUs)")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, usage, signal.ModelsDir())
		flag.PrintDefaults()
	}
	flag.Parse()
	var err error
	switch {
	case *setup != "":
		err = cmdSetup(*setup, *from)
	case *list:
		cmdList()
	case *compress && !*decompress:
		err = cmdCompress(flag.Args(), *out, *profile, *threads)
	case *decompress && !*compress && flag.NArg() == 1:
		err = cmdDecompress(flag.Arg(0), *out, *force, *threads)
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		log.Fatal("signal: ", err)
	}
}

func cmdList() {
	for _, m := range signal.Models() {
		state := "not installed (signal --setup " + m.Profile + ")"
		if m.Present() {
			state = "installed"
		}
		fmt.Printf("%d  %-6s  %-24s  %5.1f GB  %s\n", m.ID, m.Profile, m.Source.Repo, float64(m.Bundle.Bytes)/1e9, state)
	}
}

func cmdSetup(profile, from string) error {
	m, ok := signal.ModelByProfile(profile)
	if !ok {
		return fmt.Errorf("unknown profile %q", profile)
	}
	t0 := time.Now()
	var last time.Time
	err := m.Setup(from, func(format string, a ...any) {
		if time.Since(last) > 500*time.Millisecond {
			last = time.Now()
			fmt.Fprintf(os.Stderr, "\r"+format+"            ", a...)
		}
	})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return err
	}
	fmt.Printf("%s (%s): installed, sha256 checked, %.0f s\n", m.Path(), m.Profile, time.Since(t0).Seconds())
	return nil
}

type progress struct {
	what  string
	start time.Time
	last  time.Time
}

func (p *progress) update(done, total int) {
	now := time.Now()
	if done < total && now.Sub(p.last) < time.Second {
		return
	}
	p.last = now
	rate := float64(done) / max(now.Sub(p.start).Seconds(), 1e-9)
	left := time.Duration(float64(total-done)/max(rate, 1e-9)) * time.Second
	fmt.Fprintf(os.Stderr, "\r%s: %d/%d tokens, %.1f tok/s, %s left    ", p.what, done, total, rate, left.Round(time.Second))
	if done == total {
		fmt.Fprintln(os.Stderr)
	}
}

// gather expands the arguments (directories recursively) into files named relative to their common directory.
func gather(args []string) ([]signal.File, error) {
	var paths []string
	for _, a := range args {
		st, err := os.Stat(a)
		if err != nil {
			return nil, err
		}
		if !st.IsDir() {
			paths = append(paths, a)
			continue
		}
		err = filepath.WalkDir(a, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.Type().IsRegular() {
				paths = append(paths, p)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	if len(paths) == 0 {
		return nil, errors.New("no files to compress")
	}
	abs := make([]string, len(paths))
	for i, p := range paths {
		a, err := filepath.Abs(p)
		if err != nil {
			return nil, err
		}
		abs[i] = a
	}
	base := filepath.Dir(abs[0])
	for _, a := range abs[1:] {
		for !strings.HasPrefix(a, base+string(filepath.Separator)) && base != filepath.Dir(base) {
			base = filepath.Dir(base)
		}
	}
	seen := map[string]bool{}
	var files []signal.File
	for _, a := range abs {
		rel, err := filepath.Rel(base, a)
		if err != nil {
			return nil, err
		}
		rel = filepath.ToSlash(rel)
		if seen[rel] {
			continue
		}
		seen[rel] = true
		data, err := os.ReadFile(a)
		if err != nil {
			return nil, err
		}
		files = append(files, signal.File{Path: rel, Data: data})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files, nil
}

func cmdCompress(args []string, out, profile string, threads int) error {
	m, ok := signal.ModelByProfile(profile)
	if !ok {
		return fmt.Errorf("unknown profile %q", profile)
	}
	files, err := gather(args)
	if err != nil {
		return err
	}
	c, err := signal.Open(m, threads)
	if err != nil {
		return err
	}
	defer c.Close()
	if out == "" {
		out = "archive.13k"
	}
	p := &progress{what: "compress", start: time.Now()}
	a, err := c.CompressFiles(files, p.update)
	if err != nil {
		return err
	}
	if err := os.WriteFile(out, a, 0o644); err != nil {
		return err
	}
	in := 0
	for _, f := range files {
		in += len(f.Data)
	}
	fmt.Fprintf(os.Stderr, "%s: %d files, %d -> %d bytes (%s, %.1f s)\n", out, len(files), in, len(a), m.Profile,
		time.Since(p.start).Seconds())
	return nil
}

func cmdDecompress(archive, out string, force bool, threads int) error {
	a, err := os.ReadFile(archive)
	if err != nil {
		return err
	}
	h, _, err := signal.ReadHeader(a)
	if err != nil {
		return err
	}
	m, ok := signal.ModelByID(h.Model)
	if !ok {
		return fmt.Errorf("the archive needs model %d, which this signal doesn't know (update signal)", h.Model)
	}
	c, err := signal.Open(m, threads)
	if err != nil {
		return err
	}
	defer c.Close()
	if out == "" {
		out = strings.TrimSuffix(filepath.Base(archive), ".13k")
	}
	p := &progress{what: "decompress", start: time.Now()}
	_, data, err := c.Decompress(a, p.update)
	if err != nil {
		return err
	}
	write := func(path string, b []byte) error {
		if _, err := os.Stat(path); err == nil && !force {
			return fmt.Errorf("%s exists (use -f to overwrite)", path)
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return err
		}
		return os.WriteFile(path, b, 0o644)
	}
	if h.Kind == signal.KindStream {
		if err := write(out, data); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s: %d bytes (%.1f s)\n", out, len(data), time.Since(p.start).Seconds())
		return nil
	}
	files, err := signal.DecodeFiles(data)
	if err != nil {
		return err
	}
	for _, f := range files {
		if err := write(filepath.Join(out, filepath.FromSlash(f.Path)), f.Data); err != nil {
			return err
		}
	}
	fmt.Fprintf(os.Stderr, "%s/: %d files, %d bytes (%.1f s)\n", out, len(files), len(data), time.Since(p.start).Seconds())
	return nil
}
