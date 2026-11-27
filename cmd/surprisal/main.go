// Command surprisal compresses files with a language model; any machine with surprisal unpacks them exactly.
//
//	surprisal -9 /path/to/dir              # writes dir.13k unless -o is given; -0 ... -9 pick the level (default -1)
//	surprisal foo.13k                      # unpacks into ./foo/... unless -o is given
package main

import (
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/benallfree/surprisal/surprisal"
)

const usage = `surprisal compresses files with a language model; any machine with surprisal unpacks them exactly.

Usage:
  surprisal [-0 ... -9] [--compare] [--stop-at=size] [-o out.13k] <files or directories>...   compress
  surprisal [-o dir] [-f] <archive.13k>                                      unpack
  surprisal --list
  surprisal --version

One argument ending in .13k is unpacked; anything else is compressed into one archive (<name>.13k for a single
input, archive.13k for several).

Levels: -0 stores without a model; -1 (the default) to -3 use the 0.5B model, -4 to -6 the 1.5B, -7 to -9 the 7B.
Within a model, a lower level attends to a shorter window of the text: faster, slightly bigger archives. -9 gives
the smallest archives, slowest. The archive records its level, so unpacking doesn't need it. A level's model is
downloaded the first time it's needed.
--stop-at=13312 (or 13k) gives up as soon as the archive is certain to be bigger than that many bytes: it writes
nothing and exits with status 3, so a script can try the next level. It never stops an input that would fit.
Models live in $SURPRISAL_HOME, else ~/.surprisal (%s).

Flags:
`

// version is set at release builds (-ldflags "-X main.version=..."); otherwise it is the module version Go recorded
// (the tag for go install, the git tag or pseudo-version for a build in a checkout), or "dev" when there is none.
var version = "dev"

func init() {
	if version != "dev" {
		return
	}
	if bi, ok := debug.ReadBuildInfo(); ok && bi.Main.Version != "" && bi.Main.Version != "(devel)" {
		version = strings.TrimPrefix(bi.Main.Version, "v")
	}
}

func main() {
	log.SetFlags(0)
	out := flag.String("o", "", "output: the archive when compressing, the directory when unpacking (default: the input's name)")
	force := flag.Bool("f", false, "overwrite existing files")
	list := flag.Bool("list", false, "list the levels and whether each one's model is installed")
	compare := flag.Bool("compare", false, "when compressing, also compress with each zip method, 7z (LZMA2 and PPMd), gzip, xz and Brotli and show their sizes")
	threads := flag.Int("threads", 0, "CPU threads (default: all CPUs)")
	gpu := flag.String("gpu", "auto", "auto (a GPU when one works) or off (the CPU); both give the same archives")
	showVersion := flag.Bool("version", false, "print the version")
	stopAt := flag.String("stop-at", "", "when compressing, stop and exit 3 once the archive is certain to be over this many bytes (13312, 13k)")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, usage, surprisal.ModelsDir())
		flag.PrintDefaults()
	}
	args, level := takeLevel(os.Args[1:])
	if err := flag.CommandLine.Parse(args); err != nil {
		os.Exit(2)
	}
	if *showVersion {
		fmt.Println("surprisal", version)
		return
	}
	var err error
	switch {
	case *list:
		cmdList()
	case flag.NArg() == 1 && isArchive(flag.Arg(0)):
		if level >= 0 {
			err = errors.New("a level (-0 ... -9) is for compressing; an archive records its own")
			break
		}
		if *compare {
			err = errors.New("--compare is for compressing, not unpacking")
			break
		}
		if *stopAt != "" {
			err = errors.New("--stop-at is for compressing, not unpacking")
			break
		}
		err = cmdDecompress(flag.Arg(0), *out, *force, *threads, *gpu)
	case flag.NArg() > 0:
		if level < 0 {
			level = 1 // the default
		}
		limit := -1
		if *stopAt != "" {
			if limit, err = parseSize(*stopAt); err != nil {
				break
			}
		}
		err = cmdCompress(flag.Args(), *out, level, *threads, *gpu, *compare, limit)
	default:
		flag.Usage()
		os.Exit(2)
	}
	var over *surprisal.OverLimit
	if errors.As(err, &over) {
		// its own status, so a script can tell "too big" from "failed"
		fmt.Fprintln(os.Stderr, "surprisal:", strings.TrimPrefix(err.Error(), "surprisal: ")+"; no archive written")
		os.Exit(exitOverLimit)
	}
	if err != nil {
		log.Fatal("surprisal: ", strings.TrimPrefix(err.Error(), "surprisal: "))
	}
}

// exitOverLimit is the exit status when --stop-at's limit is exceeded (failures exit 1, usage errors 2).
const exitOverLimit = 3

// parseSize reads a size in bytes: 13312, or with k (1024 bytes, as js13k's 13k = 13312).
func parseSize(s string) (int, error) {
	t, mult := strings.ToLower(strings.TrimSpace(s)), 1
	if strings.HasSuffix(t, "k") {
		t, mult = strings.TrimSuffix(t, "k"), 1024
	}
	n, err := strconv.Atoi(t)
	if err != nil || n <= 0 || n > math.MaxInt/mult {
		return 0, fmt.Errorf("--stop-at=%s: give a size in bytes, like 13312 or 13k", s)
	}
	return n * mult, nil
}

// takeLevel removes the level flags (-0 ... -9, as gzip's) from args and returns the last one given, or -1 for none.
func takeLevel(args []string) ([]string, int) {
	var out []string
	level := -1
	for i, a := range args {
		if a == "--" {
			return append(out, args[i:]...), level
		}
		if len(a) == 2 && a[0] == '-' && a[1] >= '0' && a[1] <= '9' {
			level = int(a[1] - '0')
			continue
		}
		out = append(out, a)
	}
	return out, level
}

// isArchive reports whether the one argument names an archive to unpack: an existing file ending in .13k.
func isArchive(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && fi.Mode().IsRegular() && strings.HasSuffix(path, ".13k")
}

func cmdList() {
	fmt.Println("level 0  store, no model")
	for lv := 1; lv <= 9; lv++ {
		m, ok := surprisal.ModelForLevel(lv)
		if !ok {
			continue
		}
		state := "not installed (downloaded on first use)"
		if m.Present() {
			state = "installed"
		}
		win := "full context"
		if w, _ := surprisal.LevelWindow(lv); w > 0 {
			win = fmt.Sprintf("window %d", w)
		}
		fmt.Printf("level %d  %-26s  %4.1f GB  %-12s  %s\n", lv, m.Name, float64(m.Bundle.Bytes)/1e9, win, state)
	}
}

// ensure installs a level's model the first time it's needed.
func ensure(level int) error {
	if level == 0 {
		return nil
	}
	m, ok := surprisal.ModelForLevel(level)
	if !ok {
		return fmt.Errorf("compression level %d isn't 0-9", level)
	}
	if m.Present() {
		return nil
	}
	fmt.Fprintf(os.Stderr, "level %d uses %s, not installed yet: downloading it (once, %.1f GB in %s)\n",
		level, m.Name, float64(m.Bundle.Bytes)/1e9, surprisal.ModelsDir())
	t0 := time.Now()
	bar := &downloadBar{tty: isTerminal(os.Stderr)}
	var last time.Time
	err := m.Install(func(format string, a ...any) {
		if strings.HasPrefix(format, "convert: ") { // the conversion's steps: frequent, so throttled, in place
			if time.Since(last) > 500*time.Millisecond {
				last = time.Now()
				fmt.Fprintf(os.Stderr, "\r"+format+"\033[K", a...)
			}
			return
		}
		bar.finish() // anything else (a URL that failed, the fallback) gets its own line
		fmt.Fprintf(os.Stderr, format+"\n", a...)
	}, bar.update)
	bar.finish()
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: installed, sha256 checked, %.0f s\n", m.Path(), time.Since(t0).Seconds())
	return nil
}

// downloadBar draws a download's progress: a bar redrawn in place on a terminal, a plain line every few seconds when
// stderr is a file or a pipe.
type downloadBar struct {
	tty   bool
	file  string
	start time.Time
	last  time.Time
	open  bool // a bar is on screen without its final newline
}

func (b *downloadBar) update(file string, done, total int64) {
	now := time.Now()
	if file != b.file {
		b.finish()
		b.file, b.start, b.last = file, now, time.Time{}
		fmt.Fprintf(os.Stderr, "downloading %s\n", file) // the name gets its own line, so the bar line stays short
	}
	every := 100 * time.Millisecond
	if !b.tty {
		every = 5 * time.Second
	}
	if done < total && now.Sub(b.last) < every {
		return
	}
	b.last = now
	secs := max(now.Sub(b.start).Seconds(), 1e-9)
	rate := float64(done) / secs
	const mb = 1 << 20
	var stats string
	if total > 0 {
		stats = fmt.Sprintf("%3d%%  %d/%d MB", done*100/total, done/mb, total/mb)
	} else {
		stats = fmt.Sprintf("%d MB", done/mb)
	}
	if secs < 1 { // too early for a meaningful speed
		rate = 0
	} else {
		stats += fmt.Sprintf("  %.1f MB/s", rate/mb)
	}
	if total > 0 && done < total && rate > 0 {
		left := time.Duration(float64(total-done)/rate) * time.Second
		stats += fmt.Sprintf("  %s left", left.Round(time.Second))
	}
	// Fit the line to the terminal: a line that wraps can't be redrawn in place with \r.
	barWidth, room := 30, 0
	if b.tty {
		if w := termWidth(os.Stderr); w > 0 {
			room = w - 1 // never touch the last column, which wraps on some terminals
			barWidth = min(barWidth, room-len(stats)-3)
		}
	}
	line := stats
	if total > 0 && barWidth >= 5 {
		n := int(done * int64(barWidth) / total)
		line = "[" + strings.Repeat("█", n) + strings.Repeat("░", barWidth-n) + "] " + stats
	}
	if room > 0 && len([]rune(line)) > room {
		line = string([]rune(line)[:room])
	}
	if b.tty {
		fmt.Fprintf(os.Stderr, "\r%s\033[K", line)
		b.open = true
	} else {
		fmt.Fprintln(os.Stderr, line)
	}
	if total > 0 && done >= total {
		b.finish()
	}
}

// finish ends the bar's line, if one is open.
func (b *downloadBar) finish() {
	if b.open {
		fmt.Fprintln(os.Stderr)
		b.open = false
	}
}

// isTerminal reports whether f is a terminal (a character device), so progress can be redrawn in place.
func isTerminal(f *os.File) bool {
	fi, err := f.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

type progress struct {
	what  string
	start time.Time
	last  time.Time
	hist  []progressMark // recent updates, for the current rate
	done  bool           // the line has its newline
}

type progressMark struct {
	t    time.Time
	done int
}

// end finishes the progress line if it was left open (a run that stopped early).
func (p *progress) end() {
	if !p.last.IsZero() && !p.done {
		fmt.Fprintln(os.Stderr)
		p.done = true
	}
}

// progressSpan is how far back the rate looks: speed changes within a run (attention costs more later in a segment,
// and drops back at each new segment), so the average since the start would be wrong for the time left.
const progressSpan = 10 * time.Second

func (p *progress) update(done, total int) {
	now := time.Now()
	if len(p.hist) == 0 {
		p.hist = append(p.hist, progressMark{p.start, 0})
	}
	p.hist = append(p.hist, progressMark{now, done})
	for len(p.hist) > 2 && now.Sub(p.hist[1].t) >= progressSpan {
		p.hist = p.hist[1:]
	}
	if done < total && now.Sub(p.last) < time.Second {
		return
	}
	p.last = now
	h := p.hist[0]
	rate := float64(done-h.done) / max(now.Sub(h.t).Seconds(), 1e-9)
	left := time.Duration(float64(total-done)/max(rate, 1e-9)) * time.Second
	if done == total {
		rate = float64(done) / max(now.Sub(p.start).Seconds(), 1e-9) // the whole run's average
		left = 0
	}
	fmt.Fprintf(os.Stderr, "\r%s: %d/%d tokens, %.1f tok/s, %s left    ", p.what, done, total, rate, left.Round(time.Second))
	if done == total {
		fmt.Fprintln(os.Stderr)
		p.done = true
	}
}

// gather expands the arguments (directories recursively) into files named relative to their common directory.
func gather(args []string) ([]surprisal.File, error) {
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
	var files []surprisal.File
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
		files = append(files, surprisal.File{Path: rel, Data: data})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files, nil
}

func cmdCompress(args []string, out string, level, threads int, gpu string, compare bool, limit int) error {
	files, err := gather(args)
	if err != nil {
		return err
	}
	if err := ensure(level); err != nil {
		return err
	}
	t0 := time.Now() // after any first-time download, as --compare times it
	c, err := surprisal.Open(level, threads, gpu)
	if err != nil {
		return err
	}
	defer c.Close()
	if level > 0 {
		fmt.Fprintf(os.Stderr, "level %d: %s on %s\n", level, c.Info.Name, c.Device())
	}
	if out == "" {
		out = "archive.13k"
		if len(args) == 1 {
			if abs, err := filepath.Abs(args[0]); err == nil && filepath.Base(abs) != string(filepath.Separator) {
				out = filepath.Base(abs) + ".13k" // "src/" and "." name the archive after the folder
			}
		}
	}
	p := &progress{what: "compress", start: time.Now()}
	if limit >= 0 {
		c.Check = surprisal.StopAt(limit)
	}
	a, err := c.CompressFiles(files, p.update)
	if err != nil {
		p.end()
		return err
	}
	if err := os.WriteFile(out, a, 0o644); err != nil {
		return err
	}
	in := 0
	for _, f := range files {
		in += len(f.Data)
	}
	mine := row{label: fmt.Sprintf("surprisal -%d", level), size: len(a), took: time.Since(t0)}
	var rows []row
	if compare {
		if rows, err = compareRows(files); err != nil {
			return err
		}
	}
	printSizes(in, append(rows, mine))
	noun := "files"
	if len(files) == 1 {
		noun = "file"
	}
	fmt.Printf("wrote %s (%d %s)\n", out, len(files), noun)
	return nil
}

func cmdDecompress(archive, out string, force bool, threads int, gpu string) error {
	a, err := os.ReadFile(archive)
	if err != nil {
		return err
	}
	h, _, err := surprisal.ReadHeader(a)
	if err != nil {
		return err
	}
	if err := ensure(h.Level); err != nil {
		return err
	}
	c, err := surprisal.Open(h.Level, threads, gpu)
	if err != nil {
		return err
	}
	defer c.Close()
	if h.Level > 0 {
		fmt.Fprintf(os.Stderr, "level %d: %s on %s\n", h.Level, c.Info.Name, c.Device())
	}
	name := strings.TrimSuffix(filepath.Base(archive), ".13k")
	defaultOut := out == ""
	if defaultOut {
		out = name
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
	if h.Kind == surprisal.KindStream {
		if err := write(out, data); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s: %d bytes (%.1f s)\n", out, len(data), time.Since(p.start).Seconds())
		return nil
	}
	files, err := surprisal.DecodeFiles(data)
	if err != nil {
		return err
	}
	if defaultOut && len(files) == 1 && files[0].Path == name {
		// one file compressed on its own (readme.md -> readme.md.13k): unpack it back to readme.md, as gunzip would
		if err := write(name, files[0].Data); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s: %d bytes (%.1f s)\n", name, len(files[0].Data), time.Since(p.start).Seconds())
		return nil
	}
	for _, f := range files {
		if err := write(filepath.Join(out, filepath.FromSlash(f.Path)), f.Data); err != nil {
			return err
		}
	}
	fmt.Fprintf(os.Stderr, "%s/: %d files, %d bytes (%.1f s)\n", out, len(files), len(data), time.Since(p.start).Seconds())
	return nil
}
