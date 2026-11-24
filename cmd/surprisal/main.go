// Command surprisal compresses files with a language model; any machine with surprisal unpacks them exactly.
//
//	surprisal -c9 /path/to/dir             # writes archive.13k unless -o is given; -c0 ... -c9 pick the level
//	surprisal -d foo.13k                   # writes ./foo/... unless -o is given
package main

import (
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strings"
	"time"

	"github.com/benallfree/surprisal/surprisal"
)

const usage = `surprisal compresses files with a language model; any machine with surprisal unpacks them exactly.

Usage:
  surprisal -c[0-9] [-o archive.13k] <files or directories>...
  surprisal -d [-o dir] [-f] <archive.13k>
  surprisal --list
  surprisal --version

Levels: -c0 stores without a model; -c1 to -c3 use Qwen2.5-Coder-0.5B, -c4 to -c6 the 1.5B, -c7 to -c9 the 7B
(smallest archives, slowest). -c alone is -c9. The archive records its level and header version, so -d needs
neither. A level's model is downloaded from Hugging Face and converted the first time it's needed.
Models live in $SURPRISAL_MODELS, else nollama's models directory (%s).

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
	compress := flag.Bool("c", false, "compress files and directories into one archive")
	decompress := flag.Bool("d", false, "decompress an archive")
	out := flag.String("o", "", "output: the archive (-c, default archive.13k) or the directory (-d, default the archive's name)")
	force := flag.Bool("f", false, "overwrite existing files")
	level := flag.Int("level", 9, "compression level 0-9 (-c5 is -c -level 5)")
	list := flag.Bool("list", false, "list the levels and whether each one's model is installed")
	threads := flag.Int("threads", 0, "CPU threads (default: all CPUs)")
	gpu := flag.String("gpu", "auto", "auto (a GPU when one works) or off (the CPU); both give the same archives")
	showVersion := flag.Bool("version", false, "print the version")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, usage, surprisal.ModelsDir())
		flag.PrintDefaults()
	}
	os.Args = append(os.Args[:1], expandLevel(os.Args[1:])...)
	flag.Parse()
	if *showVersion {
		fmt.Println("surprisal", version)
		return
	}
	var err error
	switch {
	case *list:
		cmdList()
	case *compress && !*decompress:
		err = cmdCompress(flag.Args(), *out, *level, *threads, *gpu)
	case *decompress && !*compress && flag.NArg() == 1:
		err = cmdDecompress(flag.Arg(0), *out, *force, *threads, *gpu)
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		log.Fatal("surprisal: ", strings.TrimPrefix(err.Error(), "surprisal: "))
	}
}

// expandLevel turns -c<N> (or --c<N>) into -c -level=<N>, so the levels read like gzip's.
func expandLevel(args []string) []string {
	var out []string
	for i, a := range args {
		if a == "--" {
			return append(out, args[i:]...)
		}
		if t := strings.TrimPrefix(strings.TrimPrefix(a, "-"), "-"); t != a && len(t) == 2 && t[0] == 'c' && t[1] >= '0' && t[1] <= '9' {
			out = append(out, "-c", "-level="+t[1:])
			continue
		}
		out = append(out, a)
	}
	return out
}

func cmdList() {
	fmt.Println("level 0      store, no model")
	for _, m := range surprisal.Models() {
		lv := surprisal.LevelsFor(m.ID)
		if len(lv) == 0 {
			continue
		}
		state := "not installed (downloaded on first use)"
		if m.Present() {
			state = "installed"
		}
		fmt.Printf("level %d-%d    %-24s  %4.1f GB  %s\n", lv[0], lv[len(lv)-1], m.Source.Repo, float64(m.Bundle.Bytes)/1e9, state)
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
	fmt.Fprintf(os.Stderr, "level %d uses %s, not installed yet: downloading %s from Hugging Face and converting it (once, %.1f GB in %s)\n",
		level, m.Name, m.Source.Repo, float64(m.Bundle.Bytes)/1e9, surprisal.ModelsDir())
	t0 := time.Now()
	var last time.Time
	err := m.Install(func(format string, a ...any) {
		if time.Since(last) > 500*time.Millisecond {
			last = time.Now()
			fmt.Fprintf(os.Stderr, "\r"+format+"            ", a...)
		}
	})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s: installed, sha256 checked, %.0f s\n", m.Path(), time.Since(t0).Seconds())
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

func cmdCompress(args []string, out string, level, threads int, gpu string) error {
	files, err := gather(args)
	if err != nil {
		return err
	}
	if err := ensure(level); err != nil {
		return err
	}
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
	fmt.Fprintf(os.Stderr, "%s: %d files, %d -> %d bytes (level %d, %.1f s)\n", out, len(files), in, len(a), level,
		time.Since(p.start).Seconds())
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
	for _, f := range files {
		if err := write(filepath.Join(out, filepath.FromSlash(f.Path)), f.Data); err != nil {
			return err
		}
	}
	fmt.Fprintf(os.Stderr, "%s/: %d files, %d bytes (%.1f s)\n", out, len(files), len(data), time.Since(p.start).Seconds())
	return nil
}
