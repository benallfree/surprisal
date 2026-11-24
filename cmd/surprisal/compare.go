package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/benallfree/surprisal/surprisal"
)

// A classic compressor --compare runs: its label, the commands that provide it (the first one found is used), the
// Homebrew formula to suggest when none is, and how to run it. Stream tools read the inputs joined into one stream on
// stdin; archive tools (zip, 7z) get the inputs as files in dir and write an archive there.
type classic struct {
	label   string
	cmds    []string
	brew    string
	archive bool
	args    func(cmd string) []string // stream: the arguments; archive: the arguments before the file list
	out     string                    // archive: the archive's file name in dir
}

var classics = []classic{
	{label: "zip -9", cmds: []string{"zip"}, brew: "zip", archive: true, out: "out.zip",
		args: func(string) []string { return []string{"-9", "-q", "-X", "out.zip"} }},
	{label: "gzip -9", cmds: []string{"gzip"}, brew: "gzip",
		args: func(string) []string { return []string{"-9", "-c"} }},
	{label: "xz -9e", cmds: []string{"xz"}, brew: "xz",
		args: func(string) []string { return []string{"-9e", "-c"} }},
	{label: "7z -mx=9", cmds: []string{"7zz", "7z"}, brew: "sevenzip", archive: true, out: "out.7z",
		args: func(string) []string { return []string{"a", "-t7z", "-mx=9", "-bd", "-bso0", "-bsp0", "out.7z"} }},
	{label: "brotli -q11", cmds: []string{"brotli"}, brew: "brotli",
		args: func(string) []string { return []string{"-q", "11", "-w", "24", "-c"} }},
}

// compareRows compresses the files with each classic tool that's installed and returns a row per tool (a note in
// place of a size when the tool is missing or fails). Compressing with --compare adds these rows above its own.
func compareRows(files []surprisal.File) ([]row, error) {
	var joined []byte
	for _, f := range files {
		joined = append(joined, f.Data...)
	}

	// The archive tools get the same files, under the same names, in a scratch folder.
	dir, err := os.MkdirTemp("", "surprisal-compare-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	var names []string
	for _, f := range files {
		p := filepath.Join(dir, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return nil, err
		}
		if err := os.WriteFile(p, f.Data, 0o644); err != nil {
			return nil, err
		}
		names = append(names, filepath.FromSlash(f.Path))
	}

	var rows []row
	for _, c := range classics {
		cmd := ""
		for _, name := range c.cmds {
			if p, err := exec.LookPath(name); err == nil {
				cmd = p
				break
			}
		}
		if cmd == "" {
			rows = append(rows, row{label: c.label, note: "(brew install " + c.brew + ")"})
			continue
		}
		var size int
		t0 := time.Now()
		if c.archive {
			x := exec.Command(cmd, append(c.args(cmd), names...)...)
			x.Dir = dir
			if out, err := x.CombinedOutput(); err != nil {
				rows = append(rows, row{label: c.label, note: fmt.Sprintf("(failed: %v %s)", err, bytes.TrimSpace(out))})
				continue
			}
			fi, err := os.Stat(filepath.Join(dir, c.out))
			if err != nil {
				return nil, err
			}
			size = int(fi.Size())
		} else {
			x := exec.Command(cmd, c.args(cmd)...)
			x.Stdin = bytes.NewReader(joined)
			out, err := x.Output()
			if err != nil {
				rows = append(rows, row{label: c.label, note: fmt.Sprintf("(failed: %v)", err)})
				continue
			}
			size = len(out)
		}
		rows = append(rows, row{label: c.label, size: size, took: time.Since(t0)})
	}
	return rows, nil
}

// row is one line of the sizes table: a method and its compressed size and wall time, or a note when it couldn't run.
type row struct {
	label, note string
	size        int
	took        time.Duration
}

// printSizes prints the original size, then each row's size, ratio, change from the original and wall time. With more
// than one row, it marks the smallest.
func printSizes(orig int, rows []row) {
	best := -1
	for i, r := range rows {
		if r.note == "" && (best < 0 || r.size < rows[best].size) {
			best = i
		}
	}
	fmt.Printf("%-14s %9s\n", "original", commas(orig))
	for i, r := range rows {
		if r.note != "" {
			fmt.Printf("%-14s %s\n", r.label, r.note)
			continue
		}
		mark := ""
		if i == best && len(rows) > 1 {
			mark = "  smallest"
		}
		fmt.Printf("%-14s %9s  %6s  %-9s %8s%s\n", r.label, commas(r.size), ratio(r.size, orig), "("+pctChange(r.size, orig)+")",
			wall(r.took), mark)
	}
}

// wall formats a wall time: milliseconds under a second, else seconds, else minutes and seconds.
func wall(d time.Duration) string {
	switch {
	case d < time.Second:
		return fmt.Sprintf("%d ms", d.Milliseconds())
	case d < time.Minute:
		return fmt.Sprintf("%.1f s", d.Seconds())
	default:
		return d.Round(time.Second).String()
	}
}

// ratio is the compression ratio, original ÷ compressed, as benchmarks report it: "7.6×".
func ratio(size, orig int) string {
	if size == 0 {
		return ""
	}
	return fmt.Sprintf("%.1f×", float64(orig)/float64(size))
}

// pctChange is size's change from orig, signed: "-86.8%" (smaller) or "+0.3%" (larger).
func pctChange(size, orig int) string {
	if orig == 0 {
		return ""
	}
	return fmt.Sprintf("%+.1f%%", 100*float64(size-orig)/float64(orig))
}

// commas formats n with thousands separators.
func commas(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
