package surprisal

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The vectors in ../spec/vectors come from the reference implementation (reference/python/intref.py) and the GPU path. The
// models must be installed (SURPRISAL_HOME, default ~/.surprisal); tests skip the ones that are not.

func vec(t *testing.T, name string) []byte {
	b, err := os.ReadFile(filepath.Join("..", "spec", "vectors", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// openTest opens the codec for a level (the vectors name theirs).
func openTest(t *testing.T, level int) *Codec {
	m, _ := ModelForLevel(level)
	gpu := os.Getenv("SURPRISAL_TEST_GPU") // "off" tests the CPU path
	if !m.Present() {
		t.Skipf("level %d's model not installed in %s", level, ModelsDir())
	}
	c, err := Open(level, 0, gpu)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

// vecInput reads a vector input; "bjj3" is bjj.html three times (three segments).
func vecInput(t *testing.T, name string) []byte {
	if name == "bjj3" {
		b := vec(t, "bjj.html")
		return bytes.Repeat(b, 3)
	}
	return vec(t, name)
}

func checkVector(t *testing.T, level int, input, archive string) {
	c := openTest(t, level)
	want, a := vecInput(t, input), vec(t, archive)
	t0 := time.Now()
	h, got, err := c.Decompress(a, nil)
	if err != nil {
		t.Fatal(err)
	}
	if h.Kind != KindStream || !bytes.Equal(got, want) {
		t.Fatalf("%s: decompressed %d bytes, differ from %s", archive, len(got), input)
	}
	t.Logf("%s on %s: %d tokens decoded in %v", archive, c.Device(), h.Tokens, time.Since(t0).Round(time.Millisecond))
	if p, err := c.Compress(KindStream, want, nil); err != nil || !bytes.Equal(p, a) {
		t.Fatalf("%s: compressed %d bytes, not byte-identical to the vector (%d bytes)", input, len(p), len(a))
	}
}

func TestVectorTiny(t *testing.T)   { checkVector(t, 1, "tiny.html", "l1-tiny.13k") }
func TestVectorSlice(t *testing.T)  { checkVector(t, 1, "slice.html", "l1-slice.13k") }
func TestVector7BTiny(t *testing.T) { checkVector(t, 7, "tiny.html", "l7-tiny.13k") }

// TestVectorBJJL1 decodes the whole BJJ vector at level 1 (minutes; SURPRISAL_TEST_LONG=1).
func TestVectorBJJL1(t *testing.T) {
	if os.Getenv("SURPRISAL_TEST_LONG") == "" {
		t.Skip("set SURPRISAL_TEST_LONG=1")
	}
	checkVector(t, 1, "bjj.html", "l1-bjj.13k")
}

// vectorSet is spec/vectors' archives: file, level, input.
var vectorSet = []struct {
	file  string
	level int
	input string
}{
	{"l1-tiny.13k", 1, "tiny.html"},
	{"l1-slice.13k", 1, "slice.html"},
	{"l1-bjj.13k", 1, "bjj.html"},
	{"l3-bjj.13k", 3, "bjj.html"},
	{"l1-bjj3.13k", 1, "bjj3"},
	{"l4-bjj.13k", 4, "bjj.html"},
	{"l6-bjj.13k", 6, "bjj.html"},
	{"l7-tiny.13k", 7, "tiny.html"},
	{"l9-bjj.13k", 9, "bjj.html"},
}

// TestMakeVectors (SURPRISAL_MAKE_VECTORS=1) writes spec/vectors' archives, those whose level's model is installed,
// and prints each one's size, token count and SHA-256 for vectors.json. SURPRISAL_MAKE_ONLY=l1-bjj.13k,... limits it.
func TestMakeVectors(t *testing.T) {
	if os.Getenv("SURPRISAL_MAKE_VECTORS") == "" {
		t.Skip("set SURPRISAL_MAKE_VECTORS=1")
	}
	only := os.Getenv("SURPRISAL_MAKE_ONLY")
	for _, v := range vectorSet {
		if only != "" && !strings.Contains(","+only+",", ","+v.file+",") {
			continue
		}
		c := openTest(t, v.level)
		t0 := time.Now()
		a, err := c.Compress(KindStream, vecInput(t, v.input), nil)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join("..", "spec", "vectors", v.file), a, 0o644); err != nil {
			t.Fatal(err)
		}
		h, _, _ := ReadHeader(a)
		sum := sha256.Sum256(a)
		fmt.Printf("VECTOR %s level %d input %s: %d bytes, %d tokens, sha256 %x, %s, %v\n", v.file, v.level, v.input,
			len(a), h.Tokens, sum, c.Device(), time.Since(t0).Round(time.Millisecond))
	}
}

func TestFileSet(t *testing.T) {
	c := openTest(t, 1)
	fs := []File{
		{"README.md", []byte("# demo\n\nTwo files and an empty one.\n")},
		{"src/main.go", []byte("package main\n\nfunc main() { println(\"hi\") }\n")},
		{"src/empty.txt", nil},
		{"bin/blob", []byte{0xff, 0xfe, 0x00, 'a', 0xc3}}, // not UTF-8
	}
	a, err := c.CompressFiles(fs, nil)
	if err != nil {
		t.Fatal(err)
	}
	h, b, err := c.Decompress(a, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeFiles(b)
	if err != nil || h.Kind != KindFiles || len(got) != len(fs) {
		t.Fatalf("file set: %v, kind %#x, %d files", err, h.Kind, len(got))
	}
	for i := range fs {
		if got[i].Path != fs[i].Path || !bytes.Equal(got[i].Data, fs[i].Data) {
			t.Fatalf("file %d: %q differs", i, fs[i].Path)
		}
	}
	t.Logf("file set: %d files, %d bytes -> %d bytes", len(fs), len(b), len(a))
}

func TestStore(t *testing.T) {
	c, err := Open(0, 0, "")
	if err != nil {
		t.Fatal(err)
	}
	fs := []File{{"a.txt", []byte("hello")}, {"b/c.bin", []byte{0, 0xff}}}
	a, err := c.CompressFiles(fs, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a[:2], []byte{KindFiles, Version << 4}) {
		t.Fatalf("header % x", a[:2])
	}
	h, b, err := c.Decompress(a, nil)
	if err != nil || h.Level != 0 {
		t.Fatal(err, h)
	}
	got, err := DecodeFiles(b)
	if err != nil || len(got) != 2 || string(got[0].Data) != "hello" {
		t.Fatalf("store round trip: %v %v", err, got)
	}
}

func TestLevels(t *testing.T) {
	want := map[int]int{1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3}
	win := map[int]int{1: 2048, 2: 4096, 3: 0, 4: 2048, 5: 4096, 6: 0, 7: 4096, 8: 8192, 9: 0}
	for lv, id := range want {
		if m, ok := ModelForLevel(lv); !ok || m.ID != id {
			t.Errorf("level %d: model %d, want %d", lv, m.ID, id)
		}
		if w, sk := LevelWindow(lv); w != win[lv] || (w > 0 && sk != 16) {
			t.Errorf("level %d: window %d sink %d, want %d", lv, w, sk, win[lv])
		}
	}
	for _, lv := range []int{0, 10, -1} {
		if _, ok := ModelForLevel(lv); ok {
			t.Errorf("level %d has a model", lv)
		}
	}
	if _, _, err := ReadHeader([]byte{KindStream, 0x21, 0}); err == nil {
		t.Error("header version 2 accepted")
	}
}

func TestFileSetPaths(t *testing.T) {
	for _, p := range []string{"", "/etc/passwd", "../x", "a/../../x", "a\\b", "./a"} {
		if _, err := encodeFiles([]File{{p, nil}}); err == nil {
			t.Errorf("path %q accepted", p)
		}
	}
	if _, err := decodeFiles([]byte("a\n5\nab")); err == nil {
		t.Error("truncated file set accepted")
	}
}

func TestManifestMatchesSpec(t *testing.T) {
	spec, err := os.ReadFile(filepath.Join("..", "spec", "models-v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(spec, manifestJSON) {
		t.Fatal("surprisal/models-v1.json differs from spec/models-v1.json (copy it over)")
	}
}

// checkCompress: compressing gives the vector byte for byte (compression only: decoding the long vectors takes minutes).
func checkCompress(t *testing.T, level int, input, archive string) {
	c := openTest(t, level)
	want, a := vecInput(t, input), vec(t, archive)
	t0 := time.Now()
	p, err := c.Compress(KindStream, want, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(p, a) {
		t.Fatalf("%s: compressed %d bytes, not byte-identical to %s (%d bytes)", input, len(p), archive, len(a))
	}
	t.Logf("%s on %s: %d bytes -> %d, byte-identical, %v", input, c.Device(), len(want), len(p), time.Since(t0).Round(time.Millisecond))
}

func TestCompressBJJL1(t *testing.T)  { checkCompress(t, 1, "bjj.html", "l1-bjj.13k") }
func TestCompressBJJL3(t *testing.T)  { checkCompress(t, 3, "bjj.html", "l3-bjj.13k") }
func TestCompressBJJ3L1(t *testing.T) { checkCompress(t, 1, "bjj3", "l1-bjj3.13k") }
func TestCompressBJJL4(t *testing.T)  { checkCompress(t, 4, "bjj.html", "l4-bjj.13k") }
func TestCompressBJJL6(t *testing.T)  { checkCompress(t, 6, "bjj.html", "l6-bjj.13k") }
func TestCompressBJJL9(t *testing.T)  { checkCompress(t, 9, "bjj.html", "l9-bjj.13k") }

// TestTokenizeBJJTime: how long compression's tokenizing (with its round-trip check) takes on bjj.html.
func TestTokenizeBJJTime(t *testing.T) {
	c := openTest(t, 1)
	b := vec(t, "bjj.html")
	t0 := time.Now()
	ids := c.tokenize(b)
	t.Logf("bjj.html: %d tokens in %v", len(ids), time.Since(t0).Round(time.Millisecond))
}
