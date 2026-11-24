package surprisal

import (
	"bytes"
	"os"
	"path/filepath"
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

// openTest opens the codec for the first level that uses model id (the vectors are made at that level).
func openTest(t *testing.T, id int) *Codec {
	m, _ := ModelByID(id)
	gpu := os.Getenv("SURPRISAL_TEST_GPU") // "off" tests the CPU path
	if !m.Present() {
		t.Skipf("model %d not installed in %s", id, ModelsDir())
	}
	c, err := Open(LevelsFor(id)[0], 0, gpu)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func checkVector(t *testing.T, id int, input, archive string) {
	c := openTest(t, id)
	want, a := vec(t, input), vec(t, archive)
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

func TestVectorTiny(t *testing.T)   { checkVector(t, 1, "tiny.html", "m1-tiny.13k") }
func TestVectorSlice(t *testing.T)  { checkVector(t, 1, "slice.html", "m1-slice.13k") }
func TestVector7BTiny(t *testing.T) { checkVector(t, 3, "tiny.html", "m3-tiny.13k") }

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
	for lv, id := range want {
		if m, ok := ModelForLevel(lv); !ok || m.ID != id {
			t.Errorf("level %d: model %d, want %d", lv, m.ID, id)
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
func checkCompress(t *testing.T, id int, input, archive string) {
	c := openTest(t, id)
	want, a := vec(t, input), vec(t, archive)
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

func TestCompressBJJ05B(t *testing.T) { checkCompress(t, 1, "bjj.html", "m1-bjj.13k") }
func TestCompressBJJ15B(t *testing.T) { checkCompress(t, 2, "bjj.html", "m2-bjj.13k") }
func TestCompressBJJ7B(t *testing.T)  { checkCompress(t, 3, "bjj.html", "m3-bjj.13k") }

// TestTokenizeBJJTime: how long compression's tokenizing (with its round-trip check) takes on bjj.html.
func TestTokenizeBJJTime(t *testing.T) {
	c := openTest(t, 1)
	b := vec(t, "bjj.html")
	t0 := time.Now()
	ids := c.tokenize(b)
	t.Logf("bjj.html: %d tokens in %v", len(ids), time.Since(t0).Round(time.Millisecond))
}
