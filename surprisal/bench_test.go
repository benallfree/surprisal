package surprisal

import (
	"bytes"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestBench measures each level on one input: archive bytes and compression speed, and optionally decompression
// speed on a prefix. Skipped unless SURPRISAL_BENCH_IN is set.
//
//	SURPRISAL_BENCH_IN      the input file
//	SURPRISAL_BENCH_BYTES   use only its first n bytes
//	SURPRISAL_BENCH_LEVELS  comma-separated levels (default 1)
//	SURPRISAL_BENCH_DECODE  also decompress an archive of the first n bytes and check the round trip
//	SURPRISAL_TEST_GPU      "off" for the CPU
func TestBench(t *testing.T) {
	in := os.Getenv("SURPRISAL_BENCH_IN")
	if in == "" {
		t.Skip("set SURPRISAL_BENCH_IN")
	}
	data, err := os.ReadFile(in)
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := strconv.Atoi(os.Getenv("SURPRISAL_BENCH_BYTES")); n > 0 && n < len(data) {
		data = data[:n]
	}
	levels := os.Getenv("SURPRISAL_BENCH_LEVELS")
	if levels == "" {
		levels = "1"
	}
	dn, _ := strconv.Atoi(os.Getenv("SURPRISAL_BENCH_DECODE"))
	for _, ls := range strings.Split(levels, ",") {
		lv, err := strconv.Atoi(ls)
		if err != nil {
			t.Fatal(err)
		}
		c, err := Open(lv, 0, os.Getenv("SURPRISAL_TEST_GPU"))
		if err != nil {
			t.Fatal(err)
		}
		t0 := time.Now()
		a, err := c.Compress(KindStream, data, nil)
		if err != nil {
			t.Fatal(err)
		}
		d := time.Since(t0)
		h, _, _ := ReadHeader(a)
		line := fmt.Sprintf("BENCH level %d (%s): %d bytes, %d tokens -> %d bytes; compress %.1f s, %.1f tok/s",
			lv, c.Device(), len(data), h.Tokens, len(a), d.Seconds(), float64(h.Tokens)/d.Seconds())
		fmt.Println(line)
		if dn > 0 {
			src := data[:min(dn, len(data))]
			a2, err := c.Compress(KindStream, src, nil)
			if err != nil {
				t.Fatal(err)
			}
			t0 = time.Now()
			h2, got, err := c.Decompress(a2, nil)
			if err != nil {
				t.Fatal(err)
			}
			dd := time.Since(t0)
			line = fmt.Sprintf("BENCH level %d decompress %d tokens %.1f s, %.1f tok/s, round trip %v", lv, h2.Tokens, dd.Seconds(),
				float64(h2.Tokens)/dd.Seconds(), bytes.Equal(got, src))
			fmt.Println(line)
		}
		c.Close()
	}
}
