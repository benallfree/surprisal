package surprisal

import (
	"math/rand"
	"testing"
)

// TestMinBytes: minBytes is exactly what finish would return at that point, and never more than the final size,
// so a size limit checked against it never stops an input that would have fitted.
func TestMinBytes(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		e := newEncoder()
		var mins []int
		n := r.Intn(3000)
		for i := 0; i < n; i++ {
			total := uint64(1 << 28)
			var lo, hi uint64
			switch r.Intn(3) {
			case 0: // a near-certain token: costs almost nothing, piles up pending bits
				lo, hi = 0, total-uint64(r.Intn(200000))-1
			case 1: // a tiny interval
				lo = uint64(r.Int63n(int64(total - 2)))
				hi = lo + 1
			default:
				lo = uint64(r.Int63n(int64(total - 1)))
				hi = lo + 1 + uint64(r.Int63n(int64(total-lo-1)))
			}
			e.code(total, hi, lo)
			m := e.minBytes()
			if i > 0 && m < mins[len(mins)-1] {
				t.Fatalf("trial %d step %d: minBytes went down, %d -> %d", trial, i, mins[len(mins)-1], m)
			}
			mins = append(mins, m)
			if i%97 == 0 { // finishing here gives exactly minBytes
				c := *e
				c.out = append([]byte(nil), e.out...)
				if got := len(c.finish()); got != m {
					t.Fatalf("trial %d step %d: finish gave %d bytes, minBytes said %d", trial, i, got, m)
				}
			}
		}
		final := e.minBytes()
		if got := len(e.finish()); got != final {
			t.Fatalf("trial %d: finish gave %d bytes, minBytes said %d", trial, got, final)
		}
		for i, m := range mins {
			if m > final {
				t.Fatalf("trial %d step %d: minBytes %d > final %d", trial, i, m, final)
			}
		}
	}
}

// TestStopAt: the limit applies to the archive's size; equal fits, one over stops.
func TestStopAt(t *testing.T) {
	if err := StopAt(100)(10, 20, 100); err != nil {
		t.Fatal("100 bytes fits a 100-byte limit")
	}
	err := StopAt(100)(10, 20, 101)
	o, ok := err.(*OverLimit)
	if !ok || o.Done != 10 || o.Total != 20 || o.MinSize != 101 || o.Limit != 100 {
		t.Fatalf("got %v", err)
	}
	c := &Codec{Check: StopAt(5)}
	if _, err := c.Compress(KindStream, []byte("hello"), nil); err == nil {
		t.Fatal("a 7-byte stored archive should be over a 5-byte limit")
	}
	c.Check = StopAt(7)
	if _, err := c.Compress(KindStream, []byte("hello"), nil); err != nil {
		t.Fatal(err)
	}
}
