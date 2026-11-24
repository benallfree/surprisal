// A real LLM compressor on nollama: arithmetic coding driven by one-token forward passes (GPU when
// available), the same scheme as llm/realcodec.py.
//
//	go run . encode <model.gguf> <input> <archive.13k>
//	go run . decode <model.gguf> <archive.13k> <output>
//
// Archive: 4-byte token count, 4-byte window (0 = whole file), then a Witten-Neal-Cleary 32-bit stream. The first token is coded
// uniformly over the vocabulary; each later one with the model's distribution after the tokens before it.
// NLC_WINDOW=<tokens> sets the attention window when encoding. NOLLAMA_GPU=off runs on the CPU.
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"time"
	_ "unsafe"

	"github.com/benallfree/nollama/engine"
)

//go:linkname stepper github.com/benallfree/nollama/engine/internal/benchhooks.Stepper
var stepper func(m any, maxCtx int) func(toks []int, pos int) []float32

// predictor feeds tokens and returns the next distribution. With a window W, once the state holds 2W+1 tokens it
// restarts from the file's first token (an attention sink) plus the last W tokens, read as one batch, so every token
// sees between W and 2W tokens of context and per-step attention cost stops growing. Encoder and decoder make the
// same calls in the same order, so they get the same logits.
type predictor struct {
	feed func(toks []int, pos int) []float32
	w, n int
	hist []int
}

func newPredictor(m *engine.Model, w, ntok int) *predictor {
	maxCtx := ntok + 1
	if w > 0 {
		maxCtx = min(maxCtx, 2*w+2)
	}
	return &predictor{feed: stepper(m, maxCtx), w: w}
}

func (p *predictor) next(t int) []float32 {
	p.hist = append(p.hist, t)
	if p.w > 0 && p.n+1 > 2*p.w+1 {
		ctx := append([]int{p.hist[0]}, p.hist[len(p.hist)-p.w:]...)
		p.n = len(ctx)
		return p.feed(ctx, 0)
	}
	p.n++
	return p.feed([]int{t}, p.n-1)
}

const (
	prec  = 32
	top   = uint64(1)<<prec - 1
	half  = uint64(1) << (prec - 1)
	qtr   = uint64(1) << (prec - 2)
	total = uint64(1) << 28
)

// freqs fills cum with cumulative integer frequencies from logits, in float64, every token at least 1.
func freqs(logits []float32, cum []uint64) {
	mx := float32(math.Inf(-1))
	for _, v := range logits {
		mx = max(mx, v)
	}
	p := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		p[i] = math.Exp(float64(v - mx))
		sum += p[i]
	}
	scale := float64(total-uint64(len(logits))) / sum
	var c uint64
	for i := range p {
		c += uint64(math.Floor(p[i]*scale)) + 1
		cum[i] = c
	}
}

func uniform(cum []uint64) {
	for i := range cum {
		cum[i] = uint64(i + 1)
	}
}

type progress struct {
	t0, last time.Time
	n        int
}

func (p *progress) tick(i int) {
	if i > 0 && i%2000 == 0 {
		now := time.Now()
		fmt.Printf("  %s token %d: %.1f tok/s over the last 2000\n", now.Format("15:04:05"), i, 2000/now.Sub(p.last).Seconds())
		p.last = now
	}
}

func main() {
	mode, path, src, dst := os.Args[1], os.Args[2], os.Args[3], os.Args[4]
	t0 := time.Now()
	m, err := engine.Load(path, engine.LoadOptions{})
	if err != nil {
		panic(err)
	}
	defer m.Close()
	info := m.Info()
	where := info.GPU
	if where == "" {
		where = "CPU (" + info.GPUNote + ")"
	}
	fmt.Printf("%s: model %s %s on %s, load %.1fs\n", mode, info.Name, info.Quant, where, time.Since(t0).Seconds())
	V := m.VocabSize()
	cum := make([]uint64, V)
	t0 = time.Now()
	pr := &progress{t0: t0, last: t0}

	if mode == "encode" {
		b, _ := os.ReadFile(src)
		text := string(b)
		ids := m.Tokenize(text)
		if m.Detokenize(ids) != text {
			panic("tokenizer round-trip mismatch")
		}
		win, _ := strconv.Atoi(os.Getenv("NLC_WINDOW"))
		pr2 := newPredictor(m, win, len(ids))
		var bits []byte
		low, high, pending := uint64(0), top, 0
		emit := func(b byte) {
			bits = append(bits, b)
			for ; pending > 0; pending-- {
				bits = append(bits, 1-b)
			}
		}
		uniform(cum)
		for i, t := range ids {
			tot, hi := cum[V-1], cum[t]
			lo := uint64(0)
			if t > 0 {
				lo = cum[t-1]
			}
			rng := high - low + 1
			high = low + rng*hi/tot - 1
			low = low + rng*lo/tot
			for {
				if high < half {
					emit(0)
				} else if low >= half {
					emit(1)
					low -= half
					high -= half
				} else if low >= qtr && high < half+qtr {
					pending++
					low -= qtr
					high -= qtr
				} else {
					break
				}
				low, high = 2*low, 2*high+1
			}
			if i+1 < len(ids) {
				freqs(pr2.next(t), cum)
			}
			pr.tick(i + 1)
		}
		pending++
		if low < qtr {
			emit(0)
		} else {
			emit(1)
		}
		out := make([]byte, 8, 8+len(bits)/8+1)
		binary.LittleEndian.PutUint32(out, uint32(len(ids)))
		binary.LittleEndian.PutUint32(out[4:], uint32(win))
		for k := 0; k < len(bits); k += 8 {
			var c byte
			for j := 0; j < 8; j++ {
				c <<= 1
				if k+j < len(bits) {
					c |= bits[k+j]
				}
			}
			out = append(out, c)
		}
		os.WriteFile(dst, out, 0o644)
		dt := time.Since(t0).Seconds()
		fmt.Printf("encode window %d: %d -> %d bytes, %d tokens, %.0fs (%.1f tok/s)\n", win, len(b), len(out), len(ids), dt, float64(len(ids))/dt)
		return
	}

	data, _ := os.ReadFile(src)
	n := int(binary.LittleEndian.Uint32(data))
	win := int(binary.LittleEndian.Uint32(data[4:]))
	data = data[8:]
	pos := 0
	bit := func() uint64 {
		pos++
		if pos > 8*len(data) {
			return 0
		}
		return uint64(data[(pos-1)/8]>>(7-(pos-1)%8)) & 1
	}
	var value uint64
	for k := 0; k < prec; k++ {
		value = 2*value + bit()
	}
	pr2 := newPredictor(m, win, n)
	low, high := uint64(0), top
	ids := make([]int, 0, n)
	uniform(cum)
	for i := 0; i < n; i++ {
		tot := cum[V-1]
		rng := high - low + 1
		target := ((value-low+1)*tot - 1) / rng
		t := sort.Search(V, func(j int) bool { return cum[j] > target })
		hi, lo := cum[t], uint64(0)
		if t > 0 {
			lo = cum[t-1]
		}
		high = low + rng*hi/tot - 1
		low = low + rng*lo/tot
		for {
			if high < half {
			} else if low >= half {
				value -= half
				low -= half
				high -= half
			} else if low >= qtr && high < half+qtr {
				value -= qtr
				low -= qtr
				high -= qtr
			} else {
				break
			}
			low, high, value = 2*low, 2*high+1, 2*value+bit()
		}
		ids = append(ids, t)
		if i+1 < n {
			freqs(pr2.next(t), cum)
		}
		pr.tick(i + 1)
	}
	text := m.Detokenize(ids)
	os.WriteFile(dst, []byte(text), 0o644)
	dt := time.Since(t0).Seconds()
	fmt.Printf("decode window %d: %d -> %d bytes, %d tokens, %.0fs (%.1f tok/s)\n", win, len(data)+8, len(text), n, dt, float64(n)/dt)
}
