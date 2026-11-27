// Package surprisal compresses files with a language model driving an arithmetic coder. nollama runs the model in
// integer arithmetic (engine.IntModel), so every machine computes the same next-token counts and decodes exactly what
// any other machine encoded. This package holds the codec: tokenization policy, coder, archive format, file sets and
// the model manifest (SPEC.md).
package surprisal

import (
	"bytes"
	"fmt"
	"unicode/utf8"

	"github.com/benallfree/nollama/engine"
)

// Codec compresses and decompresses at one compression level: 0 stores the bytes, 1-9 code them with the level's
// model.
type Codec struct {
	Level        int
	Info         ModelInfo        // levels 1-9
	Model        *engine.IntModel // levels 1-9
	Window, Sink int              // the level's attention window (0: full attention)
	// Check, if set, is called during Compress after each batch of tokens with the tokens done, the total and the
	// smallest size the archive can still end up (header included; it never goes down), and once more at the end
	// with the archive's actual size. A non-nil error stops Compress, which returns it.
	Check func(done, total, minSize int) error
}

// OverLimit is the error a size limit's Check returns (StopAt): the archive is certain to be bigger than Limit.
type OverLimit struct {
	Limit, MinSize int // MinSize: the smallest the archive could have ended up (its size, when Done == Total)
	Done, Total    int // tokens coded when it stopped
}

func (e *OverLimit) Error() string {
	if e.Done == e.Total {
		return fmt.Sprintf("surprisal: the archive is %d bytes, over the %d-byte limit", e.MinSize, e.Limit)
	}
	return fmt.Sprintf("surprisal: the archive will be at least %d bytes, over the %d-byte limit (stopped at %d/%d tokens)",
		e.MinSize, e.Limit, e.Done, e.Total)
}

// StopAt is a Check that stops once the archive is certain to be over limit bytes. It never stops an input whose
// archive would fit.
func StopAt(limit int) func(done, total, minSize int) error {
	return func(done, total, minSize int) error {
		if minSize > limit {
			return &OverLimit{Limit: limit, MinSize: minSize, Done: done, Total: total}
		}
		return nil
	}
}

// Open makes a codec for a compression level (0-9). For levels 1-9 it loads the level's model, which must be
// installed (ModelInfo.Install), on a GPU when one works (gpu "auto" or "") or the CPU ("off"). threads <= 0: all
// CPUs.
func Open(level, threads int, gpu string) (*Codec, error) {
	if level == 0 {
		return &Codec{}, nil
	}
	info, ok := ModelForLevel(level)
	if !ok {
		return nil, fmt.Errorf("surprisal: compression level %d isn't 0-9", level)
	}
	if !info.Present() {
		return nil, fmt.Errorf("surprisal: level %d needs %s, which is not installed in %s", level, info.Name,
			ModelsDir())
	}
	m, err := engine.LoadIntModelOptions(info.Path(), engine.IntOptions{Threads: threads, GPU: gpu})
	if err != nil {
		return nil, err
	}
	w, sk := LevelWindow(level)
	return &Codec{Level: level, Info: info, Model: m, Window: w, Sink: sk}, nil
}

// Close releases the model.
func (c *Codec) Close() error {
	if c.Model == nil {
		return nil
	}
	return c.Model.Close()
}

// Device names where the model runs.
func (c *Codec) Device() string {
	if c.Model == nil {
		return "no model"
	}
	return c.Model.Device()
}

// Progress, if set, gets the tokens done and the total.
type Progress func(done, total int)

func (c *Codec) detok(ids []int) []byte {
	var b []byte
	for _, id := range ids {
		b = append(b, c.Model.TokenBytes(id)...)
	}
	return b
}

// tokenize: the model's tokenizer when it gives back the exact bytes; otherwise UTF-8 runs tokenized and other
// bytes as one-byte tokens; failing that, one token per byte (always exact). The decoder only needs the tokens'
// bytes, so any tokenization that round-trips is valid.
func (c *Codec) tokenize(b []byte) []int {
	if utf8.Valid(b) {
		if ids := c.Model.Tokenize(string(b)); bytes.Equal(c.detok(ids), b) {
			return ids
		}
	}
	var ids []int
	for i := 0; i < len(b); {
		j := i
		for j < len(b) {
			r, n := utf8.DecodeRune(b[j:])
			if r == utf8.RuneError && n <= 1 {
				break
			}
			j += n
		}
		if j > i {
			ids = append(ids, c.Model.Tokenize(string(b[i:j]))...)
		}
		if j < len(b) {
			ids = append(ids, c.Model.ByteToken(b[j]))
			j++
		}
		i = j
	}
	if bytes.Equal(c.detok(ids), b) {
		return ids
	}
	ids = ids[:0]
	for _, x := range b {
		ids = append(ids, c.Model.ByteToken(x))
	}
	return ids
}

// Compress codes data as an archive of the given kind (KindStream or KindFiles).
func (c *Codec) Compress(kind int, data []byte, progress Progress) ([]byte, error) {
	if c.Level == 0 {
		a := append(Header{Kind: kind, Version: Version}.bytes(), data...)
		if c.Check != nil {
			if err := c.Check(0, 0, len(a)); err != nil {
				return nil, err
			}
		}
		return a, nil
	}
	ids := c.tokenize(data)
	n := len(ids)
	head := Header{Kind: kind, Version: Version, Level: c.Level, Tokens: n}.bytes()
	enc := newEncoder()
	ses, err := c.Model.NewWindowSession(min(max(n, 1), Seg), c.Window, c.Sink)
	if err != nil {
		return nil, err
	}
	defer ses.Close()
	// Every token is known, so the model runs many positions at once (FeedBatch); the counts are the ones
	// decompression's step-by-step tables give. Chunks only pace the progress reports.
	const chunk = 2048
	for s0 := 0; s0 < n; s0 += Seg {
		ses.Reset()
		s1 := min(s0+Seg, n)
		for k0 := s0; k0 < s1; k0 += chunk {
			k1 := min(k0+chunk, s1)
			toks := make([]int, 0, k1-k0)
			if k0 == s0 {
				toks = append(toks, c.Model.BOS())
				toks = append(toks, ids[k0:k1-1]...)
			} else {
				toks = append(toks, ids[k0-1:k1-1]...)
			}
			cs, err := ses.FeedBatch(toks, ids[k0:k1])
			if err != nil {
				return nil, err
			}
			for _, ct := range cs {
				enc.code(uint64(ct.Total), uint64(ct.High), uint64(ct.Low))
			}
			if progress != nil {
				progress(k1, n)
			}
			if c.Check != nil {
				if err := c.Check(k1, n, len(head)+enc.minBytes()); err != nil {
					return nil, err
				}
			}
		}
	}
	a := append(head, enc.finish()...)
	if c.Check != nil {
		if err := c.Check(n, n, len(a)); err != nil {
			return nil, err
		}
	}
	return a, nil
}

// Decompress decodes an archive made at a level that uses this codec's model (any codec decodes level 0).
func (c *Codec) Decompress(a []byte, progress Progress) (Header, []byte, error) {
	h, body, err := ReadHeader(a)
	if err != nil {
		return h, nil, err
	}
	if h.Level == 0 {
		return h, body, nil
	}
	if m, _ := ModelForLevel(h.Level); c.Model == nil || m.ID != c.Info.ID {
		return h, nil, fmt.Errorf("surprisal: the archive is level %d, which needs %s; this codec has %s", h.Level,
			m.Name, c.Info.Name)
	}
	dec := newDecoder(body)
	ids := make([]int, 0, h.Tokens)
	w, sk := LevelWindow(h.Level) // the archive's level, which may differ from the codec's (same model)
	ses, err := c.Model.NewWindowSession(min(max(h.Tokens, 1), Seg), w, sk)
	if err != nil {
		return h, nil, err
	}
	defer ses.Close()
	for s0 := 0; s0 < h.Tokens; s0 += Seg {
		ses.Reset()
		if err := ses.Start(c.Model.BOS()); err != nil {
			return h, nil, err
		}
		s1 := min(s0+Seg, h.Tokens)
		for k := s0; k < s1; k++ {
			// the search runs where the table is and feeds the token found (on CUDA the next step then runs while
			// the decoder updates)
			num, den := dec.state()
			t, ct, err := ses.FindFeed(num, den, k+1 < s1)
			if err != nil {
				return h, nil, err
			}
			dec.update(uint64(ct.Total), uint64(ct.High), uint64(ct.Low))
			ids = append(ids, t)
			if progress != nil {
				progress(k+1, h.Tokens)
			}
		}
	}
	return h, c.detok(ids), nil
}

// CompressFiles codes a file set.
func (c *Codec) CompressFiles(fs []File, progress Progress) ([]byte, error) {
	b, err := encodeFiles(fs)
	if err != nil {
		return nil, err
	}
	return c.Compress(KindFiles, b, progress)
}

// DecodeFiles splits a file set archive's decoded bytes into its files.
func DecodeFiles(b []byte) ([]File, error) { return decodeFiles(b) }
