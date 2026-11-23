// Package signal compresses files with a language model driving an arithmetic coder. nollama runs the model in
// integer arithmetic (engine.IntModel), so every machine computes the same next-token counts and decodes exactly what
// any other machine encoded. This package holds the codec: tokenization policy, coder, archive format, file sets and
// the model manifest (SPEC.md).
package signal

import (
	"bytes"
	"fmt"
	"unicode/utf8"

	"github.com/benallfree/nollama/engine"
)

// Codec compresses and decompresses with one model.
type Codec struct {
	Info  ModelInfo
	Model *engine.IntModel
}

// Open loads a model's converted file, on a GPU when one works (gpu "auto" or "") or the CPU ("off").
// threads <= 0: all CPUs.
func Open(info ModelInfo, threads int, gpu string) (*Codec, error) {
	if !info.Present() {
		return nil, fmt.Errorf("signal: model %d (%s, %s) is not installed in %s: run signal --setup %s", info.ID,
			info.Profile, info.Source.Repo, ModelsDir(), info.Profile)
	}
	m, err := engine.LoadIntModelOptions(info.Path(), engine.IntOptions{Threads: threads, GPU: gpu})
	if err != nil {
		return nil, err
	}
	return &Codec{Info: info, Model: m}, nil
}

// Close releases the model.
func (c *Codec) Close() error { return c.Model.Close() }

// Device names where the model runs.
func (c *Codec) Device() string { return c.Model.Device() }

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
	ids := c.tokenize(data)
	n := len(ids)
	enc := newEncoder()
	ses, err := c.Model.NewSession(min(max(n, 1), Seg))
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
		}
	}
	return append(Header{Kind: kind, Model: c.Info.ID, Tokens: n}.bytes(), enc.finish()...), nil
}

// Decompress decodes an archive made with this codec's model.
func (c *Codec) Decompress(a []byte, progress Progress) (Header, []byte, error) {
	h, body, err := ReadHeader(a)
	if err != nil {
		return h, nil, err
	}
	if h.Model != c.Info.ID {
		return h, nil, fmt.Errorf("signal: the archive needs model %d, this is model %d", h.Model, c.Info.ID)
	}
	dec := newDecoder(body)
	ids := make([]int, 0, h.Tokens)
	ses, err := c.Model.NewSession(min(max(h.Tokens, 1), Seg))
	if err != nil {
		return h, nil, err
	}
	defer ses.Close()
	for s0 := 0; s0 < h.Tokens; s0 += Seg {
		ses.Reset()
		prev := c.Model.BOS()
		for k := s0; k < min(s0+Seg, h.Tokens); k++ {
			cum, err := ses.Feed(prev)
			if err != nil {
				return h, nil, err
			}
			prev = dec.next(cum)
			ids = append(ids, prev)
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
