package surprisal

import (
	"errors"
	"fmt"
)

// Archive container (SPEC.md §6): a kind byte, a byte holding the header version (high nibble) and the compression
// level (low nibble), then for levels 1-9 the token count (unsigned LEB128) and the coded body, or for level 0 the
// bytes themselves.
//
//	0xA1  a stream: the decoded bytes are one unnamed blob
//	0xA2  a file set: the decoded bytes are a container of named files (files.go)
//
// The version says which level table (spec/models-v<version>.json) gives the level's model, so a level keeps its
// meaning in every later surprisal.
//
// Inputs longer than Seg tokens are coded in segments of Seg tokens, each a fresh context; the coder runs on.
const (
	KindStream = 0xA1
	KindFiles  = 0xA2
	Seg        = 32768
	// Version is the header version this surprisal writes.
	Version = 1
)

// Header is an archive's header.
type Header struct {
	Kind, Version, Level int
	Tokens               int // levels 1-9
}

func (h Header) bytes() []byte {
	out := []byte{byte(h.Kind), byte(h.Version<<4 | h.Level)}
	if h.Level == 0 {
		return out
	}
	n := h.Tokens
	for {
		b := byte(n & 0x7f)
		n >>= 7
		if n != 0 {
			out = append(out, b|0x80)
		} else {
			return append(out, b)
		}
	}
}

// ReadHeader parses an archive's header and returns the rest: the coded body, or for level 0 the stored bytes.
func ReadHeader(a []byte) (Header, []byte, error) {
	if len(a) < 2 || (a[0] != KindStream && a[0] != KindFiles) {
		return Header{}, nil, errors.New("surprisal: not a .13k archive")
	}
	h := Header{Kind: int(a[0]), Version: int(a[1] >> 4), Level: int(a[1] & 15)}
	if h.Version > Version {
		return h, nil, fmt.Errorf("surprisal: the archive was made by a newer surprisal (header version %d; this one reads version %d): update surprisal to unpack it", h.Version, Version)
	}
	if h.Version != Version {
		return h, nil, fmt.Errorf("surprisal: the archive has header version %d, from a surprisal before 0.0.1-alpha.2, which this one can't unpack", h.Version)
	}
	if h.Level > 9 {
		return h, nil, fmt.Errorf("surprisal: bad compression level %d", h.Level)
	}
	if h.Level == 0 {
		return h, a[2:], nil
	}
	var v uint64
	i, shift := 2, 0
	for {
		if i >= len(a) || shift > 56 {
			return h, nil, errors.New("surprisal: bad token count")
		}
		b := a[i]
		v |= uint64(b&0x7f) << shift
		shift += 7
		i++
		if b&0x80 == 0 {
			break
		}
	}
	h.Tokens = int(v)
	return h, a[i:], nil
}
