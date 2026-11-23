package signal

import "errors"

// Archive container (SPEC.md §6): a kind byte, the model id, the token count (unsigned LEB128), then the coded body.
//
//	0xA1  a stream: the decoded bytes are one unnamed blob
//	0xA2  a file set: the decoded bytes are a container of named files (files.go)
//
// Inputs longer than Seg tokens are coded in segments of Seg tokens, each a fresh context; the coder runs on.
const (
	KindStream = 0xA1
	KindFiles  = 0xA2
	Seg        = 32768
)

// Header is an archive's header.
type Header struct {
	Kind, Model int
	Tokens      int
}

func (h Header) bytes() []byte {
	out := []byte{byte(h.Kind), byte(h.Model)}
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

// ReadHeader parses an archive's header and returns the coded body.
func ReadHeader(a []byte) (Header, []byte, error) {
	if len(a) < 3 || (a[0] != KindStream && a[0] != KindFiles) {
		return Header{}, nil, errors.New("signal: not a .13k archive")
	}
	var v uint64
	i, shift := 2, 0
	for {
		if i >= len(a) || shift > 56 {
			return Header{}, nil, errors.New("signal: bad token count")
		}
		b := a[i]
		v |= uint64(b&0x7f) << shift
		shift += 7
		i++
		if b&0x80 == 0 {
			break
		}
	}
	return Header{Kind: int(a[0]), Model: int(a[1]), Tokens: int(v)}, a[i:], nil
}
