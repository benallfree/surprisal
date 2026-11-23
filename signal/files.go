package signal

import (
	"bytes"
	"errors"
	"fmt"
	"path"
	"strconv"
	"strings"
)

// File is one file of a file set: a relative, slash-separated path and its bytes.
type File struct {
	Path string
	Data []byte
}

// A file set's decoded bytes: for each file, its path, "\n", its length in bytes (decimal), "\n", then its bytes.
// Paths and contents are coded by the model like everything else, so later files are compressed in the context of
// earlier ones.

func encodeFiles(fs []File) ([]byte, error) {
	var b bytes.Buffer
	for _, f := range fs {
		if err := checkPath(f.Path); err != nil {
			return nil, err
		}
		b.WriteString(f.Path)
		b.WriteByte('\n')
		b.WriteString(strconv.Itoa(len(f.Data)))
		b.WriteByte('\n')
		b.Write(f.Data)
	}
	return b.Bytes(), nil
}

func decodeFiles(b []byte) ([]File, error) {
	var fs []File
	for len(b) > 0 {
		i := bytes.IndexByte(b, '\n')
		if i < 0 {
			return nil, errors.New("signal: bad file set (path)")
		}
		p := string(b[:i])
		b = b[i+1:]
		j := bytes.IndexByte(b, '\n')
		if j < 0 {
			return nil, errors.New("signal: bad file set (length)")
		}
		n, err := strconv.Atoi(string(b[:j]))
		if err != nil || n < 0 || n > len(b)-j-1 {
			return nil, errors.New("signal: bad file set (length)")
		}
		if err := checkPath(p); err != nil {
			return nil, err
		}
		fs = append(fs, File{Path: p, Data: b[j+1 : j+1+n]})
		b = b[j+1+n:]
	}
	return fs, nil
}

// checkPath accepts relative, clean, slash-separated paths that stay inside the output directory.
func checkPath(p string) error {
	if p == "" || strings.ContainsAny(p, "\n\\\x00") || path.IsAbs(p) || path.Clean(p) != p ||
		p == ".." || strings.HasPrefix(p, "../") {
		return fmt.Errorf("signal: unsafe path %q in file set", p)
	}
	return nil
}
