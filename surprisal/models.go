package surprisal

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/benallfree/nollama/engine"
	"github.com/benallfree/nollama/models"
)

//go:embed models-v1.json
var manifestJSON []byte

// ModelInfo is a manifest entry: the model an archive names by id, where it comes from, and its converted file.
type ModelInfo struct {
	ID     int    `json:"id"`
	Name   string `json:"name"`
	Source struct {
		Repo     string   `json:"repo"`
		Revision string   `json:"revision"`
		Files    []string `json:"files"`
	} `json:"source"`
	Bundle struct {
		File   string `json:"file"`
		SHA256 string `json:"sha256"`
		Bytes  int64  `json:"bytes"`
	} `json:"bundle"`
}

type manifest struct {
	Models []ModelInfo `json:"models"`
	Levels []int       `json:"levels"`
}

func loadManifest() manifest {
	var d manifest
	if err := json.Unmarshal(manifestJSON, &d); err != nil {
		panic(err)
	}
	return d
}

// Models lists the manifest (ids are permanent).
func Models() []ModelInfo { return loadManifest().Models }

// ModelByID finds a model by its manifest id.
func ModelByID(id int) (ModelInfo, bool) {
	for _, m := range Models() {
		if m.ID == id {
			return m, true
		}
	}
	return ModelInfo{}, false
}

// ModelForLevel finds the model a compression level (1-9) uses under header version 1. Level 0 stores and has none.
func ModelForLevel(level int) (ModelInfo, bool) {
	l := loadManifest().Levels
	if level < 1 || level >= len(l) {
		return ModelInfo{}, false
	}
	return ModelByID(l[level])
}

// LevelsFor lists the levels that use a model.
func LevelsFor(id int) []int {
	var out []int
	for lv, m := range loadManifest().Levels {
		if lv > 0 && m == id {
			out = append(out, lv)
		}
	}
	return out
}

// ModelsDir is where converted models live: $SURPRISAL_MODELS, else nollama's models directory (shared with nollama).
func ModelsDir() string {
	if d := os.Getenv("SURPRISAL_MODELS"); d != "" {
		return models.ExpandHome(d)
	}
	if d := os.Getenv("NOLLAMA_MODELS"); d != "" {
		return models.ExpandHome(d)
	}
	return models.DefaultDir()
}

// Path is where the model's converted file lives.
func (m ModelInfo) Path() string { return filepath.Join(ModelsDir(), m.Bundle.File) }

// Present reports whether the converted file is installed.
func (m ModelInfo) Present() bool {
	_, err := os.Stat(m.Path())
	return err == nil
}

// Install makes the converted model: it downloads the official weights at the pinned revision from Hugging Face,
// converts them with nollama, and installs the file only if its SHA-256 is the manifest's.
func (m ModelInfo) Install(logf func(format string, a ...any)) error {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	dir := ModelsDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	src := filepath.Join(dir, "surprisal-src", filepath.Base(m.Source.Repo)+"@"+m.Source.Revision[:12])
	defer os.Remove(filepath.Dir(src)) // the surprisal-src folder, if nothing else is in it
	defer os.RemoveAll(src)
	for _, f := range m.Source.Files {
		s := models.Source{Name: m.Source.Repo + "/" + f, Path: filepath.Join(src, f),
			URL: "https://huggingface.co/" + m.Source.Repo + "/resolve/" + m.Source.Revision + "/" + f}
		if err := models.Fetch(s, nil, func(done, total int64) { logf("download %s: %d/%d MB", f, done>>20, total>>20) }); err != nil {
			return err
		}
	}
	tmp := filepath.Join(dir, fmt.Sprintf("surprisal-convert-%d.gguf", os.Getpid()))
	defer os.Remove(tmp)
	if err := engine.ConvertIntModel(src, tmp, func(format string, a ...any) { logf("convert: "+format, a...) }); err != nil {
		return err
	}
	h, err := engine.FileSHA256(tmp)
	if err != nil {
		return err
	}
	if h != m.Bundle.SHA256 {
		return fmt.Errorf("surprisal: the converted %s has sha256 %s, not the manifest's %s", m.Bundle.File, h, m.Bundle.SHA256)
	}
	return os.Rename(tmp, m.Path())
}
