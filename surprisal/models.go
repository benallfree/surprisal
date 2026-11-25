package surprisal

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/benallfree/nollama/catalog"
	"github.com/benallfree/nollama/engine"
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
	URLs []string `json:"urls"` // where the converted file is hosted, tried in order before converting locally
}

type manifest struct {
	Models  []ModelInfo `json:"models"`
	Levels  []int       `json:"levels"`
	Windows []int       `json:"windows"`
	Sink    int         `json:"sink"`
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

// LevelWindow is a level's attention window: each position attends to the last window positions of its segment and
// the first sink; window 0 means full attention (SPEC.md §3).
func LevelWindow(level int) (window, sink int) {
	d := loadManifest()
	if level < 1 || level >= len(d.Windows) || d.Windows[level] == 0 {
		return 0, 0
	}
	return d.Windows[level], d.Sink
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

// ModelsDir is where converted models live: $SURPRISAL_HOME, else ~/.surprisal.
func ModelsDir() string {
	if d := os.Getenv("SURPRISAL_HOME"); d != "" {
		return catalog.ExpandHome(d)
	}
	if h, err := os.UserHomeDir(); err == nil {
		return filepath.Join(h, ".surprisal")
	}
	return ".surprisal"
}

// Path is where the model's converted file lives.
func (m ModelInfo) Path() string { return filepath.Join(ModelsDir(), m.Bundle.File) }

// Present reports whether the converted file is installed.
func (m ModelInfo) Present() bool {
	_, err := os.Stat(m.Path())
	return err == nil
}

// Install installs the converted model. It downloads the converted file from the manifest's URLs when it has any;
// when none works, it downloads the official weights at the pinned revision from Hugging Face and converts them with
// nollama. Either way the file is installed only if its SHA-256 is the manifest's. logf gets messages (a URL that
// failed, the fallback, the conversion's steps); progress, when set, gets each download's file name, bytes done and
// total (0 when unknown).
func (m ModelInfo) Install(logf func(format string, a ...any), progress func(file string, done, total int64)) error {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	if progress == nil {
		progress = func(string, int64, int64) {}
	}
	dir := ModelsDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, u := range m.URLs {
		s := catalog.Source{Name: m.Name, Path: m.Path(), URL: u, SHA256: m.Bundle.SHA256, Size: m.Bundle.Bytes}
		err := catalog.Fetch(s, nil, func(done, total int64) { progress(m.Bundle.File, done, total) })
		if err == nil {
			return nil
		}
		logf("%v", err)
	}
	if len(m.URLs) > 0 {
		logf("no hosted copy of %s worked; building it from %s instead", m.Name, m.Source.Repo)
	}
	return m.convert(dir, logf, progress)
}

// convert builds the converted model from the official weights.
func (m ModelInfo) convert(dir string, logf func(format string, a ...any), progress func(string, int64, int64)) error {
	src := filepath.Join(dir, "surprisal-src", filepath.Base(m.Source.Repo)+"@"+m.Source.Revision[:12])
	defer os.Remove(filepath.Dir(src)) // the surprisal-src folder, if nothing else is in it
	defer os.RemoveAll(src)
	for _, f := range m.Source.Files {
		s := catalog.Source{Name: m.Source.Repo + "/" + f, Path: filepath.Join(src, f),
			URL: "https://huggingface.co/" + m.Source.Repo + "/resolve/" + m.Source.Revision + "/" + f}
		if err := catalog.Fetch(s, nil, func(done, total int64) { progress(f, done, total) }); err != nil {
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
