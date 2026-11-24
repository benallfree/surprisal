// Scores files with nollama: the archive size an arithmetic coder driven by the model would produce,
// sum of -log2 p(token) over the file, plus 6 bytes (length header and coder flush).
//
//	go run . <model.gguf> <file>...
//
// nollama's public API has no logprobs, so this reaches the engine's teacher-forcing hook (the one its
// benchmark uses to compare against Ollama) through go:linkname. The whole file is one context.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"
	_ "unsafe"

	"github.com/benallfree/nollama/engine"
)

//go:linkname teacherForce github.com/benallfree/nollama/engine/internal/benchhooks.TeacherForce
var teacherForce func(m any, prompt, cont []int) (argmax []int, logprob []float64)

func main() {
	m, err := engine.Load(os.Args[1], engine.LoadOptions{})
	if err != nil {
		panic(err)
	}
	defer m.Close()
	info := m.Info()
	fmt.Printf("model %s %s, context %d, %d threads\n", info.Name, info.Quant, info.ContextLength, info.Threads)
	for _, p := range os.Args[2:] {
		b, err := os.ReadFile(p)
		if err != nil {
			panic(err)
		}
		text := string(b)
		ids := m.Tokenize(text)
		if m.Detokenize(ids) != text {
			fmt.Printf("%s: tokenizer round-trip mismatch\n", filepath.Base(p))
			continue
		}
		if out := os.Getenv("DUMP_IDS"); out != "" { // token ids for the Ollama scorer, which must see the same tokens
			j, _ := json.Marshal(ids)
			os.WriteFile(filepath.Join(out, filepath.Base(p)+".ids.json"), j, 0o644)
			offs := []int{0} // byte offset of each token in the file, plus the end
			for _, id := range ids {
				offs = append(offs, offs[len(offs)-1]+len(m.Detokenize([]int{id})))
			}
			if offs[len(offs)-1] != len(b) {
				panic("token offsets don't add up to the file length")
			}
			j, _ = json.Marshal(offs)
			os.WriteFile(filepath.Join(out, filepath.Base(p)+".offs.json"), j, 0o644)
			fmt.Printf("%s: %d tokens written\n", filepath.Base(p), len(ids))
			continue
		}
		t0 := time.Now()
		// the first token is coded with a uniform distribution over the vocabulary; the rest are teacher-forced
		_, lp := teacherForce(m, ids[:1], ids[1:])
		bits := math.Log2(151936)
		for _, v := range lp {
			bits -= v / math.Ln2
		}
		dt := time.Since(t0)
		fmt.Printf("%s: %d bytes, %d tokens, %.3f bits/byte, %.0fs (%.0f tok/s)\n", filepath.Base(p),
			int(math.Ceil(bits/8))+6, len(ids), bits/float64(len(b)), dt.Seconds(), float64(len(ids))/dt.Seconds())
	}
}
