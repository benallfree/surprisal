# Surprisal

Surprisal is a lossless compressor that uses a language model as its predictor. At each step the model gives a
probability for every possible next token, and an arithmetic coder spends about −log₂ p bits on the token that actually
comes next, so text the model finds unsurprising costs almost nothing. The decompressor runs the same model and
replays the same predictions.

The catch with neural compressors is that floating-point inference differs between GPUs, drivers and kernels, so an
archive usually only decodes on the hardware that made it. Surprisal's forward pass is **integer-only** ([SPEC.md](SPEC.md),
spec v1): an archive made on a Mac's GPU decodes bit-for-bit on an NVIDIA GPU or a plain CPU, and the other way round.

```bash
surprisal --setup good
surprisal -c -o game.13k --profile good src/
surprisal -d game.13k
```

> **Alpha.** The archive format (spec v1) is frozen: archives made now will keep decoding. The command line and the Go
> API may still change.

## Install

Homebrew (macOS and Linux):

```bash
brew install benallfree/tap/surprisal
```

Or download a binary from the [releases](https://github.com/benallfree/surprisal/releases), or build it with Go 1.27+:

```bash
go install github.com/benallfree/surprisal/cmd/surprisal@latest
```

Then install a model. `--setup` downloads the official Qwen2.5-Coder weights from Hugging Face and converts them to the
integer format (deterministically: every machine produces the same file, checked against the SHA-256 in
[spec/models-v1.json](spec/models-v1.json)):

```bash
surprisal --setup good
```

## Usage

```bash
surprisal -c -o archive.13k [--profile best] <files or directories>...
```

```bash
surprisal -d [-o dir] [-f] archive.13k
```

```bash
surprisal --list
```

The archive records which model made it, so `-d` needs no profile. `--gpu off` forces the CPU (same archives, slower).
Models live in `$SURPRISAL_MODELS`, or in nollama's models directory by default.

## Models

| profile | model | bundle | BJJ file (55,854 bytes) |
|---|---|---:|---:|
| `good` | Qwen2.5-Coder-0.5B | 0.5 GB | 7,186 bytes |
| `better` | Qwen2.5-Coder-1.5B | 1.5 GB | 5,856 bytes |
| `best` (default) | Qwen2.5-Coder-7B | 7.3 GB | 4,970 bytes |

For comparison, on the same file: `brotli -q11` 19,723 bytes, `xz -9e` 20,448, `zstd -19` 21,278, `gzip -9` 22,390.
The BJJ file is a fully minified 3D game (HTML and JavaScript, [spec/vectors/bjj.html](spec/vectors/bjj.html)).

## Supported platforms

| OS | architecture | GPU backend | status |
|---|---|---|---|
| macOS | Apple Silicon | Metal | tested (M1 Max) |
| Linux | x86-64 | CUDA, compute capability 7.5+ (int8 tensor cores on 8.0+) | tested (H100, H200) |
| Linux | ARM64 | CUDA | builds; untested |
| Windows | x86-64 | CUDA | builds; untested |
| macOS | Intel | none (CPU) | builds; untested |
| any of the above | | CPU (`--gpu off`, also the fallback when no GPU works) | tested (M1 Max: about 50 tokens/s decoding with the 0.5B model) |

Every backend produces the same archives. The binaries are pure Go (no cgo): CUDA and Metal are loaded at run time, so
one binary runs with or without a GPU driver installed.

## Speed

Wall time on the BJJ file (55,854 bytes, 24,126 tokens) with each model of spec v1. Every archive and every decode is
byte-identical across both implementations and both machines. Python is the reference implementation, [`intref.py`](reference/python/intref.py) (numpy) on the Mac and
[`intgpu.py`](reference/python/intgpu.py) (Triton) on the H100; surprisal is this repository's CLI with [nollama](https://github.com/benallfree/nollama) (Metal on the Mac, CUDA on the H100, both from the same kernel source). Compression runs every
position at once (batched); decompression has to go one token at a time.

<!-- speed-table: keep updated when either implementation gets faster -->
<table>
<tr><th>model</th><th>M1 Max (Mac)</th><th>H100</th></tr>
<tr><td><b>0.5B</b></td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">~58 min*</td><td align="right">~58 min*</td></tr>
<tr><td>surprisal</td><td align="right">53 s</td><td align="right">3 min 52 s</td></tr>
</table>

</td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">2.0 s</td><td align="right">30.0 s</td></tr>
<tr><td>surprisal</td><td align="right">1.9 s</td><td align="right">29.4 s</td></tr>
</table>

</td></tr>
<tr><td><b>1.5B</b></td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">~2.9 h*</td><td align="right">~2.9 h*</td></tr>
<tr><td>surprisal</td><td align="right">2 min 10 s</td><td align="right">7 min 31 s</td></tr>
</table>

</td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">3.8 s</td><td align="right">54.4 s</td></tr>
<tr><td>surprisal</td><td align="right">3.4 s</td><td align="right">54.5 s</td></tr>
</table>

</td></tr>
<tr><td><b>7B</b></td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">~3.5 days*</td><td align="right">~3.5 days*</td></tr>
<tr><td>surprisal</td><td align="right">7 min 30 s</td><td align="right">22 min 37 s</td></tr>
</table>

</td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">9.3 s</td><td align="right">2 min 3 s</td></tr>
<tr><td>surprisal</td><td align="right">8.3 s</td><td align="right">2 min 0 s</td></tr>
</table>

</td></tr>
</table>

\* Estimated from the numpy reference's speed over the first 300, 120 and 30 tokens (6.9, 2.3 and 0.08 tok/s); later
tokens are slower, and the 7B doesn't fit in memory and swaps. surprisal's H100 runs: [`scripts/gostage.sh`](scripts/gostage.sh) cross-compiles the
tests and the CLI, [`scripts/gorun.py`](scripts/gorun.py) runs them on Modal; H100 decompression times vary by about 10% between runs.


## Compression

How much smaller than the alternatives, in total over the 197 games of js13k 2025 (games the models never trained on):

| method | vs best legal zip per game |
|---|---:|
| best legal zip per game (every 7-Zip method, Zopfli) | 0.0% |
| Brotli -q11 | −5.3% |
| Brotli with a dictionary trained on 2020–2024 entries | −13.6% |
| context mixing, trained on 2020–2024 entries | −18.6% |
| Qwen2.5-Coder-0.5B, off the shelf (text), context mixing (binary assets) | −37.0% |

The language-model row was measured with the float model in the original study; the integer model matches its sizes to
within a few bytes (7,185 vs 7,183 on the BJJ file). [benchmarks/](benchmarks/) has the full study, the corpus and a
runner that measures surprisal itself on the same games:

```bash
go run ./benchmarks/run -profile good -year 2025
```

## How it works

- **Model:** Qwen2.5-Coder base models, converted once to int8 weights with int16 scales, block-floating-point
  activations and lookup tables for the nonlinearities ([SPEC.md](SPEC.md)). Every operation is exact integer arithmetic,
  so the order a GPU sums in can't change the result.
- **Coder:** a 32-bit arithmetic coder over the model's integer frequency table for each next token.
- **Compression** knows every token in advance, so it runs all positions in large batches (tensor-core GEMMs and
  attention on NVIDIA GPUs). **Decompression** has to go one token at a time; on CUDA the next step is queued before the
  host reads the token, so the GPU never waits for the coder.
- **Engine:** [nollama](https://github.com/benallfree/nollama), a pure-Go inference engine, runs the integer model on
  Metal, CUDA or the CPU.

## Repository

| path | |
|---|---|
| [cmd/surprisal](cmd/surprisal) | the command-line tool |
| [surprisal/](surprisal/) | the Go package: model manifest, arithmetic coder, archive and file-list formats |
| [SPEC.md](SPEC.md), [spec/](spec/) | spec v1: the integer model, conversion, archive format, frozen tables and test vectors |
| [reference/python/](reference/python/) | the reference implementations of spec v1 (numpy, PyTorch and Triton/CUDA) |
| [benchmarks/](benchmarks/) | the js13k study, its corpus, timings and the surprisal benchmark runner |
| [scripts/](scripts/) | cross-compiling the tests and running them on cloud GPUs (Modal) |

Tests: `go test ./...` runs the archive, coder and spec-vector tests; the vector tests need the models installed
(`SURPRISAL_MODELS`) and skip the ones that aren't.

## License

MIT ([LICENSE.md](LICENSE.md)). The Qwen2.5-Coder models are Apache 2.0, from Alibaba's Qwen team.
