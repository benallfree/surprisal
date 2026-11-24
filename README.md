# Surprisal

Surprisal is a lossless compressor that uses a language model as its compression engine.

Every binary release of Surprisal runs on the CPU. If a GPU is available, Surprisal uses it (see
[supported platforms](#supported-platforms)). Archives are the same either way: one made on a Mac's GPU unpacks on an
NVIDIA GPU or a plain CPU, and the other way round.

## Usage

```bash
surprisal src/
surprisal src.13k
```

```bash
surprisal [-0 ... -9] [--compare] [-o out.13k] <files or directories>...
```

```bash
surprisal [-o dir] [-f] archive.13k
```

```bash
surprisal --list
```

Surprisal works out what to do from its arguments: one argument ending in `.13k` is unpacked, and anything else is
compressed into one archive (`<name>.13k` for a single input, `archive.13k` for several). `-1` to `-9` pick the level
(see [levels](#levels)); the default is `-1`, and `-0` stores without a model. Unpacking reads the level from the
archive, and won't overwrite existing files without `-f`. Compressing prints the archive's size, its ratio and change
from the original, and how long it took; `--compare` adds the same for zip, gzip, xz, 7z and Brotli on the same input
(with the `brew install` for any that's missing). `--list` shows which models are installed. `--gpu off` forces the
CPU (same archives, slower). Surprisal downloads the model a level needs the first time you use it and keeps it in
`~/.surprisal`; set `SURPRISAL_HOME` to keep models somewhere else.

## Install

Homebrew (macOS and Linux):

```bash
brew install benallfree/tap/surprisal
```

Or download a binary from the [releases](https://github.com/benallfree/surprisal/releases), or build it with Go 1.27+:

```bash
go install github.com/benallfree/surprisal/cmd/surprisal@latest
```

## Levels

| level | model | size |
|---|---|---:|
| `-0` | none (stored) | – |
| `-1` (default) to `-3` | `qwen2.5-coder-0.5b-int-v1` | 0.5 GB |
| `-4` to `-6` | `qwen2.5-coder-1.5b-int-v1` | 1.6 GB |
| `-7` to `-9` | `qwen2.5-coder-7b-int-v1` | 7.9 GB |

## Supported platforms

Every release binary (macOS, Linux and Windows; x86-64 and ARM64) runs on the CPU. Additionally, if a GPU is
available, here is the compatibility matrix:

| OS | architecture | GPU backend |
|---|---|---|
| macOS | Apple Silicon | Metal |
| Linux | x86-64 | CUDA, compute capability 7.5+ (int8 tensor cores on 8.0+) |
| Linux | ARM64 | CUDA |
| Windows | x86-64 | CUDA |

## Benchmarks

Compressed size in bytes, smaller is better. 🟩 marks the smallest in each row.

| file | unpacked | zip | 7z | Brotli | Surprisal |
|---|---:|---:|---:|---:|---:|
| js13k 2025: kittens-crossing | 17,069 | 12,626 | 12,962 | 12,322 | 🟩 **11,815** |
| js13k 2025: miami-mice | 19,876 | 12,450 | 12,593 | 11,808 | 🟩 **10,491** |
| js13k 2025: non-mewtonian-cat | 12,159 | 9,076 | 9,368 | 8,817 | 🟩 **8,539** |
| js13k 2025: tower-of-terror | 42,384 | 9,857 | 8,329 | 7,730 | 🟩 **1,356** |
| js13k 2025: wake-the-cat | 36,088 | 9,289 | 10,231 | 9,640 | 🟩 **2,734** |
| BJJ | 55,854 | 19,395 | 20,516 | 19,723 | 🟩 **4,970** |

zip is the smallest legal zip per game (every 7-Zip method and Zopfli), 7z is `7z -mx=9`, Brotli is `-q11 -w24`, and
Surprisal is `-9`. The js13k games are from the 2025 contest, which the models never trained on. BJJ is a fully
minified 3D game ([spec/vectors/bjj.html](spec/vectors/bjj.html)).

The full table, which grows as more files are measured, is [benchmarks/SIZES.md](benchmarks/SIZES.md). How to run
it yourself, and the original js13k study, are in [benchmarks/](benchmarks/).

## History

The last [js13k](https://js13kgames.com) contest got overrun with AI-made games. Rather than try to ban AI games,
why not embrace AI, use it to its fullest, and give it a challenge worthy of it: how great a game can you truly fit in
13 kilobytes?

Surprisal is my proposal to the js13k project: pick a model, freeze it, and make it the official js13k compression
format, refreshed with a new model every year. On the js13k games in the [benchmarks](#benchmarks), Surprisal's
archives are a third smaller than the best zip in total, so the same 13,312 bytes hold that much more game.

## Models

Surprisal doesn't run the Qwen2.5-Coder models as they're published. It runs its own models, built from the
Qwen2.5-Coder base models and converted to integer arithmetic: `qwen2.5-coder-0.5b-int-v1`, `-1.5b-int-v1` and
`-7b-int-v1`.

**Why integers.** An ordinary (floating-point) model computes with numbers that round after every operation, and the
result depends on the order the operations run in. GPUs, drivers and kernels each sum in their own order, so the same
model on two machines gives slightly different probabilities. A chat assistant never notices. A compressor does: the
decompressor has to reproduce every probability the compressor used, bit for bit, or the archive comes out as
garbage. A floating-point compressor's archives usually only unpack on the hardware that made them.

An integer model does all its arithmetic in whole numbers, which are exact, so the order doesn't matter and every
machine gets the same answer. To get there, each base model is converted once: weights to 8-bit integers with 16-bit
scales, activations to block floating point (integers with a shared exponent), and nonlinear functions such as softmax
and SiLU to lookup tables. The integer models compress within a few bytes of the floating-point originals (7,185 vs
7,183 bytes on the BJJ file with the 0.5B). [SPEC.md](SPEC.md) defines the conversion exactly.

**Where they come from.** The models are hosted on [Hugging Face](https://huggingface.co/benallfree). Surprisal
downloads the model a level needs the first time you use that level, into `~/.surprisal` (or `$SURPRISAL_HOME` if
you set it), and checks it against the SHA-256 in [spec/models-v1.json](spec/models-v1.json). You don't
have to trust the hosted files: the conversion is deterministic, and if no hosted copy can be reached, Surprisal
downloads the Qwen base model at its pinned revision and builds the same file locally.

Model ids are permanent. A later release that adds a model gives it a new id and keeps unpacking archives made with the
old ones.

## How it works

At each step the model gives a probability for every possible next token, and an arithmetic coder spends about
−log₂ p bits on the token that actually comes next, so text the model finds unsurprising costs almost nothing. The
decompressor runs the same model and replays the same predictions.

- **Coder:** a 32-bit arithmetic coder over the model's integer frequency table for each next token.
- **Compression** knows every token in advance, so it runs all positions in large batches (tensor-core GEMMs and
  attention on NVIDIA GPUs). **Decompression** has to go one token at a time; on CUDA the next step is queued before the
  host reads the token, so the GPU never waits for the coder.
- **Engine:** [nollama](https://github.com/benallfree/nollama), a pure-Go inference engine, runs the integer model on
  Metal, CUDA or the CPU. The binaries are pure Go (no cgo): CUDA and Metal load at run time, so one binary runs with or
  without a GPU driver installed.
- **Archive:** each archive records the format version and the compression level, which together name the exact model
  that unpacks it.

## License

MIT ([LICENSE.md](LICENSE.md)). The Qwen2.5-Coder base models are Apache 2.0, from Alibaba's Qwen team.
