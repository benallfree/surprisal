# Surprisal 0.0.1-alpha.2

Compression levels are now a real speed-to-size spectrum, models download on their own, and GPU compression is faster at
every level.

## What's new

- **Models download on first use.** The converted models are hosted on Hugging Face
  ([benallfree](https://huggingface.co/benallfree)); the first use of a level downloads its model into `~/.surprisal`
  (or `$SURPRISAL_HOME`), with a progress bar, and checks it against the SHA-256 in `spec/models-v1.json`. If no hosted
  copy can be reached, Surprisal builds it locally from the Qwen base model as before. `--setup`, `--profile` and
  `SURPRISAL_MODELS` are gone.
- **A simpler command line.** No `-c`/`-d`: one argument ending in `.13k` unpacks, anything else compresses (one input
  becomes `<name>.13k`). `-0` to `-9` pick the level instead of a profile (`-0` stores without a model), and unpacking
  reads the level from the archive. Compressing prints the archive's size, ratio, change from the original and time;
  `--compare` adds each zip method (Deflate, Zopfli, Deflate64, BZip2, LZMA, PPMd), 7z (LZMA2, PPMd), gzip, xz and
  Brotli on the same input.
- **Levels -1 to -9 trade speed for size.** Each level is a model and an attention window: how far back the model
  looks within a segment (plus the segment's first 16 tokens). -3, -6 and -9 use full attention, -1/-2 and -4/-5 a
  2,048- or 4,096-token window, -7/-8 a 4,096- or 8,192-token one. -1 stays the default and is now the fast one; -9
  makes the smallest archives. On BJJ, -1 compresses at 1,176 tok/s (3.7x alpha.1's level 1) for an archive 10%
  bigger than -3's. A window also forgets repeats further back than its length: BJJ three times over is 23,827 bytes
  at -1, against 16,945 with alpha.1's full attention. See the README's level table.
- **Faster GPU compression at every level.** The integer arithmetic changed (SPEC.md): activations entering a matrix
  product have 12-bit mantissas, attention queries 12 bits, cached keys 16 bits and values 12 bits, each vector with
  its own exponent. The batched matrix products then run exactly in float, and each attention score term takes one
  32-bit multiply. On an M1 Max, full-attention compression of BJJ (24,126 tokens) goes from 320 to 542 tok/s with
  the 0.5B (-3), 165 to 198 with the 1.5B (-6) and 48 to 78 with the 7B (-9). The archives stay the same size or
  shrink: 7,186 to 7,185 bytes, 5,856 to 5,855, 4,970 to 4,944.
- **Fixes:** the progress line's speed and time left now follow the recent rate (the average since the start was
  wrong within each segment); tokenizing a long run with no spaces took quadratic time (a 16 KB run of letters 2.9
  s, now 7 ms); on Macs, 7B runs commit shorter GPU command buffers so macOS's watchdog doesn't kill them, a launch with too many threads per
  group is an error instead of silently wrong output, and a zero divisor can't hang the GPU.

## Format

Archives from 0.0.1-alpha.1 don't unpack with this release (the header changed, and the arithmetic with it). The
header now records its version and the level, and an unknown version is refused. The converted models are unchanged:
the same files and SHA-256s.

## Downloads

| Platform | Archive |
|---|---|
| macOS, Apple Silicon (Metal) | `surprisal_0.0.1-alpha.2_darwin_arm64.tar.gz` |
| macOS, Intel (CPU) | `surprisal_0.0.1-alpha.2_darwin_amd64.tar.gz` |
| Linux x86-64 (CUDA or CPU) | `surprisal_0.0.1-alpha.2_linux_amd64.tar.gz` |
| Linux ARM64 (CUDA or CPU; untested) | `surprisal_0.0.1-alpha.2_linux_arm64.tar.gz` |
| Windows x86-64 (CUDA or CPU; untested) | `surprisal_0.0.1-alpha.2_windows_amd64.zip` |
| Windows ARM64 (CPU; untested, new) | `surprisal_0.0.1-alpha.2_windows_arm64.zip` |

`checksums.txt` has the SHA-256 of each. Homebrew: `brew install benallfree/tap/surprisal`. Go:
`go install github.com/benallfree/surprisal/cmd/surprisal@v0.0.1-alpha.2`.

## Known limits

- Decompression still runs one token at a time and its per-token cost didn't change (about 107 tok/s at the start
  of a file with the 0.5B, 61 with the 1.5B, about 20 with the 7B on an M1 Max). Windowed levels keep that speed
  through long files: -1 decodes all of BJJ at 118 tok/s on average.
- Checked on an NVIDIA A16 (compute 8.6, tensor cores): every vector decodes and re-encodes byte for byte, including
  BJJ at levels 1, 3, 4, 6 and 9, the same archives the Mac's GPU and CPU write.
- Windows (x86-64 and ARM64) and Linux ARM64 builds compile but haven't been run on real hardware yet.

# Surprisal 0.0.1-alpha.1

The first public build of Surprisal, a lossless compressor that uses a language model to predict the next token and
arithmetic coding to store only how wrong it was. It is an early alpha: the archive format (spec v1) is frozen, but the
command line and the library API may still change.

## What's in it

- `surprisal -c` / `surprisal -d`: compress files and directories into one `.13k` archive and unpack it again.
- Three model profiles, all Qwen2.5-Coder base models converted to the integer spec: `good` (0.5B), `better` (1.5B)
  and `best` (7B). `surprisal --setup <profile>` downloads the official weights and converts them locally;
  `surprisal --list` shows what is installed.
- **Bit-identical on any hardware.** The forward pass is integer-only (spec v1, [SPEC.md](SPEC.md)), so an archive made
  on a Mac's GPU decodes on an NVIDIA GPU or a plain CPU, and vice versa.
- **GPU backends:** Metal on Apple Silicon; CUDA on NVIDIA GPUs of compute capability 7.5 or newer, with tensor cores on
  8.0 or newer (int8 tensor-core matmuls and attention). The CPU path works everywhere and gives the same archives.
- **Speed on an H100**, on a 55,854-byte file (24,126 tokens): compress in 1.9 s / 3.4 s / 8.3 s and decompress in
  29.4 s / 54.5 s / 2 min 0 s for good / better / best. That's as fast as or faster than the Python reference's own
  CUDA path in every case. See the README's speed table.
- **Size:** the same 55,854-byte file becomes 7,186 / 5,856 / 4,970 bytes (good / better / best); `brotli -q11` makes
  it 19,723, `xz -9e` 20,448 and `gzip -9` 22,390. On the 197 js13k 2025 games, the research study measured the 0.5B model 37% smaller
  than the best legal zip per game in total. `go run ./benchmarks/run` reruns that comparison with surprisal itself
  (see [benchmarks/](benchmarks/)).

## Downloads

| Platform | Archive |
|---|---|
| macOS, Apple Silicon (Metal) | `surprisal_0.0.1-alpha.1_darwin_arm64.tar.gz` |
| macOS, Intel (CPU) | `surprisal_0.0.1-alpha.1_darwin_amd64.tar.gz` |
| Linux x86-64 (CUDA or CPU) | `surprisal_0.0.1-alpha.1_linux_amd64.tar.gz` |
| Linux ARM64 (untested) | `surprisal_0.0.1-alpha.1_linux_arm64.tar.gz` |
| Windows x86-64 (CUDA or CPU; untested) | `surprisal_0.0.1-alpha.1_windows_amd64.zip` |

`checksums.txt` has the SHA-256 of each. Homebrew: `brew install benallfree/tap/surprisal`.

## Known limits

- Models are 0.5 GB / 1.5 GB / 7.3 GB and are converted on first `--setup`; there are no hosted model downloads yet.
- Decompression runs one token at a time: minutes on a Mac for the larger models (see the speed table).
- Context is one segment of 32,768 tokens; longer inputs are coded in independent segments.
- Windows and Linux ARM64 builds compile but haven't been run on real hardware yet.
