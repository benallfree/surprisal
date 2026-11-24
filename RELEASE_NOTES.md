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
