# Benchmarks

Surprisal started as a question about [js13k](https://js13kgames.com), the game jam whose entries must fit in a
13,312-byte zip: if the contest supplied the codec on both ends, how much more game would fit if that codec carried a
language model? This folder holds that study, the corpus it ran on, and a runner that measures surprisal itself on
the same games.

- [STUDY.md](STUDY.md): the study. 1,163 real entries from 2020–2025, every legal zip method, xz, Brotli, zstd,
  trained dictionaries, a hand-built context-mixing model and several language models, compared game by game.
- [TIMING.md](TIMING.md): how fast arithmetic coding with a language model can go, from the first MLX prototype on a
  Mac to the integer codec on NVIDIA GPUs.
- [run/](run/): the surprisal benchmark runner (below).

## Running surprisal on the corpus

Fetch the corpus once. It downloads every entry's submitted zip from js13kgames.com (about 15 MB, 1,164 files, one
request at a time) into `corpus/zips/`, which git ignores; `corpus/zips.sha256` checks it against the copy the
published numbers came from:

```bash
benchmarks/corpus/fetch.sh
```

```bash
(cd benchmarks/corpus && shasum -a 256 -c --quiet zips.sha256)
```

Then, from the repository root, with a model installed (`surprisal --setup good`):

```bash
go run ./benchmarks/run -profile good -year 2025
```

It compresses each game's files exactly as `surprisal -c` would and writes a markdown table to
`benchmarks/out/surprisal-<profile>-<year>.md` (and stdout): totals for surprisal next to the submitted zips, the best
legal zip per game, Brotli with and without a trained dictionary and the trained context-mixing model (those come from
`results.json`, which `bench.mjs` wrote), the median ratio to the best zip, and a row per game. Flags:

| flag | default | |
|---|---|---|
| `-profile` | `good` | `good` (0.5B), `better` (1.5B) or `best` (7B) |
| `-year` | `2025` | the corpus year; 2025 is the held-out test year of the study |
| `-limit` | all | only the first N games |
| `-games` | all | comma-separated slugs |
| `-verify` | off | also decompress every archive and compare, file by file |
| `-gpu` | `auto` | `off` runs on the CPU (same archives, much slower) |
| `-per-game` | on | include the per-game table |

Zips that use LZMA, PPMd or Deflate64 (a few entrants squeezed their zips that way) are extracted with `7z` when it's
installed and skipped otherwise.

## The rest of the folder

| path | what it is |
|---|---|
| [corpus/](corpus/) | the slug list, the fetch script, the checksums, and a survey of the entries' source repos |
| [bench.mjs](bench.mjs), [report.mjs](report.mjs), [codec.mjs](codec.mjs), [zipread.mjs](zipread.mjs) | the classic-compressor and context-mixing benchmark (Node 20+; writes `results.json`, `results.md` and the tables in STUDY.md) |
| [time_classic.mjs](time_classic.mjs), [timing-classic.json](timing-classic.json) | timings for the classic methods |
| [llm/](llm/) | the language-model arms of the study (MLX on a Mac, plus the CUDA float experiments), with their logs |
| [bjj-case/](bjj-case/) | one large readable game in five builds, readable to Roadroller-packed, used as a case study |
| [lasso-selfextract/](lasso-selfextract/) | the first experiment: a self-extracting page with the decoder counted against the limit |
| [ollama-spike/](ollama-spike/) | an early comparison of scoring through nollama and Ollama |

The commands in STUDY.md's "Reproduce" section run from this folder (`cd benchmarks`).

## About the corpus

The games belong to their authors. They aren't in this repository: `fetch.sh` downloads them from js13kgames.com, the
same public files the contest serves, and the results here are measurements of them, not copies.
