# Compressed sizes

Compressed size in bytes, smaller is better. 🟩 marks the smallest in each column or row.

## One game, every packing step

The fair comparison: the same game, BJJ, packed each way a js13k entrant might pack it, then compressed by each
method. Each column adds a step to the one before it. The builds are in [bjj-case/](bjj-case/).

| | unpacked | HTML minified | + terser | + Roadroller |
|---|---:|---:|---:|---:|
| no compression | 67,056 | 66,984 | 55,854 | 27,673 |
| zip | 21,780 (PPMd) | 21,745 (PPMd) | 19,395 (PPMd) | 18,867 (Zopfli) |
| 7z | 23,642 | 23,613 | 20,516 | 19,305 |
| Brotli | 22,758 | 22,732 | 19,723 | 18,427 |
| Surprisal `-9` | 🟩 **5,025** | 🟩 **5,027** | 🟩 **4,929** | 🟩 **17,300** |

| column | build |
|---|---|
| unpacked | [1-readable.html](bjj-case/1-readable.html): the source as written |
| HTML minified | [1b-html-minified.html](bjj-case/1b-html-minified.html): `node minhtml.mjs` on the readable build (HTML and CSS only; the script is left alone) |
| + terser | [3-fully-minified.html](bjj-case/3-fully-minified.html): JS through terser as well (also the spec's [test vector](../spec/vectors/bjj.html)) |
| + Roadroller | [4-roadroller.html](bjj-case/4-roadroller.html): the JS of the terser build packed with Roadroller |

Roadroller is itself a context-mixing compressor, so its output looks like noise to a language model: Surprisal goes
from 4,929 bytes to 17,300. Every classic method needs the whole pipeline to get under 19 KB, while Surprisal does a
quarter of that on the readable source.

Measured 2026-11-24 with [bjj-case/matrix.sh](bjj-case/matrix.sh), every Surprisal archive unpacked and compared
byte for byte. Each build is stored as `index.html`, so every archive carries the same file name.

## js13k games as submitted

Each game's files as its author submitted them. These are already packed: minified, and for miami-mice and
non-mewtonian-cat also Roadroller (the list is [corpus/roadroller.txt](corpus/roadroller.txt)). So the first column
is not readable source, and Surprisal's lead is smallest on the Roadroller games. Add a row whenever a game is
measured; keep the columns.

| file | as submitted | zip | 7z | Brotli | Surprisal `-9` |
|---|---:|---:|---:|---:|---:|
| js13k 2025: kittens-crossing | 17,069 | 12,626 | 12,962 | 12,322 | 🟩 **11,815** |
| js13k 2025: miami-mice | 19,876 | 12,450 | 12,593 | 11,808 | 🟩 **10,491** |
| js13k 2025: non-mewtonian-cat | 12,159 | 9,076 | 9,368 | 8,817 | 🟩 **8,539** |
| js13k 2025: tower-of-terror | 42,384 | 9,857 | 8,329 | 7,730 | 🟩 **1,356** |
| js13k 2025: wake-the-cat | 36,088 | 9,289 | 10,231 | 9,640 | 🟩 **2,734** |

js13k games come from [the corpus](corpus/) (`corpus/fetch.sh`). 2025 is the held-out year: the models never trained
on it. Measured 2026-11-23 on an M1 Max (Metal).

## How each method is measured

- **zip:** the smallest legal zip of the files: every 7-Zip zip method (Deflate, Deflate64, BZip2, LZMA, PPMd) and
  Zopfli (`advzip -4`). For js13k games this is `zBest` in [results.json](results.json).
- **7z:** `7z a -t7z -mx=9 -ms=on` over the files.
- **Brotli:** `brotli -q11 -w24` over the files (`brotli` in results.json for js13k games).
- **Surprisal:** `surprisal -9`. For js13k games, `go run ./benchmarks/run -level 9 -games <slugs>`, which writes the
  same bytes.

Surprisal's output doesn't depend on the machine.
