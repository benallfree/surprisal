# Compressed sizes

Compressed size in bytes, smaller is better. 🟩 marks the smallest in each column. ✓ marks the zip methods js13k
accepts: only Deflate (Zopfli writes ordinary Deflate, just smaller).

Every classic row is what `surprisal --compare` prints for the same files: the zip and 7z rows are 7-Zip at `-mx=9`
(`-tzip -mm=<method>`, or `-t7z` with LZMA2 or `-m0=PPMd`), Zopfli is `advzip -4`, and Brotli is `-q11 -w24` over the
files joined. Surprisal is `-9`. Its output doesn't depend on the machine.

## One game, every packing step

The fair comparison: the same game, BJJ, packed each way a js13k entrant might pack it, then compressed by each
method. Each column adds a step to the one before it. The builds are in [bjj-case/](bjj-case/).

| | unpacked | HTML minified | + terser | + Roadroller |
|---|---:|---:|---:|---:|
| no compression | 67,056 | 66,984 | 55,854 | 27,673 |
| zip Deflate ✓ | 24,987 | 24,950 | 21,841 | 19,126 |
| zip Deflate (Zopfli) ✓ | 24,872 | 24,835 | 21,768 | 18,867 |
| zip Deflate64 | 24,706 | 24,675 | 21,685 | 19,126 |
| zip BZip2 | 24,566 | 24,522 | 21,588 | 19,673 |
| zip LZMA | 23,673 | 23,645 | 20,548 | 19,337 |
| zip PPMd | 21,780 | 21,745 | 19,395 | 19,813 |
| 7z LZMA2 | 23,642 | 23,613 | 20,516 | 19,305 |
| 7z PPMd | 22,678 | 22,643 | 20,270 | 19,775 |
| Brotli | 22,758 | 22,732 | 19,723 | 18,427 |
| Surprisal `-9` | 🟩 **5,025** | 🟩 **5,026** | 🟩 **4,930** | 🟩 **17,301** |

| column | build |
|---|---|
| unpacked | [1-readable.html](bjj-case/1-readable.html): the source as written |
| HTML minified | [1b-html-minified.html](bjj-case/1b-html-minified.html): `node minhtml.mjs` on the readable build (HTML and CSS only; the script is left alone) |
| + terser | [3-fully-minified.html](bjj-case/3-fully-minified.html): JS through terser as well (also the spec's [test vector](../spec/vectors/bjj.html)) |
| + Roadroller | [4-roadroller.html](bjj-case/4-roadroller.html): the JS of the terser build packed with Roadroller |

The best build js13k would accept is Roadroller plus a Zopfli zip: 18,867 bytes. Surprisal does a quarter of that on
the readable source. Roadroller is itself a context-mixing compressor, so its output looks like noise to a language
model: Surprisal goes from 4,930 bytes to 17,301.

Measured with [bjj-case/matrix.sh](bjj-case/matrix.sh) on 2026-11-24, every Surprisal archive unpacked and compared
byte for byte; the Surprisal row re-measured 2026-11-25 with 0.0.1-alpha.2's `surprisal -9 --compare` (sizes only).
Each build is stored as `index.html`, so every archive carries the same file name.

## js13k games as submitted

Each game's files as its author submitted them. These are already packed: minified, and for miami-mice and
non-mewtonian-cat also Roadroller (the list is [corpus/roadroller.txt](corpus/roadroller.txt)). So the first row is
not readable source, and Surprisal's lead is smallest on the Roadroller games. Add a column whenever a game is
measured.

| | kittens-crossing | miami-mice | non-mewtonian-cat | tower-of-terror | wake-the-cat |
|---|---:|---:|---:|---:|---:|
| as submitted | 17,069 | 19,876 | 12,159 | 42,384 | 36,088 |
| zip Deflate ✓ | 12,848 | 12,680 | 9,284 | 10,893 | 10,785 |
| zip Deflate (Zopfli) ✓ | 12,626 | 12,450 | 9,076 | 10,516 | 10,688 |
| zip Deflate64 | 12,848 | 12,680 | 9,284 | 10,893 | 10,764 |
| zip BZip2 | 13,172 | 13,057 | 9,574 | 11,399 | 10,592 |
| zip LZMA | 12,994 | 12,957 | 9,399 | 10,863 | 10,263 |
| zip PPMd | 13,329 | 12,897 | 9,649 | 9,823 | 9,311 |
| 7z LZMA2 | 12,962 | 12,594 | 9,368 | 8,273 | 10,231 |
| 7z PPMd | 13,294 | 12,880 | 9,607 | 8,520 | 9,921 |
| Brotli | 12,323 | 11,792 | 8,812 | 7,665 | 9,636 |
| Surprisal `-9` | 🟩 **11,815** | 🟩 **10,491** | 🟩 **8,539** | 🟩 **1,356** | 🟩 **2,734** |

js13k games come from [the corpus](corpus/) (`corpus/fetch.sh`). 2025 is the held-out year: the models never trained
on it. The classic rows are `surprisal --compare` on each game's extracted files (2026-11-24); Surprisal is
`go run ./benchmarks/run -level 9 -games <slugs>` (2026-11-23, M1 Max, Metal), which writes the same bytes as
`surprisal -9`.

The [study](STUDY.md) reports a "best legal zip" that takes the smallest of every zip method. It predates knowing that
js13k accepts only Deflate, so read its zip figures as best-of-any-method.
