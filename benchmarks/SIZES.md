# Compressed sizes

Compressed size in bytes, smaller is better. 🟩 marks the smallest in each row. Add a row whenever a file is measured;
keep the columns.

| file | unpacked | zip | 7z | Brotli | Surprisal `-9` |
|---|---:|---:|---:|---:|---:|
| js13k 2025: kittens-crossing | 17,069 | 12,626 | 12,962 | 12,322 | 🟩 **11,815** |
| js13k 2025: miami-mice | 19,876 | 12,450 | 12,593 | 11,808 | 🟩 **10,491** |
| js13k 2025: non-mewtonian-cat | 12,159 | 9,076 | 9,368 | 8,817 | 🟩 **8,539** |
| js13k 2025: tower-of-terror | 42,384 | 9,857 | 8,329 | 7,730 | 🟩 **1,356** |
| js13k 2025: wake-the-cat | 36,088 | 9,289 | 10,231 | 9,640 | 🟩 **2,734** |
| BJJ | 55,854 | 19,395 | 20,516 | 19,723 | 🟩 **4,970** |

## How each column is measured

- **unpacked:** the files' total size, as the [runner](run/) counts it.
- **zip:** the smallest legal zip of the files: every 7-Zip zip method (Deflate, Deflate64, BZip2, LZMA, PPMd) and
  Zopfli (`advzip -4`). For js13k games this is `zBest` in [results.json](results.json).
- **7z:** `7z a -t7z -mx=9 -ms=on` over the extracted files.
- **Brotli:** `brotli -q11 -w24` over the files (`brotli` in results.json for js13k games).
- **Surprisal:** `go run ./benchmarks/run -level 9 -games <slugs>` (the same bytes `surprisal -9` writes).

js13k games come from [the corpus](corpus/) (`corpus/fetch.sh`). 2025 is the held-out year: the models never trained
on it. BJJ is a fully minified 3D game, [spec/vectors/bjj.html](../spec/vectors/bjj.html).

Measured 2026-11-23 on an M1 Max (Metal). Surprisal's output doesn't depend on the machine.
