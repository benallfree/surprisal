# Wall clock and archive size, every compressor

One file: the minified BJJ game (`bjj-case/3-fully-minified.html`, 55,854 bytes; 24,126 Qwen tokens). Every row is a
real archive, compressed and then decompressed in a separate process, with the output compared byte for byte.
Measured on the M1 Max (64 GB), 2026-11-21. Model load time is left out: 1–3 s once the model is on disk.

| compressor | archive bytes | compress | decompress |
|---|---:|---:|---:|
| Qwen2.5-Coder-7B bf16, MLX, 1 segment | 4,928 | 22.5 min | 22.5 min |
| Qwen3-4B-Base bf16, MLX | 5,166 | 17.4 min | 17.5 min |
| Qwen2.5-Coder-1.5B bf16, MLX | 5,856 | 6.9 min | 6.8 min |
| Qwen2.5-Coder-7B bf16, MLX, 16 segments | 7,026 | 5.6 min | 6.5 min |
| Qwen2.5-Coder-0.5B bf16, MLX | 7,186 | 4.8 min | 4.7 min |
| context mixer, trained on 2020–24 js13k entries (one-time pretraining 106 s) | 16,545 | 209 ms | 209 ms |
| context mixer, untrained | 18,284 | 430 ms | 467 ms |
| zip PPMd order 6 (7-Zip) | 19,330 | 90 ms | 57 ms |
| brotli -q11 | 19,723 | 302 ms | 27 ms |
| xz -9e | 20,448 | 69 ms | 20 ms |
| zip LZMA (7-Zip) | 20,548 | 54 ms | 33 ms |
| zstd --ultra -22 | 21,278 | 52 ms | 23 ms |
| bzip2 -9 | 21,476 | 24 ms | 18 ms |
| zip Zopfli (advzip -4) | 21,768 | 258 ms | 11 ms |
| zip Deflate (7-Zip, 15 passes) | 21,841 | 105 ms | 9 ms |
| gzip -9 | 22,392 | 22 ms | 15 ms |

- LLM rows: a real arithmetic coder driven one token at a time (`llm/realcodec.py`; segments: `llm/segcodec.py`).
  Encoder and decoder must compute identical probabilities, so compressing is as slow as decompressing.
- Segments: the tokens are cut into N parts stepped together in one batched forward pass, each with its own coded
  stream. Each part starts with no context.
- Classic rows: `node time_classic.mjs <file>`, median of 3 runs, whole-process wall clock. They ran while a 7B
  encode was using the GPU, so they may read slightly high.

## How fast can a model decode at all?

Tokens per second when N streams step together in one forward pass on MLX (`llm/batchbench.py` and a forward-only
check, 2026-11-21). The BJJ game is 24,126 tokens.

| model | 1 stream | 8 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B, 4-bit (local `mlx_lm convert -q`) | 61 | 79 | 189 | 210 | 201 | 238 |
| Qwen2.5-Coder-7B, bf16 | 21 | 52 | 127 | | | |
| Qwen2.5-Coder-0.5B, bf16 | 121 | 508 | 943 | | 1,588 | |

- The 7B's ceiling is about 240 tok/s on the M1 Max, around 3.6 TFLOPS of real compute. Any scheme that runs the 7B
  on every token (segments, speculation) needs at least 100 s for this file.
- MLX's batched matmul is slow for 2–32 rows: the 7B takes 20 ms per step for 1 row and 162 ms for 8.
- The integer frequency table on the GPU costs 1–2 ms per step; the CPU (numpy float64) version was the main
  overhead in `segcodec.py`.

## Full-size archives

No segments, no window: each model at its full-context size. The BJJ game is 24,126 tokens (Qwen2.5 and Qwen3 give
the same count). "Fast" is `llm/fastcodec.py` (frequency table on the GPU), "old" is `llm/realcodec.py`. All measured rows round-tripped byte for byte.

| model | codec | archive bytes | compress | decompress | measured tok/s | weights per token |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B-Base bf16 | (scored only) | 4,821 | — | ~32 min (est.) | ~12 | 16.4 GB |
| Qwen2.5-Coder-7B bf16 | old | 4,928 | 22.5 min | 22.5 min | 17.8 | 15.2 GB |
| Qwen2.5-Coder-7B 8-bit | fast | 4,929 | 14.9 min | 14.8 min | 27.1 | 7.6 GB |
| Qwen2.5-Coder-7B 4-bit | fast | 5,127 | 10.6 min | 10.7 min | 37.7 | 4.0 GB |
| Qwen3-4B-Base bf16 | old | 5,166 | 17.4 min | 17.5 min | 22.9 | 8.0 GB |
| Qwen2.5-Coder-3B bf16 | (scored only) | 5,256 | — | — | — | 6.2 GB |
| Qwen2.5-Coder-3B 8-bit | fast | 5,262 | 7.5 min | 7.6 min | 52.8 | 3.1 GB |
| Qwen2.5-Coder-3B 4-bit | fast | 6,833 | 5.9 min | 6.1 min | 65.8 | 1.6 GB |
| Qwen2.5-Coder-1.5B bf16 | fast | 5,856 | 7.2 min | 6.6 min | 61.0 | 3.1 GB |
| Qwen2.5-Coder-1.5B 8-bit | fast | 5,858 | 4.7 min | 5.2 min | 78.0 | 1.5 GB |
| Qwen2.5-Coder-1.5B 4-bit | fast | 6,374 | 4.0 min | 4.3 min | 94.3 | 0.84 GB |
| Qwen2.5-Coder-0.5B bf16 | fast | 7,186 | 3.1 min | 3.1 min | 128.9 | 0.99 GB |
| Qwen2.5-Coder-0.5B 8-bit | fast | 7,194 | 4.3 min | 3.5 min | 115.7 | 0.51 GB |
| Qwen2.5-Coder-0.5B 4-bit | fast | 8,284 | 4.6 min | 3.4 min | 118.3 | 0.28 GB |

- 8-bit keeps the size (7B 8-bit is 1 byte over bf16, 3B and 1.5B within 6 bytes) and cuts the 7B from 22.5 to
  14.8 min. 4-bit costs 4–30% in size.
- The small models are not limited by bandwidth on this Mac. The 0.5B at 4-bit needs 320 GB/s and the M1 Max has 400,
  yet it runs at 118 tok/s: about 8 ms per token goes to fixed costs (Python, kernel launches, the CPU waiting for
  the GPU). Going faster means cutting those fixed costs with a compiled decode loop (CUDA graphs or `mx.compile`),
  not more bandwidth.
- The encoder and decoder must compute identical probabilities, so a shipped codec runs the same model, precision
  and kernels on both ends.

## Speed work on the same Mac (2026-11-21, evening)

Three ideas tried on the 0.5B and 1.5B, without giving up size:

| change | file | result |
|---|---|---|
| Pipelining + token search on the GPU (4 numbers back per token, next step queued before the CPU reads the token) | `llm/fastcodec2.py` | byte-identical archives; +3% on a 6 KB slice (bf16 0.5B) |
| Whole step compiled with `mx.compile`, KV cache preallocated in 2K buckets | `llm/fastcodec3.py` | 0.5B 8-bit: 116 → 191 tok/s on the slice, but **115 tok/s on the full file** (7,193 bytes, round trip OK). No gain at full length. |
| Output layer pruned to the 42,458 tokens the training corpus uses (speed only, no escape code yet) | scratch bench | 0.5B 8-bit 170 → 200 tok/s (+18%); 1.5B bf16 72 → 80 tok/s (+11%) |

Why the full file doesn't speed up: speed falls with context depth, compiled or not (0.5B 8-bit, tok/s):

| position | 0 | 4K | 8K | 16K | 22K |
|---|---:|---:|---:|---:|---:|
| compiled step | 179 | 160 | 152 | 105 | 117 |
| plain mlx_lm cache, pipelined | 173 | 162 | 137 | 129 | 118 |

- Pipelining alone (plain mlx_lm, GPU-side choice) already reaches the compiled step's speed: the CPU wasn't the
  bottleneck. The GPU spends about 5.5 ms per token at short context and 8.5 ms at 22K.
- BJJ has 124 tokens (54 distinct) outside the corpus vocabulary, so a pruned codec needs an escape code: about
  7–8 bits per escape plus that token's cost, roughly +100–150 bytes before any gain from renormalizing.
- Best case on this Mac with all three: around 150–200 tok/s on the 0.5B, so about 2–2.5 min per file.

### Pipelining fix, and where it stops (2026-11-22, 01:00)

The real win came from one change in `llm/fastcodec2.py`: submit the coder's four numbers in their **own** MLX command
buffer before queuing the next forward pass. MLX signals an array only when its whole command buffer finishes, so
reading the token used to wait for the next forward pass too. Archives stay byte-identical to `fastcodec.py` (checked
on every full-size row: all ten decode and re-encode to the same bytes).

Back-to-back A/B on a 6 KB slice (2,341 tokens), same archive, alternating old and new twice (decode tok/s):

| model | old `fastcodec.py` | new `fastcodec2.py` | gain |
|---|---:|---:|---:|
| 0.5B 8-bit | 112 | 223–234 | **2.0×** |
| 1.5B bf16 | 48–61 | 64–65 | +5–30% |
| 3B 8-bit | 59–65 | 64–69 | +7% |
| 7B 8-bit | 32 | 34 | +5% |

- The 0.5B now runs at mlx_lm's own generate speed (256 tok/s for this model) less about 10%. The bigger models
  were already GPU-bound, so they gain only 5–7%.
- The first full-file re-run (`llm/fast2-shared-gpu.log`) was distorted by another session's GPU work. Its
  byte-identity checks stand; its timings don't. The clean re-run is below.
- Tried and dropped: compiling the whole step (`llm/fastcodec3.py`, padded KV buckets make attention slower),
  fused q/k/v and gate/up matmuls plus compiled blocks (`llm/fastmodel.py`, no gain), vocabulary pruning to the
  corpus's 42,458 tokens (+11–18% speed but +1.1% size, `llm/prune_cost.py`: BJJ's escapes are English words).
- **Diminishing returns reached in Python/MLX.** What's left is inside MLX's GPU kernels: about 4 ms per token for
  the 0.5B at short context where reading its weights takes about 1.5 ms. Closing that needs fused custom kernels
  (one kernel per layer), which is engine work (nollama's Metal path), not codec work.

### Clean full-file timings, `fastcodec2.py` (2026-11-22, 01:57–03:57)

Full BJJ game (55,854 bytes, 24,126 tokens). The Mac had the GPU to itself (`llm/fast2.log`). Every row decoded
the existing archive byte for byte, and re-encoding gave a byte-identical archive.

| model | archive bytes | compress | decompress | compress tok/s | decompress tok/s (old) |
|---|---:|---:|---:|---:|---:|
| 0.5B bf16 | 7,186 | 2.4 min | 2.7 min | 165.7 | 147.0 (128.9) |
| 0.5B 8-bit | 7,194 | 2.1 min | 2.6 min | 192.2 | 155.5 (115.7) |
| 0.5B 4-bit | 8,284 | 1.8 min | 2.4 min | 218.9 | 169.1 (118.3) |
| 1.5B 4-bit | 6,374 | 2.9 min | 3.6 min | 140.9 | 110.7 (94.3) |
| 1.5B 8-bit | 5,858 | 3.4 min | 4.6 min | 118.6 | 88.3 (78.0) |
| 1.5B bf16 | 5,856 | 5.2 min | 5.7 min | 76.8 | 70.1 (61.0) |
| 3B 4-bit | 6,833 | 4.3 min | 5.3 min | 92.8 | 75.2 (65.8) |
| 3B 8-bit | 5,262 | 6.6 min | 8.5 min | 60.7 | 47.6 (52.8) |
| 7B 4-bit | 5,127 | 10.5 min | 12.8 min | 38.2 | 31.4 (37.7) |
| 7B 8-bit | 4,929 | 13.7 min | 17.3 min | 29.3 | 23.2 (27.1) |

- Up to 1.5B and 3B 4-bit, decompression is 13–43% faster than the old codec, and compression is faster still.
  The best size-keeping row on this Mac is now the 1.5B 8-bit: 5,858 bytes in 4.6 min.
- The 3B 8-bit and both 7B decodes came out 10–17% *slower* than the old codec's earlier timings. Their encodes
  weren't slower, and the slice A/B above shows the new loop is +5–7% faster on these models. A full-file 7B 4-bit
  decode of old against new (`llm/ab7b.log`) was meant to settle it, but Spotlight indexing loaded the Mac during
  that run, and both came out about 5× slower. Under the same load the new codec was still faster: 7.1 against
  5.5 tok/s, +29%. Best reading: sustained load (heat, or other processes) on the long big-model runs, not a
  regression. Archives are identical either way, so the size column is unaffected.


## Measured on NVIDIA GPUs: `llm/cudacodec.py` on Modal (2026-11-22)

The same coder (Witten-Neal-Cleary, frequency total 2^28) driven from PyTorch on rented GPUs, billed per second
(`.venv/bin/modal run llm/cudacodec.py --gpu H200 --models 0.5B,1.5B,3B,7B`). Qwen2.5-Coder base models in bf16, the
full BJJ game (55,854 bytes, 24,126 tokens). Every row decoded back to the input byte for byte. Logs:
`llm/cuda-H200.log`, `llm/cuda-H100.log`, `llm/cuda-h200.out`, `llm/cuda-h100-0.5b.out` (profile).

**The 0.5B decompresses the file in 27.0 s on an H200 and 29.0 s on an H100.**

| model (bf16) | archive bytes | MLX bf16 bytes | GPU | compress | decompress | decompress tok/s |
|---|---:|---:|---|---:|---:|---:|
| 0.5B | 7,183 | 7,186 | H200 | 26.3 s | **27.0 s** | **893** |
| 0.5B | 7,183 | 7,186 | H100 | 28.2 s | **29.0 s** | **832** |
| 1.5B | 5,855 | 5,856 | H200 | 46.1 s | 46.8 s | 516 |
| 3B | 5,276 | 5,256 (scored) | H200 | 69.0 s | 69.5 s | 347 |
| 7B | 4,991 | 4,928 | H200 | 113.5 s | 114.2 s | 211 |

- The times are the codec loop only. A cold run also pays for the container (a few seconds once the image is
  cached), loading weights (1.5 s for the 0.5B, 11 s for the 7B from Modal's volume) and capturing the CUDA graphs
  (13–17 s, mostly `torch.compile`; it could be cached). So a cold 0.5B decompress is about 50 s end to end today.
- Archive sizes differ from MLX by a few bytes (the 7B by 1.3%): the kernels differ, so the probabilities differ
  slightly. Each archive only decodes with the same code on the same kind of GPU.
- The 7B runs at 211 tok/s × 15.2 GB of weights = 3.2 TB/s, 67% of the H200's 4.8 TB/s, which matches the
  70% that real kernels usually reach. The 0.5B is limited by kernel count, not bandwidth.
- What it took (H100, 0.5B, full file):
  1. Plain PyTorch in a CUDA graph: 231 tok/s. About 700 tiny kernels per token.
  2. `torch.compile` to fuse norms, rope and the MLP: 510.
  3. Chunked "flash-decoding" attention: 577. A [2 heads × 7 queries] × 24K matmul left most of the GPU idle.
  4. One cache write per layer instead of two: 644.
  5. A custom Triton attention kernel (one program per head and 256-token chunk, then a merge): 714.
  6. A Triton cache store instead of `index_put_`, which hid 2 memory copies per layer: **832**.
- How it's built: the whole step (forward pass, frequency table, the decoder's search) is replayed as CUDA graphs,
  one per 1,024-token bucket. Compression is graph replays only, with no CPU round trip. Decompression queues the next
  forward pass before reading the found token, so the CPU's coder update overlaps the GPU.
- The bigger models are limited by bandwidth: they get faster only with more of it (B200) or smaller weights.
- Cost: Modal bills the H200 at $4.54/h and the H100 at $3.95/h, by the second. The final four-model H200 run took
  10 min (about $0.76). A 0.5B decompress on an H100 is about 3¢ of GPU time plus the cold start.

### B200 (2026-11-22, 12:15)

Same script, no B200-specific tuning (`llm/cuda-B200.log`):

| Model | GPU | Archive | Compress | Decompress | tok/s |
|---|---|---:|---:|---:|---:|
| 0.5B | B200 | 7,184 | 28.5 s | 29.6 s | 815 |
| 1.5B | B200 | 5,853 | 45.1 s | 46.1 s | 523 |

- The B200 buys nothing at this batch size of one: the 0.5B is slower than on the H200 (893 tok/s), and the 1.5B is level (516). Its extra bandwidth doesn't help kernels that are bound by launch latency and small reductions.
- The archives differ from the H200's by a byte or two (7,184 vs 7,183; 5,853 vs 5,855). The same model on another GPU gives different floats, so these archives are tied to the GPU type as well as the model. A codec that depends only on the model needs integer (fixed-point) inference; see the README's fine print.

## Archives that depend on the model only: integer inference (2026-11-22, afternoon)

Float inference gives different last bits on every GPU and library (the B200 archives above differ from the H200's), so
an arithmetic-coded archive only decodes on the hardware that made it. `llm/intref.py` defines a Qwen2.5 forward pass in
integer arithmetic only, so any implementation on any hardware computes the same frequency table, bit for bit:

- A fixed bundle built once from the bf16 weights (`intref.py build`, 491 MB for the 0.5B): int8 weights with an integer
  scale per 64 inputs, integer biases and norm weights, and stored tables (rope cos/sin, exp, sigmoid). The bundle is the model.
- Every step is integer and independent of order: integer sums, max, shifts, truncating division, `isqrt`, table lookups.
  Activations enter matmuls as 16-bit mantissas with one shared exponent; the residual stream is int64 at 2^-24;
  the KV cache is clipped to 2^22 so every sum stays below 2^53. Start token: the config's `bos_token_id`.
- `intref.py` (numpy) is the spec. `llm/intgpu.py` is the same thing in PyTorch (any device) and, with `--fast`,
  in Triton kernels replayed as CUDA graphs (about 10 launches per layer).

**Size costs nothing.** On the 6 KB slice (ideal sizes, before the 4-byte header): float64 652.1 bytes, float64
with the bundle's int8 weights 651.8, full integer 651.8. On the full BJJ file (55,854 bytes, 24,126 tokens): **7,185
bytes**, against 7,183 for the bf16 float codec on an H100/H200 and 7,184 on a B200.

**Cross-machine checks, all exact:**
- Frequency-table hashes for the first 30 tokens match across numpy on the Mac, PyTorch on the Mac CPU, PyTorch on an
  H100, and the Triton path on an H100.
- The slice encoded on an H100 decodes on the Mac (numpy) back to the same 6,000 bytes; the slice encoded on the Mac
  is byte-identical to the H100's archive (655 bytes).
- The full file round-trips on an H100, and the archive is identical across three kernel versions.

**Speed (0.5B, full file, `llm/int-H100.log`, `llm/int-H200.log`):**

| Path | GPU | Compress | Decompress | tok/s |
|---|---|---:|---:|---:|
| bf16 float codec (`cudacodec.py`, for reference) | H100 | 28.2 s | 29.0 s | 832 |
| bf16 float codec | H200 | 26.3 s | 27.0 s | 893 |
| integer, Triton + CUDA graphs | H100 | 28.9 s | **29.6 s** | 815 |
| integer, Triton + CUDA graphs | H200 | 28.0 s | **28.7 s** | 842 |
| integer, plain PyTorch (float64 matmuls) | H100 | — | — | 7 |
| integer, numpy spec on the M1 Max CPU | — | — | — | 6 |

- The H100 and H200 archives are byte-identical (7,185 bytes). The float codec's archives differ between GPU types.
- Cold start: about 17 s to load the bundle and capture 49 graphs (once Triton's cache is warm).
- A forward step takes 0.94 ms at 1K context, 1.26 ms at 13K and 1.44 ms at 24K (H100).
- The Mac only has the numpy spec today (6 tok/s), so decoding the full file there would take over an hour. A Metal
  version is the next step for the Mac.

How it got faster (H100, decode tok/s; the ladder starts at the slice and ends on the full file):
1. Plain PyTorch: 7.
2. Triton int8 matvec, with torch.compile for the integer glue: 364 on the slice.
3. Split-K, plus exact float64 attention dot products: 505.
4. Hand-fused integer kernels, using integer atomics to combine split-K partials and attention chunks: 657.
5. Dropped stray fill kernels, gave attention a fixed program count: 819 on the slice; 529 on the full file, where
   long context dominated.
6. KV cache stored as three int8 digit planes; attention as int8 tensor-core dots over all 7 q heads per kv head:
   24K-context step 2.58 → 1.59 ms.
7. Tuned matvec splits, `clz` for bit lengths, float64-estimate division with exact integer correction in the
   norm (5.2 → 3.3 µs), about 192 attention programs per kv head: 30.3 s.
8. Output table as a matvec epilogue plus three small kernels, instead of six generic ones: 29.6 s.

One spec change along the way: K and V are now clipped at ±(2^22 − 1), so their top byte fits in int8. Real values
stay below 2^20, and every archive was byte-identical before and after.

### Integer 1.5B (2026-11-22, evening)

Same code, bundle `coder1.5b.npz` (1.5 GB). The attention scale is now `round(2^16 / sqrt(head_dim))` as an integer
(the 1.5B has head_dim 128). For head_dim 64 that is exactly 2^13, the old rule, so the 0.5B archives are unchanged.

| Model | GPU | Archive | Compress | Decompress | tok/s |
|---|---|---:|---:|---:|---:|
| 1.5B float (`cudacodec.py`) | H200 | 5,855 | 46.1 s | 46.8 s | 516 |
| 1.5B integer | H200 | **5,855** | 46.5 s | 47.2 s | 511 |
| 1.5B integer | H100 | **5,855** | 50.8 s | 53.5 s | 451 |

- **Size:** the same as float, byte for byte in length, and 18.5% smaller than the 0.5B's 7,185.
- **Exactness:** the first 30 tables match the numpy spec on the Mac, and the full file round-trips.
- **Speed:** the step takes 1.65 ms at 1K context and 2.40 ms at 24K (H100).
  - The int8 weights are 1.54 GB per token: 0.46 ms at the H100's peak bandwidth. The matvecs take 0.74 ms.
  - The other kernels (two norms, qkv epilogue, two attention passes, merge, MLP per layer) add about 1 ms across
    28 layers.
  - Attention adds about 0.75 ms by 24K.
  - Tuning launch settings barely moves this. Bigger gains need fusing the norms into the matvecs and a different
    attention design.
- **Exact skip of empty attention chunks:** a chunk whose best score is at least 11,357/1024 nats below the head's max
  has every weight `EXPA[...] = 0` by definition, so the weighted-sum pass skips it. This is exact. It bought only
  about 1–5%: attention on this file is not sparse enough.

### Batched compression (2026-11-22, night)

The compressor knows every token in advance, so it can run all positions as matrix-matrix work, prefill-style, in rows of
2,048 positions. Under the integer spec a batched forward pass computes exactly the same tables as the step-by-step one.
Every archive below matched the step-by-step archive byte for byte, and each was round-tripped.

| Model | Archive | Batched compress (H100) | Step-by-step compress | Speedup |
|---|---:|---:|---:|---:|
| 0.5B | 7,185 | **2.04 s** | 28.9 s | 14× |
| 1.5B | 5,855 | **3.82 s** | 50.8 s | 13× |
| 7B | 4,969 | **9.79 s** | ~120 s (decode measured: 120.8 s) | 12× |

- Not counted: about 20 s for the first run to compile the Triton kernels (cached afterwards), plus loading the bundle
  (25 s for the 7B's 7.6 GB from Modal's volume).
- How it stays exact:
  - **GEMMs:** int8 tensor-core dots. Activations are split into two offset bytes (`x = 256 h + l + 128`); the offset
    comes back through precomputed per-group sums of the int8 weights. The integer group scales are applied in float64,
    which is exact: each term is below 2^43 and even 296 groups sum below 2^52.
  - **Attention:** two passes over the keys per 16 query rows and one q head. The digit products give q·K exactly
    (below 2^44), and the score `floor(q·K · 2^-n + 1/2)` is exact in float64. A tile whose scores are all at least
    11,357/1024 nats below the row max has every weight `EXPA[…] = 0` by the spec, so it's skipped.
  - **Output head:** three passes (max, Σ exp, frequencies with the running count up to the target token), recomputing
    the head GEMM each time so the 152K logits per position are never stored.
- **One spec change:** the attention scale now multiplies q *before* its block-float rounding, which makes the
  float64 score path exact. For head_dim 64 (the 0.5B) the scale is 2^13 and nothing changes: the 0.5B archives are
  byte-identical before and after. The 1.5B and 7B tables changed, re-checked against the numpy spec (1.5B, 30
  tables) and by round trip. The 1.5B archive kept its size; the 7B's moved from 4,972 to 4,969 bytes.
- Where the 7B's 9.8 s goes: attention 6.5 s, GEMMs 2.3 s, output head 0.7 s. Attention runs at a small fraction
  of the int8 tensor-core peak (two passes, three-digit operands, float64 bookkeeping). The GEMMs reach about 15%,
  held back by the per-64-group scale step.
- Possible next steps, both spec changes:
  - Attention in one pass, using integer online-softmax rescaling with a fixed chunk order defined by the spec.
  - Larger quantization groups (fewer scale steps in the GEMMs), which needs a size check first.

### Other corpora: a C++ source tree (2026-11-22, night)

`meshenvy/envyos/packages/meshcore/src`: 166 C++ files (`.cpp`/`.h`), 863,664 bytes, 329,238 tokens. Each file was
compressed on its own with the integer 7B (committed codec 9d9bc1a, batched), fresh context per file. The one
105 KB font-data file needed four 32K-token segments.

| Method | Bytes | vs zip |
|---|---:|---:|
| zip -9 | 267,227 | 1.00 |
| raw deflate -9 per file (no zip headers) | 245,223 | 0.92 |
| tar + xz -9e (solid) | 160,692 | 0.60 |
| tar + brotli -11 (solid) | 159,744 | 0.60 |
| **integer 7B, per file** (streams 40,348 + 4-byte lengths + xz'd file list 1,140) | **42,152** | **0.158** |

- That is 0.34 bits per byte on `.cpp` and 0.45 on `.h`: 6.3× smaller than zip and 3.8× smaller than xz or brotli,
  even though the solid baselines see the whole tree and the 7B sees one file at a time.
- 106 s of H100 time for the whole tree, after loading and kernel compilation. Three files round-tripped through the
  step-by-step decoder, including `Mesh.cpp` (28,701 B → 1,073 B).
- Memorization check: part of this tree is public code that predates the model. But brand-new code compresses about as
  well: nollama's Go source, written days before this run, came out at 0.47 bits per byte (partial run, 138 files).
  Readable code is simply very predictable to a 7B coder model; the minified BJJ game is much harder.

Unseen-code control, `nollama/engine` (first commit 2026-11-21, long after the model's training data): 111 hand-written
files (Go, CUDA, Metal, assembly, one JSON; the 2.5 MB of generated PTX left out), 891,145 bytes, 341,417 tokens.

| Method | Bytes | 7B is smaller by |
|---|---:|---:|
| zip (default) | 312,600 | 5.6× |
| zip -9 | 306,344 | 5.5× |
| tar + xz -9e | 208,100 | 3.7× |
| tar + brotli -11 | 204,953 | 3.7× |
| **integer 7B, per file** | **56,038** | |

By language: Go 0.50 bits/byte, CUDA 0.40, Metal 0.37, assembly 0.31, JSON 1.64. 154 s of H100 time, with
`kvcache.go` and `kvcache_test.go` round-tripped through the step-by-step decoder. Code the model has never seen
compresses almost as well as meshcore (5.5× vs 6.3× against zip -9), so memorization explains little of that result.

### Tried: one-pass attention (2026-11-22, late; branch `one-pass-attention`)

The idea was a spec in which attention takes one pass over fixed 64-key chunks in ascending order, with integer
online-softmax rescaling: `(x·f + 2^14) >> 15`, exact in int64 via a 32-bit split. The batched compressor then needs
only one pass over the keys. Step-by-step decode keeps its chunk parallelism by summing chunks within runs of equal
running max and rescaling only between runs.

- It is exact and was fully verified:
  - numpy, plain PyTorch and both GPU paths agree over 300 tables (five chunks);
  - the slice archives match numpy;
  - the 0.5B full file round-trips.
- It is not worth it:
  - 7B batched compress went from 9.79 to 9.50 s (attention 6.47 → 6.17 s);
  - 0.5B batched compress went from 2.04 to 2.25 s (some row's max rises in nearly every tile, so the int64 rescale
    runs almost every chunk);
  - step-by-step decode went from 29.6 to 32.7 s (two more kernels per layer).
- Not adopted; the two-pass global-max rule stays.
- Lesson: score arithmetic isn't what limits batched attention. Each 16-row, one-head program re-reads the K/V digit
  planes, so traffic scales with heads × row blocks. The next attempt should share the K/V tiles across all 7 q
  heads of a kv head, and across more rows.
- Found along the way: a kernel that reads a value and then zeroes it in the same program needs a barrier in between,
  because scalars and small vectors are held by several threads. It caused nondeterministic tables in the one-pass
  decode merge. The main code's one such site (the MLP-max reset in the norm kernel) now has the barrier.

Solid, the simple way: the same 111 files as one ustar tar (977,408 bytes, 425,346 tokens; the extra 84K tokens are
mostly NUL padding), compressed as one document in 14 windows of 32K tokens (fresh context per window):
**48,999 bytes**, 12.6% smaller than one file at a time. It also restores the exact tar, names and metadata included.
That is 6.3× smaller than zip -9 and 4.2× smaller than tar + xz -9e (208,100). 211 s of H100 time, including about
20 s of kernel compilation.
