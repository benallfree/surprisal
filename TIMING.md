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

## How fast can a model decode at all? (feasibility for a 30-second decoder)

Tokens per second when N streams step together in one forward pass on MLX (`llm/batchbench.py` and a forward-only
check, 2026-11-21). The BJJ game is 24,126 tokens, so 30 seconds needs about 800 tok/s.

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

## Full-size archives against the 30-second target

No segments, no window: each model at its full-context size. The BJJ game is 24,126 tokens (Qwen2.5 and Qwen3 give
the same count), so decompressing it in 30 seconds takes **804 tok/s** in every row. "Fast" is `llm/fastcodec.py`
(frequency table on the GPU), "old" is `llm/realcodec.py`. All measured rows round-tripped byte for byte.

Bandwidth needed = weights read per token × 804 tok/s ÷ 0.7 (real kernels reach about 70% of a GPU's published
memory bandwidth). It assumes near-zero per-token software overhead. GPU bandwidths are published specs.

| model | codec | archive bytes | compress | decompress | measured tok/s | needed tok/s | weights per token | bandwidth needed | a GPU that has it |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-8B-Base bf16 | (scored only) | 4,821 | — | ~32 min (est.) | ~12 | 804 | 16.4 GB | 18.8 TB/s | none alone: 3× B200 (8 TB/s each) or an SRAM chip (Groq, Cerebras) |
| Qwen2.5-Coder-7B bf16 | old | 4,928 | 22.5 min | 22.5 min | 17.8 | 804 | 15.2 GB | 17.5 TB/s | none alone: 3× B200 |
| Qwen2.5-Coder-7B 8-bit | fast | 4,929 | 14.9 min | 14.8 min | 27.1 | 804 | 7.6 GB | 8.7 TB/s | B200 / B300 (8 TB/s) just short; 2× H200 |
| Qwen2.5-Coder-7B 4-bit | fast | 5,127 | 10.6 min | 10.7 min | 37.7 | 804 | 4.0 GB | 4.6 TB/s | H200 (4.8 TB/s), MI300X (5.3 TB/s) |
| Qwen3-4B-Base bf16 | old | 5,166 | 17.4 min | 17.5 min | 22.9 | 804 | 8.0 GB | 9.2 TB/s | 2× H200; B200 just short |
| Qwen2.5-Coder-3B bf16 | (scored only) | 5,256 | — | — | — | 804 | 6.2 GB | 7.1 TB/s | B200 (8 TB/s) |
| Qwen2.5-Coder-3B 8-bit | fast | 5,262 | 7.5 min | 7.6 min | 52.8 | 804 | 3.1 GB | 3.6 TB/s | H200; H100 SXM (3.35 TB/s) just short |
| Qwen2.5-Coder-3B 4-bit | fast | 6,833 | 5.9 min | 6.1 min | 65.8 | 804 | 1.6 GB | 1.8 TB/s | RTX 5090 (1.79 TB/s), H100 |
| Qwen2.5-Coder-1.5B bf16 | fast | 5,856 | 7.2 min | 6.6 min | 61.0 | 804 | 3.1 GB | 3.6 TB/s | H200; H100 SXM just short |
| Qwen2.5-Coder-1.5B 8-bit | fast | 5,858 | 4.7 min | 5.2 min | 78.0 | 804 | 1.5 GB | 1.7 TB/s | RTX 5090 |
| Qwen2.5-Coder-1.5B 4-bit | fast | 6,374 | 4.0 min | 4.3 min | 94.3 | 804 | 0.84 GB | 0.97 TB/s | RTX 4090 (1.0 TB/s), RTX 5090 |
| Qwen2.5-Coder-0.5B bf16 | fast | 7,186 | 3.1 min | 3.1 min | 128.9 | 804 | 0.99 GB | 1.1 TB/s | RTX 5090 |
| Qwen2.5-Coder-0.5B 8-bit | fast | 7,194 | 4.3 min | 3.5 min | 115.7 | 804 | 0.51 GB | 0.59 TB/s | RTX 4090; M4 Max (546 GB/s) just short |
| Qwen2.5-Coder-0.5B 4-bit | fast | 8,284 | 4.6 min | 3.4 min | 118.3 | 804 | 0.28 GB | 0.32 TB/s | this M1 Max (400 GB/s) has the bandwidth |

- 8-bit keeps the size (7B 8-bit is 1 byte over bf16, 3B and 1.5B within 6 bytes) and cuts the 7B from 22.5 to
  14.8 min. 4-bit costs 4–30% in size. The best size-preserving row is the 7B 8-bit: 18× too slow for 30 s.
- The small models are not limited by bandwidth on this Mac. The 0.5B at 4-bit needs 320 GB/s and the M1 Max has 400,
  yet it runs at 118 tok/s: about 8 ms per token goes to fixed costs (Python, kernel launches, the CPU waiting for
  the GPU). Hitting 804 tok/s needs about 1.2 ms per token all-in, which in practice means a compiled decode loop
  (CUDA graphs or `mx.compile`), not more bandwidth.
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
- Best case on this Mac with all three: around 150–200 tok/s on the 0.5B, so about 2–2.5 min per file. 30 seconds
  needs 804 tok/s, 4–5× more than any of these reach.

### Pipelining fix, and where it stops (2026-11-22, 01:00)

The real win came from one change in `llm/fastcodec2.py`: submit the coder's four numbers in their **own** MLX command
buffer before queuing the next forward pass. MLX signals an array only when its whole command buffer finishes, so
reading the token used to wait for the next forward pass too. Archives stay byte-identical to `fastcodec.py` (checked
on every full-size row: all ten decode and re-encode to the same bytes).

Back-to-back A/B on a 6 KB slice (2,341 tokens), same archive, alternating old and new twice (decode tok/s):

| model | old `fastcodec.py` | new `fastcodec2.py` | gain | still needed for 30 s on the full file |
|---|---:|---:|---:|---:|
| 0.5B 8-bit | 112 | 223–234 | **2.0×** | 804 tok/s (3.5× more) |
| 1.5B bf16 | 48–61 | 64–65 | +5–30% | 804 (12× more) |
| 3B 8-bit | 59–65 | 64–69 | +7% | 804 (12× more) |
| 7B 8-bit | 32 | 34 | +5% | 804 (24× more) |

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
the existing archive byte for byte, and re-encoding gave a byte-identical archive. 30 s needs **804 tok/s**.

| model | archive bytes | compress | decompress | compress tok/s | decompress tok/s (old) | still needed for 30 s |
|---|---:|---:|---:|---:|---:|---:|
| 0.5B bf16 | 7,186 | 2.4 min | 2.7 min | 165.7 | 147.0 (128.9) | 5.5× |
| 0.5B 8-bit | 7,194 | 2.1 min | 2.6 min | 192.2 | 155.5 (115.7) | 5.2× |
| 0.5B 4-bit | 8,284 | 1.8 min | 2.4 min | 218.9 | 169.1 (118.3) | 4.8× |
| 1.5B 4-bit | 6,374 | 2.9 min | 3.6 min | 140.9 | 110.7 (94.3) | 7.3× |
| 1.5B 8-bit | 5,858 | 3.4 min | 4.6 min | 118.6 | 88.3 (78.0) | 9.1× |
| 1.5B bf16 | 5,856 | 5.2 min | 5.7 min | 76.8 | 70.1 (61.0) | 11.5× |
| 3B 4-bit | 6,833 | 4.3 min | 5.3 min | 92.8 | 75.2 (65.8) | 10.7× |
| 3B 8-bit | 5,262 | 6.6 min | 8.5 min | 60.7 | 47.6 (52.8) | 16.9× |
| 7B 4-bit | 5,127 | 10.5 min | 12.8 min | 38.2 | 31.4 (37.7) | 25.6× |
| 7B 8-bit | 4,929 | 13.7 min | 17.3 min | 29.3 | 23.2 (27.1) | 34.7× |

- Up to 1.5B and 3B 4-bit, decompression is 13–43% faster than the old codec, and compression is faster still.
  The best size-keeping row on this Mac is now the 1.5B 8-bit: 5,858 bytes in 4.6 min.
- The 3B 8-bit and both 7B decodes came out 10–17% *slower* than the old codec's earlier timings. Their encodes
  weren't slower, and the slice A/B above shows the new loop is +5–7% faster on these models. A full-file 7B 4-bit
  decode of old against new (`llm/ab7b.log`) was meant to settle it, but Spotlight indexing loaded the Mac during
  that run, and both came out about 5× slower. Under the same load the new codec was still faster: 7.1 against
  5.5 tok/s, +29%. Best reading: sustained load (heat, or other processes) on the long big-model runs, not a
  regression. Archives are identical either way, so the size column is unaffected.
- The tok/s needed doesn't change: **804 tok/s** to decompress this file in 30 s, at any model size.


## Measured on NVIDIA GPUs: `llm/cudacodec.py` on Modal (2026-11-22)

The same coder (Witten-Neal-Cleary, frequency total 2^28) driven from PyTorch on rented GPUs, billed per second
(`.venv/bin/modal run llm/cudacodec.py --gpu H200 --models 0.5B,1.5B,3B,7B`). Qwen2.5-Coder base models in bf16, the
full BJJ game (55,854 bytes, 24,126 tokens). Every row decoded back to the input byte for byte. Logs:
`llm/cuda-H200.log`, `llm/cuda-H100.log`, `llm/cuda-h200.out`, `llm/cuda-h100-0.5b.out` (profile).

**The 0.5B decompresses the file in under 30 s: 27.0 s on an H200 and 29.0 s on an H100.**

| model (bf16) | archive bytes | MLX bf16 bytes | GPU | compress | decompress | decompress tok/s | still needed for 30 s |
|---|---:|---:|---|---:|---:|---:|---:|
| 0.5B | 7,183 | 7,186 | H200 | 26.3 s | **27.0 s** | **893** | done |
| 0.5B | 7,183 | 7,186 | H100 | 28.2 s | **29.0 s** | **832** | done |
| 1.5B | 5,855 | 5,856 | H200 | 46.1 s | 46.8 s | 516 | 1.6× |
| 3B | 5,276 | 5,256 (scored) | H200 | 69.0 s | 69.5 s | 347 | 2.3× |
| 7B | 4,991 | 4,928 | H200 | 113.5 s | 114.2 s | 211 | 3.8× |

- The times are the codec loop only. A cold run also pays for the container (a few seconds once the image is
  cached), loading weights (1.5 s for the 0.5B, 11 s for the 7B from Modal's volume) and capturing the CUDA graphs
  (13–17 s, mostly `torch.compile`; it could be cached). So a cold 0.5B decompress is about 50 s end to end today.
- Archive sizes differ from MLX by a few bytes (the 7B by 1.3%): the kernels differ, so the probabilities differ
  slightly. Each archive only decodes with the same code on the same kind of GPU.
- The 7B runs at 211 tok/s × 15.2 GB of weights = 3.2 TB/s, 67% of the H200's 4.8 TB/s, which matches the
  70% assumed in the bandwidth column above. The 0.5B is limited by kernel count, not bandwidth.
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
- Bigger models need more bandwidth per token. Guesses, not measured: the 1.5B needs 1.6× and should get it from a
  B200 (8 TB/s) or from 8-bit weights. The 7B needs 3.8×, which would take both a B200 and FP8 weights.
- Cost: Modal bills the H200 at $4.54/h and the H100 at $3.95/h, by the second. The final four-model H200 run took
  10 min (about $0.76). A 0.5B decompress on an H100 is about 3¢ of GPU time plus the cold start.

### B200 (2026-11-22, 12:15)

Same script, no B200-specific tuning (`llm/cuda-B200.log`):

| Model | GPU | Archive | Compress | Decompress | tok/s |
|---|---|---:|---:|---:|---:|
| 0.5B | B200 | 7,184 | 28.5 s | 29.6 s | 815 |
| 1.5B | B200 | 5,853 | 45.1 s | 46.1 s | 523 |

- The B200 buys nothing at this batch size of one: the 0.5B is slower than on the H200 (893 tok/s), and the 1.5B is level (516). Its extra bandwidth doesn't help kernels that are bound by launch latency and small reductions, so the 1.5B is still 1.5× short of 30 s.
- The archives differ from the H200's by a byte or two (7,184 vs 7,183; 5,853 vs 5,855). The same model on another GPU gives different floats, so these archives are tied to the GPU type as well as the model. A codec that depends only on the model needs integer (fixed-point) inference; see the README's fine print.
