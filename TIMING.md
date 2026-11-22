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
- The full-file re-run (`llm/fast2.log`) was distorted by other GPU work on the Mac (several rows came out slower
  than the old codec while the A/B shows the new one is never slower); its byte-identity checks stand, its timings don't.
  Clean full-file decodes: 0.5B bf16 155 tok/s (was 129), 0.5B 8-bit 127 (was 116); speed falls with context
  depth (about 230 tok/s at the start of the file, about 120 by 22K tokens: attention over the growing KV cache).
- Tried and dropped: compiling the whole step (`llm/fastcodec3.py`, padded KV buckets make attention slower),
  fused q/k/v and gate/up matmuls plus compiled blocks (`llm/fastmodel.py`, no gain), vocabulary pruning to the
  corpus's 42,458 tokens (+11–18% speed but +1.1% size, `llm/prune_cost.py`: BJJ's escapes are English words).
- **Diminishing returns reached in Python/MLX.** What's left is inside MLX's GPU kernels: about 4 ms per token for
  the 0.5B at short context where reading its weights takes about 1.5 ms. Closing that needs fused custom kernels
  (one kernel per layer), which is engine work (nollama's Metal path), not codec work.
