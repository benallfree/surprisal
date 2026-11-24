# Surprisal integer codec, spec version 1

An LLM-driven arithmetic coder whose archives depend only on the model, not on the hardware. Every step of the
model's forward pass is integer arithmetic whose result does not depend on the order of operations, so any correct
implementation on any machine (CPU, GPU, any language) computes the same probability tables, bit for bit, and decodes
any archive made by any other.

The spec has two parts.

- **The integer model** (§1.2–§4): conversion, the model file, the arithmetic, the forward pass and the frequency
  table. Implemented by nollama's `engine` package (pure Go, CPU, NEON kernels: `LoadIntModel`, `NewSession`/`Feed`,
  `ConvertIntModel`), which knows nothing about compression.
- **The codec** (§1.1, §5–§7): the model manifest, the coder, the archive and file-set formats, and tokenization.
  Implemented by **surprisal** (`surprisal/`, `cmd/surprisal/`), which calls nollama for the model.

Reference implementation of both: `reference/python/intref.py` (numpy, slow, authoritative). Also `reference/python/intgpu.py` (PyTorch, plain
and Triton/CUDA). Conformance: `spec/vectors/`.

Compressed archives use the extension `.13k`; converted models are GGUF files.

Contents: [1 Models](#1-models) · [2 Arithmetic](#2-arithmetic) · [3 Forward pass](#3-forward-pass) ·
[4 Frequency table](#4-frequency-table) · [5 Coder](#5-coder) · [6 Archive](#6-archive) · [7 Text](#7-text) ·
[8 Conformance](#8-conformance) · [9 Implementation notes](#9-implementation-notes)

## 1. Models

### 1.1 Levels and the manifest

`spec/models-v1.json` lists the models and the compression levels. An archive records its header version and level
(§6); the manifest of that version maps the level to the exact model, so an archive always names the model that
decodes it.

| level | id | source (official, pinned revision) | converted model | BJJ archive |
|---|---:|---|---|---:|
| 0 | – | none: the bytes are stored uncoded | – | 55,856 |
| 1–3 | 1 | `Qwen/Qwen2.5-Coder-0.5B` @ `8123ea2e` | `qwen2.5-coder-0.5b-int-v1.gguf`, 520 MB | 7,186 |
| 4–6 | 2 | `Qwen/Qwen2.5-Coder-1.5B` @ `df3ce67c` | `qwen2.5-coder-1.5b-int-v1.gguf`, 1.6 GB | 5,856 |
| 7–9 | 3 | `Qwen/Qwen2.5-Coder-7B` @ `0396a761` | `qwen2.5-coder-7b-int-v1.gguf`, 7.9 GB | 4,970 |

A version's level table never changes. A new mapping (another model at some level, say) is a new header version
with its own manifest, and decoders keep every earlier table. A decoder refuses an archive whose version it doesn't
know, rather than guess.

The converted model is a modified copy of the official one: the same weights, rounded to integers (§1.2). Each
entry has its SHA-256, its size and download URLs. Ids are permanent: an id is never reused or deleted,
so old archives keep decoding. A new or updated model gets a new id.

### 1.2 Conversion (official bf16 weights → converted model)

Run once per model. Anyone can repeat it and must get the manifest's SHA-256: the Mac (arm64) and Linux x86 builds
are byte-identical.

- **Read** the bf16 safetensors. bf16 → float32 → float64 is exact.
- **Matrices** (`w` of shape [out, in], in a multiple of 64): for each row and each group of 64 inputs,
  `sr = max|w| / 127` (`sr = 1e-30` if 0). Then:
  - `E = exponent(max sr) − 1 − 14`, where `exponent(x)` is frexp's exponent, i.e. x = f·2^exponent with
    0.5 ≤ f < 1. This is `floor(log2 max sr) − 14`, taken from the float's bits, with no libm call.
  - `s = clip(rint(sr / 2^E), 1, 32767)` as int16.
  - `q = clip(rint(w / (s·2^E)), −127, 127)` as int8.

  `rint` rounds half to even. All of this is IEEE division and power-of-two scaling, so it's identical everywhere.
  Then w ≈ q · s · 2^E.
- **Fused matrices:**
  - `qkv` = rows of q_proj, then k_proj, then v_proj;
  - `gu` = rows of gate_proj, then up_proj;
  - `o` = o_proj; `d` = down_proj; `emb` = embed_tokens;
  - `head` = lm_head, present only when `tie_word_embeddings` is false. Otherwise the head is `emb`.
- **Biases:** `bqkv = rint(b · 2^24)` (int64), with q, k, v concatenated.
- **Norm weights:** `g = rint(g · 2^16)` (int64), for `g1` (input_layernorm), `g2` (post_attention_layernorm) and
  `gf` (final norm).
- **Tables** are copied from `spec/tables-v1.gguf` (computed once and frozen, so no platform's libm is
  involved):
  - `EXPA[i] = round(2^15 e^{−i/1024})`, i < 16,384;
  - `EXPF[i] = round(2^30 e^{−i/1024})`, i < 24,576;
  - `SIG[i] = round(2^16 / (1 + e^{−(i−16384)/1024}))`, i ≤ 32,768;
  - `ROPE_C/S_hd[pos, j] = round(cos/sin(pos · 10^6^{−2j/hd}) · 2^14)` for pos < 32,768 and hd ∈ {64, 128}.

  Go's math library recomputes all seven tables byte-identically on arm64. nollama's converter therefore generates
  them and checks each against its SHA-256: EXPA `6a5c35d8…`, EXPF `f1bc8936…`, SIG `bf265d6d…`,
  ROPE_C/S_64 `d8095cb9…`/`f2f65aef…`, ROPE_C/S_128 `b23e31ed…`/`6f1899b8…`.
- **Requirements** (Qwen2.5): rope_theta = 10^6, 32,768 positions, rms_norm_eps = 10^−6, and every `in` dimension a
  multiple of 128.

### 1.3 Converted model file (GGUF)

A GGUF v3 file, written deterministically (everything is little-endian; tensor data aligned to 32 bytes).

**Metadata, in this order:**
1. `general.architecture` = "qwen2-intcodec", then `general.alignment` = 32 (uint32), then
   `intcodec.spec_version` = 1 (uint32).
2. The config: `intcodec.<key>` for bos_token_id, hidden_size, intermediate_size, max_position_embeddings,
   num_attention_heads, num_hidden_layers, num_key_value_heads, rms_norm_eps, rope_theta, tie_word_embeddings and
   vocab_size. Integers are uint32, rms_norm_eps and rope_theta float64, tie_word_embeddings bool.
3. The exponents: `intcodec.exp.<matrix>` (int32), in tensor order.
4. The tokenizer, in the standard GGUF form:
   - `tokenizer.ggml.model` = "gpt2", `tokenizer.ggml.pre` = "qwen2";
   - `tokenizer.ggml.tokens`: padded to vocab_size with `[PAD<i>]`;
   - `tokenizer.ggml.token_type`: 1 normal, 3 control, 4 user-defined, 5 padding;
   - `tokenizer.ggml.merges`.

   These are taken from the source revision's tokenizer.json.

**Tensors**, GGUF integer types (I8 = 24, I16 = 25, I32 = 26, I64 = 27), dims in GGUF order (dims[0] contiguous):
1. `emb.q` (I8), `emb.s` (I16), then `head.q`, `head.s` if the head is untied.
2. `gf` (I64).
3. Per layer l: `l.qkv.q, l.qkv.s, l.bqkv, l.o.q, l.o.s, l.gu.q, l.gu.s, l.d.q, l.d.s, l.g1, l.g2`.
4. `ROPE_C, ROPE_S` (I16), `EXPA, EXPF, SIG` (I32).

Nothing in the file depends on where the weights came from: building from the official repo and from a faithful copy
gives the same bytes.

## 2. Arithmetic

Every integer quantity carries an exponent e: value = int · 2^e. All arithmetic is on mathematical integers. The
bounds below keep every intermediate within int64.

- `rshift(x, n)` = `(x + 2^(n−1)) >> n` for n > 0 (arithmetic shift, i.e. floor: round half up), and `x << −n`
  for n ≤ 0. Moving a value from exponent f to exponent t is `rshift(x, t − f)`.
- `tdiv(a, b)` = division truncating toward zero (b > 0).
- `bitlen(v)` = the number of bits of v ≥ 0 (bitlen(0) = 0).
- `blockf(x, e)`: sh = max(0, bitlen(max|x|) − 15); m = clip(rshift(x, sh), −32767, 32767); exponent e + sh. A
  vector of 16-bit mantissas with one shared exponent.
- `matvec(W, (m, e))`: `acc[j] = Σ_g s[j,g] · Σ_{k∈g} q[j,k]·m[k]` (|acc| < 2^52), at exponent `E + e`.

## 3. Forward pass

Feed one token at position p and get the frequency table for the next token. D = hidden size, I = intermediate size,
nq/nkv = query/kv heads, hd = D/nq, grp = nq/nkv. Positions restart at 0 in every segment (§6), where the first token
fed is the start token `bos_token_id` (151643).

1. **Embedding:** h = rshift(q[tok] · s[tok, ⌊k/64⌋], −24 − E) (int64, e = −24).
2. **Each layer:**
   1. **rmsnorm(h, g):**
      - sh = max(0, bitlen(max|h|) − 20); x = rshift(h, sh);
      - ms = Σx² // D + (2^max(0, 48−2·sh)) // 10^6; r = max(1, isqrt(ms)) (exact integer square root);
      - y = tdiv(x · g · 2^14, r) at e = −30; return blockf(y, −30).
   2. **qkv:** x = rshift(matvec(Wqkv, ·), −24 − (E+e)) + bqkv.
      - q = rows [0, nq·hd); k = the next nkv·hd rows; v = the last nkv·hd rows.
      - Each head is hd values.
   3. **rope** on each q and k head with c = ROPE_C[p], s = ROPE_S[p]. With x1, x2 = the first and second halves of
      the head: (rshift(x1·c − x2·s, 14), rshift(x2·c + x1·s, 14)).
   4. **Cache:** K[p] = clip(rshift(k, 12), ±(2^22−1)), V[p] = clip(rshift(v, 12), ±(2^22−1)), both at e = −12.
   5. **Attention**, for each q head i, using kv head ⌊i / grp⌋ and keys t = 0..p:
      - SC = round(2^16 / √hd), which is 8192 for hd 64 and 5793 for hd 128.
      - (mq, sh) = blockf(q_i · SC); score_t = rshift(mq · K_t, 42 − sh) (e = −10).
      - M = max_t score_t; w_t = EXPA[min(M − score_t, 16383)]; Z = Σ w_t; O = Σ w_t · V_t.
      - o_i = sign(O) · ⌊|O| · 2^12 / Z⌋ (e = −24), computed as hi = |O| // Z, rem = |O| − hi·Z,
        o = (hi << 12) + (rem << 12) // Z.
   6. **Output projection:** h += rshift(matvec(Wo, blockf(o, −24)), −24 − (E+e)), with o = the heads concatenated.
   7. **MLP:** (m, e) = rmsnorm(h, g2); gu = rshift(matvec(Wgu, (m, e)), −16 − (E+e)); g = gu[:I], u = gu[I:].
      - a = rshift(g · SIG[clip(rshift(g, 6), −16384, 16384) + 16384], 16); pv = rshift(a · u, 16).
      - h += rshift(matvec(Wd, blockf(pv, −16)), −24 − (E+e)).
3. **Logits:** (m, e) = rmsnorm(h, gf); l = rshift(matvec(Whead, (m, e)), −10 − (E+e)).

## 4. Frequency table

M = max l; f_i = EXPF[min(M − l_i, 24575)]; Z = Σ f_i; freq_i = f_i · (2^28 − V) // Z + 1 (V = vocab size);
cum = the inclusive prefix sums of freq. Token t is coded with total = cum[V−1], hi = cum[t], lo = cum[t−1]
(0 for t = 0). Every token has freq ≥ 1, and total ≤ 2^28.

## 5. Coder

Witten-Neal-Cleary, 32-bit: TOP = 2^32 − 1, HALF = 2^31, QTR = 2^30.

- **Encoder:** start with low = 0, high = TOP, pending = 0. Per token:
  - rng = high − low + 1; high = low + rng·hi // total − 1; low = low + rng·lo // total.
  - Then loop:
    - if high < HALF: emit 0;
    - else if low ≥ HALF: emit 1, low −= HALF, high −= HALF;
    - else if low ≥ QTR and high < HALF + QTR: pending += 1, low −= QTR, high −= QTR;
    - else stop.
  - After each case except stop: low = 2·low, high = 2·high + 1.
  - emit(b) writes b, then `pending` copies of 1−b, and resets pending.
- **Finish:** pending += 1; emit(0 if low < QTR else 1). Pad to whole bytes with 0 bits, most significant bit first.
- **Decoder:** value = the first 32 bits (bits past the end read as 0); low = 0, high = TOP. Per token:
  - target = ((value − low + 1)·total − 1) // (high − low + 1); t = the number of i with cum[i] ≤ target.
  - Apply the encoder's update with that t's counts, then the same renormalization loop, also shifting
    value = 2·value + next bit (and subtracting HALF or QTR from value along with low and high).

## 6. Archive

```
byte 0   kind          0xA1: a stream (the decoded bytes are one unnamed blob)
                       0xA2: a file set (the decoded bytes are a container of named files, below)
byte 1   version/level high nibble: the header version (1, the levels in spec/models-v1.json, §1.1)
                       low nibble: the compression level, 0-9
then, levels 1-9:
varint   n             token count, unsigned LEB128
rest     coder output  §5
or, level 0:
rest     the decoded bytes, uncoded
```

Decoders refuse any other version or a level above 9. Archives from before header versions (surprisal 0.0.1-alpha.1)
had the model id in byte 1, so they read as version 0 and are refused.

**File set:** the decoded bytes are, for each file, its path, `\n`, its length in bytes (decimal), `\n`, then its
bytes.
- **Paths** are relative and slash-separated, with no `..`, backslash, newline or NUL, and already clean.
  Decoders refuse anything else.
- **Order:** surprisal sorts files by path and names them relative to their common directory.
- **Paths are coded too:** they go through the model like the contents, so later files are compressed in the context
  of earlier ones.

**Long inputs:** the n tokens are coded in segments of at most 32,768 tokens: [0, 32768), [32768, 65536), and so on.
Each segment starts a fresh context (empty cache, start token at position 0). The coder runs on, unbroken, across
segments.

## 7. Text

v1 codes UTF-8 text with the Qwen2.5 tokenizer, which is the same for every model.

- **Decoder:** maps token ids back to bytes (byte-level BPE decode, no clean-up of tokenization spaces).
- **Encoder:** may use any token sequence that decodes to exactly the input bytes. The reference tries the standard
  tokenization, then the one without the tokenizer's normalizer, and refuses input that neither round-trips.

Only the decoder's model must match; the encoder's choice of tokenization is free.

## 8. Conformance

`spec/vectors/vectors.json` indexes the test vectors:

- **Inputs:** `tiny.html` (400 bytes), `slice.html` (6,000 bytes), `bjj.html` (55,854 bytes).
- **Archives:**
  - `m1-tiny.13k`: made by the numpy spec on a Mac, decoded on an H100;
  - `m1-slice.13k`: byte-identical between the numpy spec and the GPU;
  - `m1-bjj.13k`, `m2-bjj.13k`, `m3-bjj.13k`: the full file for each model, made on an H100 by the batched and
    step-by-step paths, which agree, and round-tripped;
  - `m1-bjj3.13k`: `bjj.html` three times (72,378 tokens, three segments), testing the long-input rule;
  - `m3-tiny.13k`: the 7B on `tiny.html`, made on an H100 and reproduced byte for byte by nollama in Go on a Mac.

  Each is made at the first level of its model (1, 4 or 7; vectors.json gives the level). An implementation must
  decode each to its input, and should reproduce each byte for byte when encoding.
- **Table hashes:**
  - `m1-slice-tables.txt`: 300 positions;
  - `m2-tiny-tables.txt`: 30 positions.

  Line "k h": h = the first 16 hex digits of the SHA-256 of the int64 little-endian cumulative table produced after
  feeding position k. These pinpoint the first position where an implementation diverges.

## 9. Implementation notes

What may differ between implementations, and the traps found so far:

- **Integer sums in any order or grouping** (split-K partials, chunked attention, atomics) are fine; integers have
  no rounding. Floating point is allowed only where it is exact. Examples:
  - float64 products and sums of integers whose partial sums stay below 2^53;
  - float64 `floor(x·2^−n + 1/2)` for a score x < 2^44;
  - a float64 square-root estimate followed by an exact integer correction.
- **int8 tensor cores** can compute the products exactly. Split each operand into signed or offset bytes, sum in
  int32 (below 2^31), and recombine with shifts.
- **The batched (prefill) form computes exactly the same tables as the step-by-step form.** Compression can process
  all positions of a segment as matrix-matrix work.
- **Read before reset.** A kernel that reads a value and then zeroes it in the same program must make sure every
  thread has read it first (barrier). Otherwise the result is nondeterministic.
- **The table hashes are the fastest way to find a divergence.** Diff them position by position.
