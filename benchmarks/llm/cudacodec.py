"""The model-assisted codec on an NVIDIA GPU (via Modal): same coder as fastcodec2.py, in PyTorch with CUDA graphs.

    .venv/bin/modal run llm/cudacodec.py --gpu H200 --models 0.5B,1.5B,3B,7B [--limit 6000]

For each Qwen2.5-Coder base model (bf16): encode the BJJ file, decode it back, check byte identity, print timings.
Archives differ from the MLX ones (different kernels), but encoder and decoder replay the same captured CUDA graphs on
the same GPU, so they agree bit for bit.

How the step is built:
- Our own Qwen2 forward pass for one token: fused q/k/v and gate/up matmuls, a static KV cache, GQA done as
  "kv heads x group" queries so the stock fused attention kernel runs, attention limited to a 1,024-position bucket
  above the current position (one graph per bucket, so attention never reads the whole buffer).
- The frequency table (softmax, floor, cumsum, int64) and the decoder's search run inside the graph too.
- Encoding: every token is known, so the whole file is graph replays with no CPU round trip; the counts are coded
  at the end.
- Decoding: per token, replay the search graph, then immediately the next forward graph (it reads the found token on
  the GPU), and only then wait for the search's four numbers; the CPU's coder update overlaps the next forward pass.
"""
import time
from pathlib import Path

import modal

app = modal.App("compress13k")
image = modal.Image.debian_slim(python_version="3.12").apt_install("gcc", "g++").pip_install(
    "torch==2.8.0", "transformers==4.56.2", "accelerate==1.10.1"
).env({"HF_HOME": "/cache"})
vol = modal.Volume.from_name("compress13k-hf", create_if_missing=True)

PREC = 32
TOP, HALF, QTR = (1 << PREC) - 1, 1 << (PREC - 1), 1 << (PREC - 2)
TOTAL = 1 << 28
BUCKET = 1024


class Encoder:
    def __init__(self):
        self.low, self.high, self.pending, self.bits = 0, TOP, 0, []

    def emit(self, b):
        self.bits.append(b)
        self.bits.extend([1 - b] * self.pending)
        self.pending = 0

    def code_counts(self, total, hi_c, lo_c):
        rng = self.high - self.low + 1
        self.high = self.low + rng * hi_c // total - 1
        self.low = self.low + rng * lo_c // total
        while True:
            if self.high < HALF:
                self.emit(0)
            elif self.low >= HALF:
                self.emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QTR and self.high < HALF + QTR:
                self.pending += 1
                self.low -= QTR
                self.high -= QTR
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1

    def finish(self):
        self.pending += 1
        self.emit(0 if self.low < QTR else 1)
        b = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, b[k : k + 8])), 2) for k in range(0, len(b), 8))


class Decoder:
    def __init__(self, data):
        self.bits, self.pos = "".join(f"{b:08b}" for b in data), 0
        self.value = 0
        for _ in range(PREC):
            self.value = 2 * self.value + self.bit()
        self.low, self.high = 0, TOP

    def bit(self):
        self.pos += 1
        return int(self.bits[self.pos - 1]) if self.pos <= len(self.bits) else 0

    def update(self, total, hi_c, lo_c):
        rng = self.high - self.low + 1
        self.high = self.low + rng * hi_c // total - 1
        self.low = self.low + rng * lo_c // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QTR and self.high < HALF + QTR:
                self.value -= QTR
                self.low -= QTR
                self.high -= QTR
            else:
                break
            self.low, self.high, self.value = 2 * self.low, 2 * self.high + 1, 2 * self.value + self.bit()


def encode_text(hf, text):
    norm = hf.backend_tokenizer.normalizer
    for n in (norm, None):
        hf.backend_tokenizer.normalizer = n
        ids = hf.encode(text, add_special_tokens=False)
        if hf.decode(ids, clean_up_tokenization_spaces=False) == text:
            return ids
    raise ValueError("tokenizer round-trip mismatch")


_KERNELS = {}


def kernels():
    """Flash-decoding attention for one query token: one program per (kv head, 256-position chunk), then a merge."""
    if _KERNELS:
        return _KERNELS["chunk"], _KERNELS["merge"]
    import triton
    import triton.language as tl

    @triton.jit
    def chunk(q_ptr, k_ptr, v_ptr, pos_ptr, o_ptr, m_ptr, l_ptr, scale,
              NKV: tl.constexpr, GRP: tl.constexpr, CH: tl.constexpr, HD: tl.constexpr, MB: tl.constexpr):
        h, c, S = tl.program_id(0), tl.program_id(1), tl.num_programs(1)
        pos = tl.load(pos_ptr)
        rows, cols, js = tl.arange(0, MB), tl.arange(0, HD), tl.arange(0, CH)
        q = tl.load(q_ptr + (h * GRP + rows[:, None]) * HD + cols[None, :], mask=rows[:, None] < GRP, other=0.0)
        valid = c * CH + js <= pos
        base = (c * NKV + h) * CH * HD
        k = tl.load(k_ptr + base + js[:, None] * HD + cols[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale  # [MB, CH], f32
        s = tl.where(valid[None, :], s, float("-inf"))
        m = tl.max(s, 1)
        p = tl.exp(s - tl.where(m == float("-inf"), 0.0, m)[:, None])
        l = tl.sum(p, 1)
        v = tl.load(v_ptr + base + js[:, None] * HD + cols[None, :], mask=valid[:, None], other=0.0)
        o = tl.dot(p.to(tl.bfloat16), v)  # [MB, HD], f32
        out = (h * S + c) * MB
        tl.store(o_ptr + (out + rows[:, None]) * HD + cols[None, :], o)
        tl.store(m_ptr + out + rows, m)
        tl.store(l_ptr + out + rows, l)

    @triton.jit
    def merge(o_ptr, m_ptr, l_ptr, out_ptr, S, GRP: tl.constexpr, HD: tl.constexpr, MB: tl.constexpr, SB: tl.constexpr):
        h, r = tl.program_id(0), tl.program_id(1)
        cs, cols = tl.arange(0, SB), tl.arange(0, HD)
        ok = cs < S
        b = (h * S + cs) * MB + r
        m = tl.load(m_ptr + b, mask=ok, other=float("-inf"))
        l = tl.load(l_ptr + b, mask=ok, other=0.0)
        o = tl.load(o_ptr + b[:, None] * HD + cols[None, :], mask=ok[:, None], other=0.0)
        w = tl.exp(m - tl.max(m, 0))  # chunk 0 always has a valid position, so the max is finite
        res = tl.sum(o * w[:, None], 0) / tl.sum(l * w, 0)
        tl.store(out_ptr + (h * GRP + r) * HD + cols, res.to(tl.bfloat16))

    @triton.jit
    def store(kv_ptr, cache_ptr, pos_ptr, KSTRIDE, NKV: tl.constexpr, CH: tl.constexpr, HD: tl.constexpr):
        """Write row r of kv [2 nkv, hd] (k heads, then v heads) into the cache at pos."""
        r = tl.program_id(0)
        pos = tl.load(pos_ptr)
        cols = tl.arange(0, HD)
        dst = (r // NKV) * KSTRIDE + (((pos // CH) * NKV + r % NKV) * CH + pos % CH) * HD
        tl.store(cache_ptr + dst + cols, tl.load(kv_ptr + r * HD + cols))

    _KERNELS.update(chunk=chunk, merge=merge, store=store)
    return chunk, merge


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def codec(repo, text, limit, prof=False):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev, bf = "cuda", torch.bfloat16
    t0 = time.time()
    hf = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=bf, device_map=dev)
    vol.commit()
    t_load = time.time() - t0
    start = hf.bos_token_id if hf.bos_token_id is not None else hf.eos_token_id
    if limit:
        text = text[:limit]
    ids = encode_text(hf, text)
    n = len(ids)
    log(f"{repo}: loaded in {t_load:.1f}s, {len(text.encode())} bytes, {n} tokens")

    cfg, inner = m.config, m.model
    nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd, grp, eps = cfg.hidden_size // nq, nq // nkv, cfg.rms_norm_eps
    layers = []
    with torch.no_grad():
        for l in inner.layers:
            a = l.self_attn
            layers.append((
                l.input_layernorm.weight,
                torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]),
                torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]),
                a.o_proj.weight,
                l.post_attention_layernorm.weight,
                torch.cat([l.mlp.gate_proj.weight, l.mlp.up_proj.weight]),
                l.mlp.down_proj.weight,
            ))
    emb, fnorm, head = inner.embed_tokens.weight, inner.norm.weight, m.lm_head.weight
    V = head.shape[0]
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    nb = (n + BUCKET) // BUCKET  # positions 0..n-1 are fed (start token + first n-1 tokens)
    maxlen = nb * BUCKET
    CH = 256  # attention chunk (flash-decoding): the cache is [layer, chunk, kv head, CH, hd]
    KV = torch.zeros(len(layers), 2, maxlen // CH, nkv, CH, hd, dtype=bf, device=dev)  # [layer, k/v, chunk, head, CH, hd]

    tok = torch.zeros(1, dtype=torch.long, device=dev)
    pos = torch.zeros(1, dtype=torch.long, device=dev)
    cum = torch.zeros(V, dtype=torch.long, device=dev)
    st = torch.zeros(3, dtype=torch.long, device=dev)
    res = torch.zeros(4, dtype=torch.long, device=dev)
    ids_gpu = torch.tensor(ids + [0], dtype=torch.long, device=dev)
    counts = torch.zeros(n + 1, 3, dtype=torch.long, device=dev)
    scale = hd ** -0.5

    def rms(x, w):
        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(bf)

    def rope(x, cos, sin):
        x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    # Fused pieces (torch.compile, no CUDA graphs of its own: the whole step is captured below).
    @torch.compile(dynamic=False)
    def head_in(tok, pos):
        ang = pos.float()[:, None] * inv
        cos = torch.cat([ang.cos(), ang.cos()], -1).to(bf)[:, None]  # [1, 1, hd]
        sin = torch.cat([ang.sin(), ang.sin()], -1).to(bf)[:, None]
        return emb[tok], cos, sin

    @torch.compile(dynamic=False)
    def pre(h, cos, sin, n1, wqkv, bqkv):
        q, k, v = F.linear(rms(h, n1), wqkv, bqkv).split([nq * hd, nkv * hd, nkv * hd], -1)
        q = rope(q.view(1, nq, hd), cos, sin)
        k = rope(k.view(1, nkv, hd), cos, sin)
        return q.view(nkv, grp, hd), torch.cat([k.view(nkv, hd), v.view(nkv, hd)])  # kv: [2 nkv, hd]

    k_chunk, k_merge = kernels()
    k_store = _KERNELS["store"]
    MB, maxS = 16, maxlen // CH
    SB = 1 << (maxS - 1).bit_length()
    o_part = torch.zeros(nkv, maxS, MB, hd, dtype=torch.float32, device=dev)
    m_part = torch.zeros(nkv, maxS, MB, dtype=torch.float32, device=dev)
    l_part = torch.zeros_like(m_part)
    att = torch.zeros(nkv, grp, hd, dtype=bf, device=dev)

    def attn(q, i, S):
        k_chunk[(nkv, S)](q.contiguous(), KV[i, 0], KV[i, 1], pos, o_part, m_part, l_part, scale, NKV=nkv, GRP=grp, CH=CH, HD=hd, MB=MB)
        k_merge[(nkv, grp)](o_part, m_part, l_part, att, S, GRP=grp, HD=hd, MB=MB, SB=SB)
        return att

    @torch.compile(dynamic=False)
    def post(h, o, wo, n2, wgu, wd):
        h = h + F.linear(o.reshape(1, nq * hd), wo)
        g, u = F.linear(rms(h, n2), wgu).chunk(2, -1)
        return h + F.linear(F.silu(g) * u, wd)

    @torch.compile(dynamic=False)
    def table(h):
        p = torch.softmax(F.linear(rms(h, fnorm), head)[0].float(), -1)
        return torch.cumsum(torch.floor(p * (TOTAL - V)).long() + 1, 0)

    def forward(Lb):
        """Feed tok at pos, write the frequency table for the next token into cum, advance pos."""
        h, cos, sin = head_in(tok, pos)
        for i, (n1, wqkv, bqkv, wo, n2, wgu, wd) in enumerate(layers):
            q, kv = pre(h, cos, sin, n1, wqkv, bqkv)
            k_store[(2 * nkv,)](kv.contiguous(), KV[i], pos, KV.stride(1), NKV=nkv, CH=CH, HD=hd)
            h = post(h, attn(q, i, Lb // CH), wo, n2, wgu, wd)
        cum.copy_(table(h))
        pos.add_(1)

    def gather3(t):
        """[total, cum[t], cum[t-1] or 0] for a 1-element long tensor t."""
        hi = cum.gather(0, t)
        lo = torch.where(t > 0, cum.gather(0, (t - 1).clamp(min=0)), torch.zeros_like(t))
        return torch.cat([cum[-1:], hi, lo])

    def enc_step(Lb):
        k = pos.clone()
        forward(Lb)
        t = ids_gpu.gather(0, k)
        counts.index_copy_(0, k, gather3(t).view(1, 3))
        tok.copy_(t)

    def search():
        total = cum[-1]
        target = ((st[0] - st[1] + 1) * total - 1) // st[2]
        t = (cum <= target).sum().view(1)
        g = gather3(t)
        res.copy_(torch.cat([t, g[1:], g[:1]]))
        tok.copy_(t)

    def reset():
        KV.zero_(); pos.zero_(); tok.fill_(start)
        st.copy_(torch.tensor([0, 0, TOP + 1]))  # a valid coder state for the search graph's warmup

    t0 = time.time()
    pool = torch.cuda.graph_pool_handle()

    def capture(fn, *a):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                reset(); fn(*a)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool), torch.no_grad():
            fn(*a)
        return g

    fwd = [capture(forward, (b + 1) * BUCKET) for b in range(nb)]
    encg = [capture(enc_step, (b + 1) * BUCKET) for b in range(nb)]
    srch = capture(search)
    torch.cuda.synchronize()
    t_cap = time.time() - t0
    log(f"captured {2 * nb + 1} graphs in {t_cap:.1f}s")
    if prof:
        from torch.profiler import profile, ProfilerActivity
        for b in (0, nb // 2, nb - 1):  # forward-graph time per step at short, middle and full context
            reset(); pos.fill_(b * BUCKET); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(200):
                fwd[b].replay()
            e1.record(); torch.cuda.synchronize()
            log(f"forward graph at context {(b + 1) * BUCKET}: {e0.elapsed_time(e1) / 200:.3f} ms/step")
        reset(); pos.fill_((nb - 1) * BUCKET)
        with torch.no_grad():
            for _ in range(3):
                forward(nb * BUCKET)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as pr:
                for _ in range(10):
                    forward(nb * BUCKET)
                torch.cuda.synchronize()
        print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=70), flush=True)

    # Encode: all graph replays, then code the counts on the CPU.
    reset(); torch.cuda.synchronize()
    t0 = time.time()
    for k in range(n):
        encg[k // BUCKET].replay()
    cnt = counts[:n].tolist()
    t_gpu = time.time() - t0
    bad = [k for k, (tot, hi, lo) in enumerate(cnt) if not 0 <= lo < hi <= tot]
    if bad:
        raise ValueError(f"bad counts at token {bad[0]}: {cnt[bad[0]]} ({len(bad)} bad in all)")
    enc = Encoder()
    for tot, hi, lo in cnt:
        enc.code_counts(tot, hi, lo)
    body = enc.finish()
    t_enc = time.time() - t0
    log(f"encode: {n} tokens -> {4 + len(body)} bytes, {t_enc:.2f}s ({n / t_enc:.0f} tok/s; GPU part {t_gpu:.2f}s, "
        f"{n / t_gpu:.0f} tok/s)")

    # Decode.
    reset(); torch.cuda.synchronize()
    t0 = time.time()
    dec, out = Decoder(body), []
    st_h = torch.zeros(3, dtype=torch.long).pin_memory()
    res_h = torch.zeros(4, dtype=torch.long).pin_memory()
    ev = torch.cuda.Event()
    fwd[0].replay()
    for k in range(n):
        st_h[0], st_h[1], st_h[2] = dec.value, dec.low, dec.high - dec.low + 1
        st.copy_(st_h, non_blocking=True)
        srch.replay()
        res_h.copy_(res, non_blocking=True)
        ev.record()
        if k + 1 < n:
            fwd[(k + 1) // BUCKET].replay()
        ev.synchronize()
        t, hi, lo, tot = res_h.tolist()
        dec.update(tot, hi, lo)
        out.append(t)
    torch.cuda.synchronize()
    t_dec = time.time() - t0
    back = hf.decode(out, clean_up_tokenization_spaces=False)
    ok = back == text
    log(f"decode: {t_dec:.2f}s ({n / t_dec:.0f} tok/s), round trip {'OK' if ok else 'FAILED'}")
    return dict(repo=repo, bytes_in=len(text.encode()), tokens=n, archive=4 + len(body), load=t_load, capture=t_cap,
                encode=t_enc, encode_gpu=t_gpu, decode=t_dec, ok=ok)


def body(models, text, limit, prof=False):
    import os, traceback
    if os.environ.get("CC_DEBUG"):  # synchronous launches, so a CUDA error points at its own line
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    import torch
    log(f"GPU: {torch.cuda.get_device_name()}, CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    try:
        return [codec(f"Qwen/Qwen2.5-Coder-{s}", text, limit, prof) for s in models]
    except Exception:
        raise RuntimeError(traceback.format_exc()) from None


kw = dict(image=image, volumes={"/cache": vol}, timeout=3600)


@app.function(gpu="L4", **kw)
def run_l4(models, text, limit, prof=False):
    return body(models, text, limit, prof)


@app.function(gpu="H100", **kw)
def run_h100(models, text, limit, prof=False):
    return body(models, text, limit, prof)


@app.function(gpu="H200", **kw)
def run_h200(models, text, limit, prof=False):
    return body(models, text, limit, prof)


@app.function(gpu="B200", **kw)
def run_b200(models, text, limit, prof=False):
    return body(models, text, limit, prof)


@app.local_entrypoint()
def main(gpu: str = "L4", models: str = "0.5B", limit: int = 0, prof: bool = False, file: str = "bjj-case/3-fully-minified.html"):
    fn = {"L4": run_l4, "H100": run_h100, "H200": run_h200, "B200": run_b200}[gpu]
    text = Path(file).read_bytes().decode("utf-8")
    t0 = time.time()
    rows = fn.remote(models.split(","), text, limit, prof)
    wall = time.time() - t0
    lines = [f"{gpu}, {time.strftime('%Y-%m-%d %H:%M')}, wall {wall:.0f}s including container start"]
    for r in rows:
        lines.append(
            f"{r['repo']}: {r['bytes_in']} -> {r['archive']} bytes, {r['tokens']} tokens | load {r['load']:.1f}s, "
            f"capture {r['capture']:.1f}s | encode {r['encode']:.2f}s ({r['tokens'] / r['encode']:.0f} tok/s) | "
            f"decode {r['decode']:.2f}s ({r['tokens'] / r['decode']:.0f} tok/s) | round trip {'OK' if r['ok'] else 'FAILED'}")
    print("\n".join(lines))
    with open(f"llm/cuda-{gpu}.log", "a") as f:
        f.write("\n".join(lines) + "\n\n")
