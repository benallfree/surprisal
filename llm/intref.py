"""Integer-only Qwen2.5 inference: the reference (the spec) for a codec whose archives depend on the model alone.

    .venv/bin/python llm/intref.py build  <hf-model-dir> <bundle.npz>        # quantize once; the bundle IS the model
    .venv/bin/python llm/intref.py score  <hf-model-dir> <bundle.npz> <text> [variants]   # ideal sizes: float,f8,int
    .venv/bin/python llm/intref.py encode <hf-model-dir> <bundle.npz> <text> <archive>
    .venv/bin/python llm/intref.py decode <hf-model-dir> <bundle.npz> <archive> <text>

Why: floating-point sums depend on the order of additions, and every GPU, library and kernel orders them differently,
so float probabilities differ in the last bits from machine to machine and an arithmetic-coded archive only decodes on
the hardware that made it. Here every step is integer arithmetic whose result does not depend on order (sums of
integers, max, shifts, integer division, table lookups), so any implementation that follows these rules on any hardware
computes the same frequency table, bit for bit.

The bundle (built once, from the bf16 weights, with float math; after that it is fixed data):
- Every matrix: int8 q[out, in] with one integer scale s[out, in/64] (1..32767) per 64 inputs and one exponent E:
  w = q * s * 2^E. Embedding and output head share one matrix (tied).
- Biases: int64, value * 2^24. Norm weights: int64, value * 2^16.
- Tables (stored, not recomputed, so no libm differences): ROPE_C/S[pos, 32] = round(cos/sin * 2^14);
  EXPA[i] = round(2^15 exp(-i/1024)) (attention), EXPF[i] = round(2^30 exp(-i/1024)) (output), SIG[i] =
  round(2^16 sigmoid((i - 16384)/1024)).

Number formats: every integer vector has an exponent e, value = int * 2^e. The residual stream h is e = -24 (int64).
rshift(x, n) = (x + 2^(n-1)) >> n for n > 0 (arithmetic shift: round half up), x << -n for n <= 0.
tdiv = division truncating toward zero. blockf(x): sh = max(0, bitlen(max|x|) - 15), m = clip(rshift(x, sh), +-32767),
exponent += sh (a 16-bit mantissa vector with one shared exponent).

Ops (D = hidden, per token):
- matvec(W, (m, e)): acc[j] = sum_g s[j,g] * sum_{k in g} q[j,k] m[k]   (exact int64; < 2^50), exponent E + e.
- rmsnorm(h, g): sh = max(0, bitlen(max|h|) - 20); x = rshift(h, sh); ms = sum(x^2) // D + (2^(48-2sh)) // 10^6;
  r = max(1, isqrt(ms)); y = tdiv(x * g * 2^14, r) (e = -30); return blockf(y).
- layer: (m,e) = rmsnorm(h, g1); qkv = rshift(matvec, to e=-24) + b; rope on q, k:
  (x1*c - x2*s, x2*c + x1*s) >> 14 (rshift); cache K, V = clip(rshift(k or v, 12), +-(2^22-1)) (e = -12; keeps sums below 2^53 and lets a fast
  implementation split K, V into three signed bytes).
  attention per q head: (mq, eq) = blockf(q_head); score_t = rshift((mq . K_t) * SC, 42 - sh) (e = -10), where
  SC = round(2^16 / sqrt(head_dim)) (2^13 for head_dim 64) and sh is the q head's blockf shift;
  M = max score; w_t = EXPA[min(M - score_t, len-1)]; Z = sum w; O = sum w_t V_t;
  o = sign(O) * (|O| * 2^12 // Z)  (computed as hi/remainder so nothing overflows; e = -24).
  h += rshift(matvec(Wo, blockf(o)), to -24).
  (m,e) = rmsnorm(h, g2); g, u = rshift(matvec(Wgu), to -16); a = rshift(g * SIG[clip(rshift(g, 6), +-16384) + 16384], 16);
  p = rshift(a * u, 16); h += rshift(matvec(Wd, blockf(p)), to -24).
- start token: config bos_token_id (151643 for Qwen2.5), fed at position 0.
- output: (m,e) = rmsnorm(h, gf); l = rshift(matvec(Wemb), to e=-10); M = max l; f_i = EXPF[min(M - l_i, len-1)];
  Z = sum f; freq_i = f_i * (2^28 - V) // Z + 1; the coder uses cumsum(freq) exactly like segcodec.py.
- embedding: h = rshift(q[tok] * s[tok], to -24 from E).

This numpy version is slow and only meant to be the spec: matrix products run as float64 BLAS calls on exact integers,
which is exact (every partial sum is an integer below 2^53, checked), so it equals the integer definition.
"""
import json, math, struct, sys, time
from pathlib import Path
import numpy as np

G = 64
TOTAL = 1 << 28
NA, NF, NS = 16 * 1024, 24 * 1024, 16 * 1024  # table lengths: exp for attention, exp for output, sigmoid half-range
I64 = np.int64


# ---------------------------------------------------------------- loading and building
def load_st(d):
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        b = f.read_bytes()
        n = struct.unpack("<Q", b[:8])[0]
        hdr = json.loads(b[8 : 8 + n])
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            assert v["dtype"] == "BF16", v["dtype"]
            raw = np.frombuffer(b, dtype=np.uint16, count=(e - s) // 2, offset=8 + n + s)
            out[k] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])
    return out


def quant(w):
    """float [out, in] -> int8 q, int16 s [out, in/64], E."""
    w = w.astype(np.float64)
    o, i = w.shape
    wg = w.reshape(o, i // G, G)
    sr = np.abs(wg).max(axis=2) / 127
    sr[sr == 0] = 1e-30
    E = int(math.floor(math.log2(sr.max()))) - 14
    s = np.clip(np.rint(sr / 2.0**E), 1, 32767)
    q = np.clip(np.rint(wg / (s[:, :, None] * 2.0**E)), -127, 127)
    return q.reshape(o, i).astype(np.int8), s.astype(np.int16), E


def build(src, dst):
    cfg = json.loads((Path(src) / "config.json").read_text())
    W = load_st(src)
    L, I = cfg["num_hidden_layers"], cfg["intermediate_size"]
    out = {"cfg": np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)}

    def put(name, w):
        q, s, E = quant(w)
        out[name + ".q"], out[name + ".s"], out[name + ".E"] = q, s, np.array(E)

    put("emb", W["model.embed_tokens.weight"])
    out["gf"] = np.rint(W["model.norm.weight"].astype(np.float64) * 2**16).astype(I64)
    for l in range(L):
        p = f"model.layers.{l}."
        a = p + "self_attn."
        put(f"{l}.qkv", np.concatenate([W[a + x + "_proj.weight"] for x in "qkv"]))
        out[f"{l}.bqkv"] = np.rint(np.concatenate([W[a + x + "_proj.bias"] for x in "qkv"]).astype(np.float64) * 2**24).astype(I64)
        put(f"{l}.o", W[a + "o_proj.weight"])
        put(f"{l}.gu", np.concatenate([W[p + "mlp.gate_proj.weight"], W[p + "mlp.up_proj.weight"]]))
        put(f"{l}.d", W[p + "mlp.down_proj.weight"])
        out[f"{l}.g1"] = np.rint(W[p + "input_layernorm.weight"].astype(np.float64) * 2**16).astype(I64)
        out[f"{l}.g2"] = np.rint(W[p + "post_attention_layernorm.weight"].astype(np.float64) * 2**16).astype(I64)
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    inv = cfg["rope_theta"] ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.arange(cfg["max_position_embeddings"], dtype=np.float64)[:, None] * inv[None]
    out["ROPE_C"] = np.rint(np.cos(ang) * 2**14).astype(np.int16)
    out["ROPE_S"] = np.rint(np.sin(ang) * 2**14).astype(np.int16)
    i = np.arange(max(NA, NF), dtype=np.float64)
    out["EXPA"] = np.rint(2.0**15 * np.exp(-i[:NA] / 1024)).astype(I64)
    out["EXPF"] = np.rint(2.0**30 * np.exp(-i[:NF] / 1024)).astype(I64)
    x = (np.arange(2 * NS + 1, dtype=np.float64) - NS) / 1024
    out["SIG"] = np.rint(2.0**16 / (1 + np.exp(-x))).astype(I64)
    np.savez(dst, **out)


# ---------------------------------------------------------------- integer helpers
def att_scale(hd):
    """Attention score scale as an integer: round(2^16 / sqrt(hd)) (8192 = 2^13 exactly for hd = 64)."""
    return int(round(2**16 / math.sqrt(hd)))


def rshift(x, n):
    n = int(n)
    if n <= 0:
        return x << -n
    return (x + (I64(1) << (n - 1))) >> n


def rshift_arr(x, n):
    n = np.asarray(n, dtype=I64)
    r = (x + (I64(1) << np.maximum(n - 1, 0))) >> np.maximum(n, 0)
    return np.where(n > 0, r, x << np.maximum(-n, 0))


def bitlen(v):
    return int(v).bit_length()


def blockf(x, e):
    sh = max(0, bitlen(np.abs(x).max()) - 15)
    return np.clip(rshift(x, sh), -32767, 32767), e + sh


def tdiv(a, b):
    return np.sign(a) * (np.abs(a) // b)


def exact_mm(a, b, bound):
    """a @ b for integer arrays via float64 BLAS; exact because |every partial sum| <= bound < 2^53."""
    if bound >= 2**53:
        return a.astype(I64) @ b.astype(I64)
    return np.rint(a.astype(np.float64) @ b.astype(np.float64)).astype(I64)


class IntModel:
    def __init__(self, bundle, maxlen):
        z = np.load(bundle)
        cfg = json.loads(bytes(z["cfg"]).decode())
        self.D, self.I, self.L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
        self.nq, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = self.D // self.nq
        self.V = cfg["vocab_size"]
        self.SC = att_scale(self.hd)

        def mat(n):
            q, s = z[n + ".q"], z[n + ".s"]
            o, i = q.shape
            wp = (q.astype(np.float64).reshape(o, i // G, G) * s.astype(np.float64)[:, :, None]).reshape(o, i)
            return wp, int(z[n + ".E"]), int(np.abs(s).max()) * 127 * 32767 * i

        self.emb_q, self.emb_s = z["emb.q"], z["emb.s"].astype(I64)
        self.emb = mat("emb")
        self.layers = []
        for l in range(self.L):
            self.layers.append(dict(qkv=mat(f"{l}.qkv"), o=mat(f"{l}.o"), gu=mat(f"{l}.gu"), d=mat(f"{l}.d"),
                                    b=z[f"{l}.bqkv"], g1=z[f"{l}.g1"], g2=z[f"{l}.g2"]))
        self.gf = z["gf"]
        self.C, self.S = z["ROPE_C"].astype(I64), z["ROPE_S"].astype(I64)
        self.EXPA, self.EXPF, self.SIG = z["EXPA"], z["EXPF"], z["SIG"]
        self.K = np.zeros((self.L, self.nkv, maxlen, self.hd), np.int64)
        self.Vc = np.zeros((self.L, self.nkv, maxlen, self.hd), np.int64)
        self.pos = 0

    def matvec(self, W, me, to):
        wp, E, bound = W
        m, e = me
        assert bound < 2**53
        acc = np.rint(wp @ m.astype(np.float64)).astype(I64)
        return rshift(acc, to - (E + e))

    def norm(self, h, g):
        sh = max(0, bitlen(np.abs(h).max()) - 20)
        x = rshift(h, sh)
        ms = int((x * x).sum()) // self.D + (1 << max(0, 48 - 2 * sh)) // 10**6
        r = max(1, math.isqrt(ms))
        return blockf(tdiv(x * g * (1 << 14), r), -30)

    def rope(self, x, pos):
        c, s = self.C[pos], self.S[pos]
        h = self.hd // 2
        x1, x2 = x[:, :h], x[:, h:]
        return np.concatenate([rshift(x1 * c - x2 * s, 14), rshift(x2 * c + x1 * s, 14)], axis=1)

    def attn(self, l, q, T):
        grp, hd = self.nq // self.nkv, self.hd
        out = np.empty((self.nq, hd), I64)
        for j in range(self.nkv):
            K, Vv = self.K[l, j, :T], self.Vc[l, j, :T]
            qs = q[j * grp : (j + 1) * grp]
            sh = np.array([max(0, bitlen(np.abs(r).max()) - 15) for r in qs], dtype=I64)
            mq = np.clip(rshift_arr(qs, sh[:, None]), -32767, 32767)
            acc = exact_mm(K, mq.T, int(np.abs(K).max()) * 32767 * hd)  # [T, grp]
            # q e = -24 + sh, K e = -12, times 2^-3 -> e = -39 + sh; to e = -10
            s = rshift_arr(acc * self.SC, (42 - sh)[None, :])
            d = s.max(axis=0)[None, :] - s
            w = self.EXPA[np.minimum(d, NA - 1)]
            Z = w.sum(axis=0)  # [grp]
            O = exact_mm(w.T, Vv, int(Z.max()) * int(np.abs(Vv).max()))  # [grp, hd]
            a = np.abs(O)
            hi = a // Z[:, None]
            rem = a - hi * Z[:, None]
            out[j * grp : (j + 1) * grp] = np.sign(O) * ((hi << 12) + (rem << 12) // Z[:, None])
        return out

    def step(self, tok):
        """Feed one token; return int64 cumulative frequencies for the next token."""
        pos, D, I = self.pos, self.D, self.I
        wp, E, _ = self.emb
        row = self.emb_q[tok].astype(I64) * np.repeat(self.emb_s[tok], G)
        h = rshift(row, -24 - E)
        nq, nkv, hd = self.nq, self.nkv, self.hd
        for l, Ly in enumerate(self.layers):
            x = self.matvec(Ly["qkv"], self.norm(h, Ly["g1"]), -24) + Ly["b"]
            q = self.rope(x[: nq * hd].reshape(nq, hd), pos)
            k = self.rope(x[nq * hd : (nq + nkv) * hd].reshape(nkv, hd), pos)
            v = x[(nq + nkv) * hd :].reshape(nkv, hd)
            lim = 2**22 - 1
            self.K[l, :, pos] = np.clip(rshift(k, 12), -lim, lim)
            self.Vc[l, :, pos] = np.clip(rshift(v, 12), -lim, lim)
            o = self.attn(l, q, pos + 1).reshape(-1)
            h = h + self.matvec(Ly["o"], blockf(o, -24), -24)
            gu = self.matvec(Ly["gu"], self.norm(h, Ly["g2"]), -16)
            g, u = gu[:I], gu[I:]
            a = rshift(g * self.SIG[np.clip(rshift(g, 6), -NS, NS) + NS], 16)
            p = rshift(a * u, 16)
            h = h + self.matvec(Ly["d"], blockf(p, -16), -24)
        lg = self.matvec(self.emb, self.norm(h, self.gf), -10)
        f = self.EXPF[np.minimum(lg.max() - lg, NF - 1)]
        Z = int(f.sum())
        self.pos += 1
        return np.cumsum(f * (TOTAL - self.V) // Z + 1)


# ---------------------------------------------------------------- float reference (same architecture, float64)
class FloatModel:
    def __init__(self, src, bundle=None, maxlen=4096):
        cfg = json.loads((Path(src) / "config.json").read_text())
        self.D, self.I, self.L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
        self.nq, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd, self.eps = self.D // self.nq, cfg["rms_norm_eps"]
        f64 = lambda a: a.astype(np.float64)
        W = load_st(src)
        if bundle is not None:  # same float math, but with the bundle's int8 weights
            z = np.load(bundle)

            def deq(n):
                q, s = z[n + ".q"], z[n + ".s"]
                o, i = q.shape
                return (q.astype(np.float64).reshape(o, i // G, G) * s[:, :, None] * 2.0 ** int(z[n + ".E"])).reshape(o, i)
        self.emb = deq("emb") if bundle is not None else f64(W["model.embed_tokens.weight"])
        self.layers = []
        for l in range(self.L):
            p = f"model.layers.{l}."
            a = p + "self_attn."
            if bundle is not None:
                qkv, o, gu, d = deq(f"{l}.qkv"), deq(f"{l}.o"), deq(f"{l}.gu"), deq(f"{l}.d")
            else:
                qkv = f64(np.concatenate([W[a + x + "_proj.weight"] for x in "qkv"]))
                o = f64(W[a + "o_proj.weight"])
                gu = f64(np.concatenate([W[p + "mlp.gate_proj.weight"], W[p + "mlp.up_proj.weight"]]))
                d = f64(W[p + "mlp.down_proj.weight"])
            self.layers.append(dict(qkv=qkv, o=o, gu=gu, d=d,
                                    b=f64(np.concatenate([W[a + x + "_proj.bias"] for x in "qkv"])),
                                    g1=f64(W[p + "input_layernorm.weight"]), g2=f64(W[p + "post_attention_layernorm.weight"])))
        self.gf = f64(W["model.norm.weight"])
        self.inv = cfg["rope_theta"] ** (-np.arange(0, self.hd, 2, dtype=np.float64) / self.hd)
        self.K = np.zeros((self.L, self.nkv, maxlen, self.hd))
        self.Vc = np.zeros((self.L, self.nkv, maxlen, self.hd))
        self.V = cfg["vocab_size"]
        self.pos = 0
        self.stats = {}

    def norm(self, x, g):
        return x / np.sqrt((x * x).mean() + self.eps) * g

    def rope(self, x, pos):
        c, s = np.cos(pos * self.inv), np.sin(pos * self.inv)
        h = self.hd // 2
        x1, x2 = x[:, :h], x[:, h:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=1)

    def track(self, k, v):
        self.stats[k] = max(self.stats.get(k, 0), float(np.abs(v).max()))

    def step(self, tok):
        pos, nq, nkv, hd, I = self.pos, self.nq, self.nkv, self.hd, self.I
        h = self.emb[tok].copy()
        grp = nq // nkv
        for l, Ly in enumerate(self.layers):
            x = Ly["qkv"] @ self.norm(h, Ly["g1"]) + Ly["b"]
            q = self.rope(x[: nq * hd].reshape(nq, hd), pos)
            k = self.rope(x[nq * hd : (nq + nkv) * hd].reshape(nkv, hd), pos)
            v = x[(nq + nkv) * hd :].reshape(nkv, hd)
            self.track("q", q); self.track("k", k); self.track("v", v)
            self.K[l, :, pos], self.Vc[l, :, pos] = k, v
            o = np.empty((nq, hd))
            for j in range(nkv):
                s = self.K[l, j, : pos + 1] @ q[j * grp : (j + 1) * grp].T / math.sqrt(hd)
                w = np.exp(s - s.max(axis=0))
                o[j * grp : (j + 1) * grp] = (w.T @ self.Vc[l, j, : pos + 1]) / w.sum(axis=0)[:, None]
            h = h + Ly["o"] @ o.reshape(-1)
            gu = Ly["gu"] @ self.norm(h, Ly["g2"])
            g, u = gu[:I], gu[I:]
            self.track("gu", gu)
            h = h + Ly["d"] @ (g / (1 + np.exp(-g)) * u)
            self.track("h", h)
        lg = self.emb @ self.norm(h, self.gf)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        self.pos += 1
        return np.cumsum(np.floor(p * (TOTAL - self.V)).astype(I64) + 1)


# ---------------------------------------------------------------- CLI
def tokenizer(src):
    sys.path.insert(0, str(Path(__file__).parent))
    from transformers import AutoTokenizer
    from segcodec import encode_text
    hf = AutoTokenizer.from_pretrained(src)
    start = json.loads((Path(src) / "config.json").read_text())["bos_token_id"]  # spec: the config's bos, not the tokenizer's
    return hf, start, encode_text


def bits(cum, t):
    lo = int(cum[t - 1]) if t else 0
    return -math.log2((int(cum[t]) - lo) / int(cum[-1]))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "build":
        t0 = time.time()
        build(sys.argv[2], sys.argv[3])
        print(f"built {sys.argv[3]} in {time.time() - t0:.0f}s", flush=True)
    elif mode == "score":
        src, bundle, path = sys.argv[2:5]
        variants = (sys.argv[5] if len(sys.argv) > 5 else "float,f8,int").split(",")
        hf, start, encode_text = tokenizer(src)
        text = Path(path).read_bytes().decode("utf-8")
        ids = encode_text(hf, text)
        for var in variants:
            t0 = time.time()
            m = {"float": lambda: FloatModel(src, None, len(ids) + 1), "f8": lambda: FloatModel(src, bundle, len(ids) + 1),
                 "int": lambda: IntModel(bundle, len(ids) + 1)}[var]()
            tl = time.time() - t0
            t0 = time.time()
            prev, tot = start, 0.0
            for k, t in enumerate(ids):
                tot += bits(m.step(prev), t)
                prev = t
                if (k + 1) % 500 == 0:
                    print(f"  {time.strftime('%T')} {var} {k + 1}/{len(ids)} {tot / 8:.0f} bytes so far", flush=True)
            dt = time.time() - t0
            extra = f" stats {json.dumps({k: round(v, 1) for k, v in m.stats.items()})}" if var != "int" else ""
            print(f"{time.strftime('%T')} {var}: {len(text.encode())} bytes, {len(ids)} tokens -> {tot / 8:.1f} bytes ideal "
                  f"(+4 header), {dt:.0f}s ({len(ids) / dt:.1f} tok/s), load {tl:.0f}s{extra}", flush=True)
    elif mode == "hashes":  # sha256 of each frequency table for the first N tokens: compare implementations
        import hashlib
        src, bundle, path, n = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
        hf, start, encode_text = tokenizer(src)
        ids = encode_text(hf, Path(path).read_bytes().decode("utf-8"))[:n]
        m, prev = IntModel(bundle, n + 1), start
        for k, t in enumerate(ids):
            print(k, hashlib.sha256(m.step(prev).astype("<i8").tobytes()).hexdigest()[:16], flush=True)
            prev = t
    elif mode in ("encode", "decode"):
        sys.path.insert(0, str(Path(__file__).parent))
        from segcodec import Encoder, Decoder
        src, bundle, a, b = sys.argv[2:6]
        hf, start, encode_text = tokenizer(src)
        t0 = time.time()
        if mode == "encode":
            text = Path(a).read_bytes().decode("utf-8")
            ids = encode_text(hf, text)
            m, enc, prev = IntModel(bundle, len(ids) + 1), Encoder(), start
            for t in ids:
                enc.code(m.step(prev), t)
                prev = t
            body = enc.finish()
            Path(b).write_bytes(struct.pack("<I", len(ids)) + body)
            print(f"encode: {len(text.encode())} -> {4 + len(body)} bytes, {len(ids)} tokens, {time.time() - t0:.0f}s", flush=True)
        else:
            data = Path(a).read_bytes()
            (n,) = struct.unpack("<I", data[:4])
            m, dec, prev, ids = IntModel(bundle, n + 1), Decoder(data[4:]), start, []
            for _ in range(n):
                prev = dec.code(m.step(prev))
                ids.append(prev)
            text = hf.decode(ids, clean_up_tokenization_spaces=False)
            Path(b).write_bytes(text.encode("utf-8"))
            print(f"decode: {len(data)} -> {len(text.encode())} bytes, {n} tokens, {time.time() - t0:.0f}s", flush=True)
