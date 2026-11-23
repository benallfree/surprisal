"""The integer codec of intref.py in PyTorch, for any device (cpu, cuda). Same spec, same tables, bit for bit.

Local check against the numpy reference (prints one hash per frequency table, same format as `intref.py hashes`):

    python llm/intgpu.py hashes <bundle.npz> <hf-model-dir> <text> <N> [cpu|cuda]

On Modal (bundle in Volume `compress13k-int` at /int/<name>.npz):

    .venv/bin/modal run llm/intgpu.py --gpu H100 --task roundtrip --fast     # full file: encode, decode, compare
    (--fast = the Triton/CUDA-graph path, over 100x faster; without it, the plain PyTorch version of the spec)
    .venv/bin/modal run llm/intgpu.py --gpu H100 --task hashes --n 64        # hashes, to diff against the Mac
    .venv/bin/modal run llm/intgpu.py --gpu H100 --task decode --archive a.13k   # decode an archive made elsewhere
    .venv/bin/modal run llm/intgpu.py --gpu H100 --task encode --file x.html     # encode here, decode elsewhere

Matrix products run as float64 matmuls on exact integers: every partial sum is an integer below 2^53 (int8 x int16
scale = 2^22 per weight, x 2^15 activation, summed over <= 4864 inputs is < 2^50; K, V are clipped to 2^22), so the
result is exact in any order, on any hardware with IEEE float64. Everything else is int64 elementwise math, integer
sums, max, and table gathers.
"""
import hashlib, json, math, os, struct, sys, time
from pathlib import Path

G, TOTAL = 64, 1 << 28
NA, NF, NS = 16 * 1024, 24 * 1024, 16 * 1024
PREC = 32
TOP, HALF, QTR = (1 << PREC) - 1, 1 << (PREC - 1), 1 << (PREC - 2)


def model_class():
    import numpy as np
    import torch

    L64 = torch.int64

    class TorchInt:
        def __init__(self, bundle, maxlen, dev):
            z = np.load(bundle)
            cfg = json.loads(bytes(z["cfg"]).decode())
            self.dev = dev
            self.D, self.I, self.L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
            self.nq, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
            self.hd = self.D // self.nq
            self.V = cfg["vocab_size"]
            self.start = cfg["bos_token_id"]
            self.SC = int(round(2**16 / math.sqrt(self.hd)))  # attention scale, see intref.att_scale
            T = lambda a, dt=L64: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)

            def mat(n):
                q, s = z[n + ".q"], z[n + ".s"]
                o, i = q.shape
                qd = torch.from_numpy(q).to(dev).to(torch.float64).view(o, i // G, G)
                wp = (qd * torch.from_numpy(s).to(dev).to(torch.float64)[:, :, None]).view(o, i)
                return wp, int(z[n + ".E"])

            self.emb_q, self.emb_s = T(z["emb.q"]), T(z["emb.s"])
            self.emb = mat("emb")
            self.layers = []
            for l in range(self.L):
                self.layers.append(dict(qkv=mat(f"{l}.qkv"), o=mat(f"{l}.o"), gu=mat(f"{l}.gu"), d=mat(f"{l}.d"),
                                        b=T(z[f"{l}.bqkv"]), g1=T(z[f"{l}.g1"]), g2=T(z[f"{l}.g2"])))
            self.gf = T(z["gf"])
            self.C, self.S = T(z["ROPE_C"]), T(z["ROPE_S"])
            self.EXPA, self.EXPF, self.SIG = T(z["EXPA"]), T(z["EXPF"]), T(z["SIG"])
            self.POW2 = torch.tensor([1 << k for k in range(63)], dtype=L64, device=dev)
            self.EPS = torch.tensor([(1 << max(0, 48 - 2 * s)) // 10**6 for s in range(64)], dtype=L64, device=dev)
            self.maxlen = maxlen
            self.K = torch.zeros(self.L, self.nkv, maxlen, self.hd, dtype=torch.float64, device=dev)
            self.Vc = torch.zeros(self.L, self.nkv, maxlen, self.hd, dtype=torch.float64, device=dev)
            self.ar = torch.arange(maxlen, device=dev)
            self.pos = 0

        # ---- helpers (n: int64 tensor, any shape broadcastable to x)
        def rshift(self, x, n):
            n = torch.as_tensor(n, dtype=L64, device=self.dev)
            one = torch.ones((), dtype=L64, device=self.dev)
            r = (x + (one << (n - 1).clamp(min=0))) >> n.clamp(min=0)
            return torch.where(n > 0, r, x << (-n).clamp(min=0))

        def bitlen(self, v):  # v: nonnegative int64 tensor [...]; number of bits
            return (v[..., None] >= self.POW2).sum(-1)

        def blockf(self, x, e):
            sh = (self.bitlen(x.abs().max()) - 15).clamp(min=0)
            return self.rshift(x, sh).clamp(-32767, 32767), e + sh

        def matvec(self, W, me, to):
            wp, E = W
            m, e = me
            acc = (wp @ m.to(torch.float64)).to(L64)
            return self.rshift(acc, to - (E + e))

        def isqrt(self, v):
            r = torch.sqrt(v.to(torch.float64)).to(L64)
            for _ in range(2):
                r = r - (r * r > v).to(L64)
                r = r + ((r + 1) * (r + 1) <= v).to(L64)
            return r

        def norm(self, h, g):
            sh = (self.bitlen(h.abs().max()) - 20).clamp(min=0)
            x = self.rshift(h, sh)
            ms = (x * x).sum() // self.D + self.EPS[sh]
            r = self.isqrt(ms).clamp(min=1)
            y = torch.div(x * g * (1 << 14), r, rounding_mode="trunc")
            return self.blockf(y, torch.tensor(-30, dtype=L64, device=self.dev))

        def rope(self, x, pos):
            c, s = self.C[pos], self.S[pos]
            h = self.hd // 2
            x1, x2 = x[:, :h], x[:, h:]
            return torch.cat([self.rshift(x1 * c - x2 * s, 14), self.rshift(x2 * c + x1 * s, 14)], dim=1)

        def attn(self, l, q, T):
            grp, hd = self.nq // self.nkv, self.hd
            qs = q.view(self.nkv, grp, hd)
            sh = (self.bitlen(qs.abs().amax(-1)) - 15).clamp(min=0)  # [nkv, grp]
            mq = self.rshift(qs, sh[..., None]).clamp(-32767, 32767)
            K, Vv = self.K[l, :, :T], self.Vc[l, :, :T]  # [nkv, T, hd] float64 holding ints
            acc = torch.matmul(K, mq.to(torch.float64).transpose(1, 2)).to(L64)  # [nkv, T, grp]
            s = self.rshift(acc * self.SC, (42 - sh)[:, None, :])
            d = s.amax(1, keepdim=True) - s
            w = self.EXPA[d.clamp(max=NA - 1)]
            Z = w.sum(1)  # [nkv, grp]
            O = torch.matmul(w.transpose(1, 2).to(torch.float64), Vv).to(L64)  # [nkv, grp, hd]
            a, Zb = O.abs(), Z[..., None]
            hi = a // Zb
            rem = a - hi * Zb
            return (torch.sign(O) * ((hi << 12) + (rem << 12) // Zb)).reshape(self.nq, hd)

        def step(self, tok):
            pos, I, nq, nkv, hd = self.pos, self.I, self.nq, self.nkv, self.hd
            e24 = torch.tensor(-24, dtype=L64, device=self.dev)
            _, E = self.emb
            row = self.emb_q[tok] * torch.repeat_interleave(self.emb_s[tok], G)
            h = self.rshift(row, -24 - E)
            lim = 2**22 - 1
            tr = getattr(self, "tr", None)
            rec = (lambda k, v: tr.append((k, v.clone()))) if tr is not None else (lambda k, v: None)
            rec("h0", h)
            for l, Ly in enumerate(self.layers):
                n1 = self.norm(h, Ly["g1"])
                if l == 0: rec("n1m", n1[0]); rec("n1e", n1[1])
                x = self.matvec(Ly["qkv"], n1, -24) + Ly["b"]
                q = self.rope(x[: nq * hd].view(nq, hd), pos)
                k = self.rope(x[nq * hd : (nq + nkv) * hd].view(nkv, hd), pos)
                v = x[(nq + nkv) * hd :].view(nkv, hd)
                self.K[l, :, pos] = self.rshift(k, 12).clamp(-lim, lim).to(torch.float64)
                self.Vc[l, :, pos] = self.rshift(v, 12).clamp(-lim, lim).to(torch.float64)
                if l == 0: rec("kq", self.K[l, :, pos].to(L64)); rec("vq", self.Vc[l, :, pos].to(L64))
                o = self.attn(l, q, pos + 1).reshape(-1)
                bo = self.blockf(o, e24)
                if l == 0: rec("mo", bo[0]); rec("eo", bo[1])
                h = h + self.matvec(Ly["o"], bo, -24)
                if l == 0: rec("h_att", h)
                gu = self.matvec(Ly["gu"], self.norm(h, Ly["g2"]), -16)
                g, u = gu[:I], gu[I:]
                a = self.rshift(g * self.SIG[self.rshift(g, 6).clamp(-NS, NS) + NS], 16)
                p = self.rshift(a * u, 16)
                bp = self.blockf(p, e24 + 8)
                if l == 0: rec("mm", bp[0]); rec("em", bp[1])
                h = h + self.matvec(Ly["d"], bp, -24)
                if l == 0: rec("h_l0", h)
            rec("h_final", h)
            lg = self.matvec(self.emb, self.norm(h, self.gf), -10)
            f = self.EXPF[(lg.max() - lg).clamp(max=NF - 1)]
            Z = f.sum()
            self.pos += 1
            return torch.cumsum(f * (TOTAL - self.V) // Z + 1, 0)

    return TorchInt


# ---------------------------------------------------------------- coder (same as segcodec.py)
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
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8))


class Decoder:
    def __init__(self, data):
        self.data, self.pos = data, 0
        self.low, self.high, self.value = 0, TOP, 0
        for _ in range(PREC):
            self.value = 2 * self.value + self.bit()

    def bit(self):
        i = self.pos
        self.pos += 1
        return (self.data[i >> 3] >> (7 - (i & 7))) & 1 if (i >> 3) < len(self.data) else 0

    def target(self, total):
        return ((self.value - self.low + 1) * total - 1) // (self.high - self.low + 1)

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


def table_hash(cum):
    import numpy as np
    return hashlib.sha256(cum.cpu().numpy().astype("<i8").tobytes()).hexdigest()[:16]


def run_hashes(m, start, ids, out=print):
    import torch
    prev = start
    for k, t in enumerate(ids):
        out(f"{k} {table_hash(m.step(torch.tensor(prev, device=m.dev)))}")
        prev = t


def run_encode(m, start, ids):
    import torch
    enc, prev = Encoder(), start
    for t in ids:
        cum = m.step(torch.tensor(prev, device=m.dev))
        c = torch.stack([cum[-1], cum[t], cum[t - 1] if t else torch.zeros((), dtype=cum.dtype, device=cum.device)]).tolist()
        enc.code_counts(*c)
        prev = t
    return struct.pack("<I", len(ids)) + enc.finish()


def run_decode(m, start, data):
    import torch
    (n,) = struct.unpack("<I", data[:4])
    dec, prev, ids = Decoder(data[4:]), start, []
    for _ in range(n):
        cum = m.step(torch.tensor(prev, device=m.dev))
        total = int(cum[-1])
        t = int((cum <= dec.target(total)).sum())
        hi, lo = int(cum[t]), int(cum[t - 1]) if t else 0
        dec.update(total, hi, lo)
        ids.append(t)
        prev = t
    return ids


# ---------------------------------------------------------------- fast CUDA version (same spec)
def fast_class():
    """Same integer spec on CUDA, fast. Per layer, 10 Triton launches replayed as a CUDA graph (one per 1,024-token
    bucket of attention length):
      matvec qkv -> qkv epilogue (bias, rope, q blockf, KV store) -> scores -> weights and sums -> merge (divide, blockf)
      -> matvec o -> residual + norm -> matvec gate/up -> MLP -> matvec down (applies the MLP's shift) -> residual + norm.
    Matvecs read int8 weights, form exact int32 group sums times the integer scale in int64, and split K across
    programs whose partials are added with integer atomics (exact in any order). The KV cache is stored as three int8
    digit planes (K, V < 2^22), so attention runs as int8 tensor-core dots on digit-split operands, recombined with
    shifts; one program covers all q heads of a kv head. The softmax max and the sums across chunks use integer atomics.
    The output table (logits, exp table, frequencies, prefix sums) is a matvec epilogue plus three small kernels.
    Nothing here depends on the order of operations, so the tables match intref.py bit for bit.
    (fused=False runs an older, slower path built from torch.compile pieces; kept for stage-by-stage debugging.)"""
    import numpy as np
    import torch
    import triton
    import triton.language as tl

    L64 = torch.int64
    torch._dynamo.config.cache_size_limit = 256  # per-bucket shapes
    from triton.language.extra import libdevice
    globals().update(tl=tl, triton=triton, libdevice=libdevice)  # Triton resolves callee globals in module globals

    @triton.jit
    def k_mv(q_ptr, s_ptr, m_ptr, out_ptr, O, K: tl.constexpr, BO: tl.constexpr, TPS: tl.constexpr):
        # split-K: program (i, sk) covers rows i*BO.. and 128-column tiles sk*TPS..; partials summed by the consumer
        NG: tl.constexpr = K // 64
        NT: tl.constexpr = K // 128
        sk = tl.program_id(1)
        rows = tl.program_id(0) * BO + tl.arange(0, BO)
        rmask = rows < O
        acc = tl.zeros([BO], dtype=tl.int64)
        for g2 in range(sk * TPS, tl.minimum(sk * TPS + TPS, NT)):
            cols = g2 * 128 + tl.arange(0, 128)
            w = tl.load(q_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None], other=0).to(tl.int32)
            x = tl.load(m_ptr + cols).to(tl.int32)
            gs = tl.sum(tl.reshape(w * x[None, :], [BO, 2, 64]), axis=2)  # exact int32 (< 2^28)
            sc = tl.load(s_ptr + rows[:, None] * NG + g2 * 2 + tl.arange(0, 2)[None, :], mask=rmask[:, None], other=0)
            acc += tl.sum(gs.to(tl.int64) * sc.to(tl.int64), axis=1)
        tl.store(out_ptr + sk * O + rows, acc, mask=rmask)

    @triton.jit
    def k_att1(q_ptr, sh_ptr, K_ptr, S_ptr, M_ptr, pos_ptr, maxlen, NCH, GRP: tl.constexpr, CH: tl.constexpr, HD: tl.constexpr):
        j, c = tl.program_id(0), tl.program_id(1)
        T = tl.load(pos_ptr) + 1
        t = c * CH + tl.arange(0, CH)
        d = tl.arange(0, HD)
        kk = tl.load(K_ptr + j * maxlen * HD + t[:, None] * HD + d[None, :], mask=(t < T)[:, None], other=0).to(tl.float64)
        for g in tl.static_range(GRP):
            qg = tl.load(q_ptr + (j * GRP + g) * HD + d).to(tl.float64)
            acc = tl.sum(kk * qg[None, :], axis=1).to(tl.int64)  # exact: integer sums below 2^44
            n = 29 - tl.load(sh_ptr + j * GRP + g)
            r = (acc + (tl.full([CH], 1, tl.int64) << tl.maximum(n - 1, 0))) >> tl.maximum(n, 0)
            r = tl.where(n > 0, r, acc << tl.maximum(-n, 0))
            r = tl.where(t < T, r, tl.full([CH], -(2**60), tl.int64))
            tl.store(S_ptr + (j * GRP + g) * maxlen + t, r)
            tl.store(M_ptr + (j * NCH + c) * GRP + g, tl.max(r, axis=0))

    @triton.jit
    def k_att2(S_ptr, Mg_ptr, V_ptr, E_ptr, O_ptr, Z_ptr, pos_ptr, maxlen, NCH, NA: tl.constexpr, GRP: tl.constexpr,
               CH: tl.constexpr, HD: tl.constexpr):
        j, c = tl.program_id(0), tl.program_id(1)
        T = tl.load(pos_ptr) + 1
        t = c * CH + tl.arange(0, CH)
        d = tl.arange(0, HD)
        vv = tl.load(V_ptr + j * maxlen * HD + t[:, None] * HD + d[None, :], mask=(t < T)[:, None], other=0).to(tl.float64)
        for g in tl.static_range(GRP):
            s = tl.load(S_ptr + (j * GRP + g) * maxlen + t)
            w = tl.load(E_ptr + tl.minimum(tl.load(Mg_ptr + j * GRP + g) - s, NA - 1))
            tl.store(Z_ptr + (j * NCH + c) * GRP + g, tl.sum(w, axis=0))
            # exact: w <= 2^15, |V| < 2^22, CH-term sums < 2^44
            tl.store(O_ptr + ((j * NCH + c) * GRP + g) * HD + d, tl.sum(w.to(tl.float64)[:, None] * vv, axis=0).to(tl.int64))

    # ---- fused single-program integer kernels (same arithmetic as the spec, fewer launches)
    @triton.jit
    def _rsh(x, n):  # n: int64 scalar or tensor broadcastable to x
        one = x * 0 + 1
        r = (x + (one << tl.maximum(n - 1, 0))) >> tl.maximum(n, 0)
        return tl.where(n > 0, r, x << tl.maximum(-n, 0))

    @triton.jit
    def _rshc(x, N: tl.constexpr):
        return (x + (1 << (N - 1))) >> N

    @triton.jit
    def _bitlen(v):  # v >= 0 (int64, any shape): number of bits, via count-leading-zeros
        return (64 - libdevice.clz(v)).to(tl.int64)  # clz dispatches to __nv_clzll for int64

    @triton.jit
    def _dig(x):  # |x| < 2^22, or 0 <= x <= 2^15: three balanced base-256 digits, each fits int8; x = a0 + a1 2^8 + a2 2^16
        a0 = ((x + 128) & 255) - 128
        x1 = (x - a0) >> 8
        a1 = ((x1 + 128) & 255) - 128
        a2 = (x1 - a1) >> 8
        return a0.to(tl.int8), a1.to(tl.int8), a2.to(tl.int8)

    @triton.jit
    def _dot9(a0, a1, a2, b0, b1, b2):  # exact integer product of two digit-split matrices (int8 dots, int32 sums < 2^22)
        d00 = tl.dot(a0, b0).to(tl.int64)
        m1 = (tl.dot(a0, b1) + tl.dot(a1, b0)).to(tl.int64)
        m2 = (tl.dot(a0, b2) + tl.dot(a1, b1) + tl.dot(a2, b0)).to(tl.int64)
        m3 = (tl.dot(a1, b2) + tl.dot(a2, b1)).to(tl.int64)
        m4 = tl.dot(a2, b2).to(tl.int64)
        return d00 + (m1 << 8) + (m2 << 16) + (m3 << 24) + (m4 << 32)

    @triton.jit
    def _sumsk(acc_ptr, SK: tl.constexpr, O, idx, mask):
        s = tl.load(acc_ptr + idx, mask=mask, other=0)
        for sk in tl.static_range(1, SK):
            s += tl.load(acc_ptr + sk * O + idx, mask=mask, other=0)
        return s

    @triton.jit
    def k_qkv(acc_ptr, SK: tl.constexpr, E_ptr, ein_ptr, b_ptr, C_ptr, S_ptr, pos_ptr, K_ptr, V_ptr, maxlen, mq_ptr, sh_ptr,
              Mg_ptr, NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, QP: tl.constexpr, ZERO: tl.constexpr = False,
              DIG: tl.constexpr = False, ZB: tl.constexpr = 2048):
        """sum partials, rshift to -24, + bias, rope q and k, blockf each q head, store clipped k, v at pos."""
        hz = tl.arange(0, QP)
        tl.store(Mg_ptr + hz, tl.full([QP], -(2**62), tl.int64), mask=hz < NQ)
        O = (NQ + 2 * NKV) * HD
        HH: tl.constexpr = HD // 2
        n = -24 - (tl.load(E_ptr) + tl.load(ein_ptr))
        pos = tl.load(pos_ptr)
        d2 = tl.arange(0, HH)
        c = tl.load(C_ptr + pos * HH + d2)[None, :]
        s = tl.load(S_ptr + pos * HH + d2)[None, :]
        # q heads
        hq = tl.arange(0, QP)
        mq = (hq < NQ)[:, None]
        i1 = hq[:, None] * HD + d2[None, :]
        x1 = _rsh(_sumsk(acc_ptr, SK, O, i1, mq), n) + tl.load(b_ptr + i1, mask=mq, other=0)
        x2 = _rsh(_sumsk(acc_ptr, SK, O, i1 + HH, mq), n) + tl.load(b_ptr + i1 + HH, mask=mq, other=0)
        q1 = _rshc(x1 * c - x2 * s, 14)
        q2 = _rshc(x2 * c + x1 * s, 14)
        mx = tl.maximum(tl.max(tl.abs(q1), axis=1), tl.max(tl.abs(q2), axis=1))
        sh = tl.maximum(_bitlen(mx) - 15, 0)
        tl.store(mq_ptr + i1, tl.minimum(tl.maximum(_rsh(q1, sh[:, None]), -32767), 32767), mask=mq)
        tl.store(mq_ptr + i1 + HH, tl.minimum(tl.maximum(_rsh(q2, sh[:, None]), -32767), 32767), mask=mq)
        tl.store(sh_ptr + hq, sh, mask=hq < NQ)
        # k heads (rope) and v heads
        hk = tl.arange(0, NKV)
        k1i = NQ * HD + hk[:, None] * HD + d2[None, :]
        x1 = _rsh(_sumsk(acc_ptr, SK, O, k1i, k1i >= 0), n) + tl.load(b_ptr + k1i)
        x2 = _rsh(_sumsk(acc_ptr, SK, O, k1i + HH, k1i >= 0), n) + tl.load(b_ptr + k1i + HH)
        LIM: tl.constexpr = 2**22 - 1
        k1 = tl.minimum(tl.maximum(_rshc(_rshc(x1 * c - x2 * s, 14), 12), -LIM), LIM)
        k2 = tl.minimum(tl.maximum(_rshc(_rshc(x2 * c + x1 * s, 14), 12), -LIM), LIM)
        dst = hk[:, None] * maxlen * HD + pos * HD + d2[None, :]
        d = tl.arange(0, HD)
        vi = (NQ + NKV) * HD + hk[:, None] * HD + d[None, :]
        v = _rsh(_sumsk(acc_ptr, SK, O, vi, vi >= 0), n) + tl.load(b_ptr + vi)
        v = tl.minimum(tl.maximum(_rshc(v, 12), -LIM), LIM)
        vdst = hk[:, None] * maxlen * HD + pos * HD + d[None, :]
        if DIG:  # three int8 digit planes (plane stride NKV * maxlen * HD)
            PS = NKV * maxlen * HD
            a0, a1, a2 = _dig(k1)
            tl.store(K_ptr + dst, a0); tl.store(K_ptr + PS + dst, a1); tl.store(K_ptr + 2 * PS + dst, a2)
            a0, a1, a2 = _dig(k2)
            tl.store(K_ptr + dst + HH, a0); tl.store(K_ptr + PS + dst + HH, a1); tl.store(K_ptr + 2 * PS + dst + HH, a2)
            a0, a1, a2 = _dig(v)
            tl.store(V_ptr + vdst, a0); tl.store(V_ptr + PS + vdst, a1); tl.store(V_ptr + 2 * PS + vdst, a2)
        else:
            tl.store(K_ptr + dst, k1.to(tl.int32))
            tl.store(K_ptr + dst + HH, k2.to(tl.int32))
            tl.store(V_ptr + vdst, v.to(tl.int32))
        if ZERO:
            tl.debug_barrier()
            zi = tl.arange(0, ZB)
            tl.store(acc_ptr + zi, tl.zeros([ZB], dtype=tl.int64), mask=zi < O)

    @triton.jit
    def k_mergea(O_ptr, Z_ptr, m_ptr, e_ptr, NQ: tl.constexpr, HD: tl.constexpr, QP: tl.constexpr):
        """o = O / Z at e = -24 (exact hi/remainder division), blockf over all heads; zero the accumulators."""
        hh = tl.arange(0, QP)
        hm = hh < NQ
        d = tl.arange(0, HD)
        oi = hh[:, None] * HD + d[None, :]
        Oa = tl.load(O_ptr + oi, mask=hm[:, None], other=0)
        Z = tl.where(hm, tl.load(Z_ptr + hh, mask=hm, other=1), 1)[:, None]
        tl.store(O_ptr + oi, tl.zeros([QP, HD], dtype=tl.int64), mask=hm[:, None])
        tl.store(Z_ptr + hh, tl.zeros([QP], dtype=tl.int64), mask=hm)
        a = tl.abs(Oa)
        hi = a // Z
        o = (hi << 12) + ((a - hi * Z) << 12) // Z
        o = tl.where(Oa < 0, -o, o)
        sh = tl.maximum(_bitlen(tl.max(tl.max(tl.abs(o), axis=1), axis=0)) - 15, 0)
        tl.store(m_ptr + oi, tl.minimum(tl.maximum(_rsh(o, sh), -32767), 32767), mask=hm[:, None])
        tl.store(e_ptr, sh - 24)

    @triton.jit
    def k_mv2(q_ptr, s_ptr, m_ptr, out_ptr, pmax_ptr, O, K: tl.constexpr, BO: tl.constexpr, TPS: tl.constexpr,
              ATOMIC: tl.constexpr, PRO: tl.constexpr):
        """k_mv; ATOMIC: split-K partials added into out (zeroed by the consumer) with integer atomics;
        PRO: the input is raw p and its blockf shift comes from max |p| (pmax), applied while loading."""
        NG: tl.constexpr = K // 64
        NT: tl.constexpr = K // 128
        sk = tl.program_id(1)
        rows = tl.program_id(0) * BO + tl.arange(0, BO)
        rmask = rows < O
        acc = tl.zeros([BO], dtype=tl.int64)
        if PRO:
            psh = tl.maximum(_bitlen(tl.load(pmax_ptr)) - 15, 0)
        for g2 in range(sk * TPS, tl.minimum(sk * TPS + TPS, NT)):
            cols = g2 * 128 + tl.arange(0, 128)
            w = tl.load(q_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None], other=0).to(tl.int32)
            xm = tl.load(m_ptr + cols)
            if PRO:
                xm = tl.minimum(tl.maximum(_rsh(xm, psh), -32767), 32767)
            x = xm.to(tl.int32)
            gs = tl.sum(tl.reshape(w * x[None, :], [BO, 2, 64]), axis=2)
            sc = tl.load(s_ptr + rows[:, None] * NG + g2 * 2 + tl.arange(0, 2)[None, :], mask=rmask[:, None], other=0)
            acc += tl.sum(gs.to(tl.int64) * sc.to(tl.int64), axis=1)
        if ATOMIC:
            tl.atomic_add(out_ptr + rows, acc.to(tl.uint64, bitcast=True), mask=rmask, sem="relaxed")
        else:
            tl.store(out_ptr + rows, acc, mask=rmask)

    @triton.jit
    def k_mvlog(q_ptr, s_ptr, m_ptr, lg_ptr, E_ptr, ein_ptr, gmax_ptr, O, K: tl.constexpr, BO: tl.constexpr):
        """output-head matvec (no K split) with the logit epilogue: lg = rshift(acc, to e = -10); max by atomic max."""
        NG: tl.constexpr = K // 64
        NT: tl.constexpr = K // 128
        rows = tl.program_id(0) * BO + tl.arange(0, BO)
        rmask = rows < O
        acc = tl.zeros([BO], dtype=tl.int64)
        for g2 in range(0, NT):
            cols = g2 * 128 + tl.arange(0, 128)
            w = tl.load(q_ptr + rows[:, None] * K + cols[None, :], mask=rmask[:, None], other=0).to(tl.int32)
            x = tl.load(m_ptr + cols).to(tl.int32)
            gs = tl.sum(tl.reshape(w * x[None, :], [BO, 2, 64]), axis=2)
            sc = tl.load(s_ptr + rows[:, None] * NG + g2 * 2 + tl.arange(0, 2)[None, :], mask=rmask[:, None], other=0)
            acc += tl.sum(gs.to(tl.int64) * sc.to(tl.int64), axis=1)
        lg = _rsh(acc, -10 - (tl.load(E_ptr) + tl.load(ein_ptr)))
        tl.store(lg_ptr + rows, lg, mask=rmask)
        tl.atomic_max(gmax_ptr, tl.max(tl.where(rmask, lg, -(2**62)), axis=0), sem="relaxed")

    @triton.jit
    def k_tab2(lg_ptr, gmax_ptr, EXPF_ptr, f_ptr, Z_ptr, V, NF: tl.constexpr, BLOCK: tl.constexpr):
        """f = EXPF[min(M - lg, NF - 1)]; Z = sum f by integer atomic add."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mk = i < V
        lg = tl.load(lg_ptr + i, mask=mk, other=0)
        f = tl.load(EXPF_ptr + tl.minimum(tl.load(gmax_ptr) - lg, NF - 1), mask=mk, other=0)
        tl.store(f_ptr + i, f, mask=mk)
        tl.atomic_add(Z_ptr, tl.sum(f, axis=0).to(tl.uint64, bitcast=True), sem="relaxed")

    @triton.jit
    def k_tab3(f_ptr, Z_ptr, cum_ptr, bsum_ptr, V, TOT: tl.constexpr, BLOCK: tl.constexpr):
        """freq = f * (2^28 - V) // Z + 1; block-local inclusive prefix sums and block totals."""
        b = tl.program_id(0)
        i = b * BLOCK + tl.arange(0, BLOCK)
        mk = i < V
        f = tl.load(f_ptr + i, mask=mk, other=0)
        q = tl.where(mk, f * (TOT - V) // tl.load(Z_ptr) + 1, 0)
        tl.store(cum_ptr + i, tl.cumsum(q, axis=0), mask=mk)
        tl.store(bsum_ptr + b, tl.sum(q, axis=0))

    @triton.jit
    def k_tab4(cum_ptr, bsum_ptr, gmax_ptr, Z_ptr, V, NB: tl.constexpr, BLOCK: tl.constexpr):
        """add the sum of earlier blocks; program 0 resets the max and total for the next step."""
        b = tl.program_id(0)
        j = tl.arange(0, NB)
        off = tl.sum(tl.load(bsum_ptr + j, mask=j < b, other=0), axis=0)
        i = b * BLOCK + tl.arange(0, BLOCK)
        mk = i < V
        tl.store(cum_ptr + i, tl.load(cum_ptr + i, mask=mk, other=0) + off, mask=mk)
        if b == 0:
            tl.store(gmax_ptr, tl.full([], -(2**62), tl.int64))
            tl.store(Z_ptr, tl.zeros([], dtype=tl.int64))

    @triton.jit
    def k_norm2(h_ptr, acc_ptr, E_ptr, ein_ptr, pmax_ptr, g_ptr, EPS_ptr, m_ptr, e_ptr,
                D: tl.constexpr, BLOCK: tl.constexpr, ADD: tl.constexpr, EINP: tl.constexpr):
        """optionally h += rshift(matvec sum, to -24) (and zero that sum); then rmsnorm(h, g) -> blockf -> (m, e).
        EINP: the matvec input was the MLP's raw p, so its exponent comes from max |p| (pmax, reset here)."""
        i = tl.arange(0, BLOCK)
        mk = i < D
        h = tl.load(h_ptr + i, mask=mk, other=0)
        if ADD:
            if EINP:
                pm = tl.load(pmax_ptr)
                ein = tl.maximum(_bitlen(pm) - 15, 0) - 16
                tl.store(pmax_ptr, pm * 0)
            else:
                ein = tl.load(ein_ptr)
            n = -24 - (tl.load(E_ptr) + ein)
            h = h + _rsh(tl.load(acc_ptr + i, mask=mk, other=0), n)
            tl.store(acc_ptr + i, h * 0, mask=mk)
            tl.store(h_ptr + i, h, mask=mk)
        sh = tl.maximum(_bitlen(tl.max(tl.abs(h), axis=0)) - 20, 0)
        x = _rsh(h, sh)
        ms = tl.sum(x * x, axis=0) // D + tl.load(EPS_ptr + sh)
        r = tl.sqrt(ms.to(tl.float64)).to(tl.int64)
        r = r - (r * r > ms).to(tl.int64)
        r = r + ((r + 1) * (r + 1) <= ms).to(tl.int64)
        r = r - (r * r > ms).to(tl.int64)
        r = r + ((r + 1) * (r + 1) <= ms).to(tl.int64)
        r = tl.maximum(r, 1)
        a = x * tl.load(g_ptr + i, mask=mk, other=0) * 16384
        aa = tl.abs(a)
        # trunc(aa / r): float64 estimate (quotient < 2^41, so off by at most 1), then exact integer correction
        qt = (aa.to(tl.float64) / r.to(tl.float64)).to(tl.int64)
        qt = qt - (qt * r > aa).to(tl.int64)
        qt = qt + ((qt + 1) * r <= aa).to(tl.int64)
        qt = qt - (qt * r > aa).to(tl.int64)
        qt = qt + ((qt + 1) * r <= aa).to(tl.int64)
        y = tl.where(a < 0, -qt, qt)
        sh2 = tl.maximum(_bitlen(tl.max(aa * 0 + tl.abs(y), axis=0)) - 15, 0)
        tl.store(m_ptr + i, tl.minimum(tl.maximum(_rsh(y, sh2), -32767), 32767), mask=mk)
        tl.store(e_ptr, sh2 - 30)

    @triton.jit
    def k_mlp2(acc_ptr, E_ptr, ein_ptr, SIG_ptr, p_ptr, pmax_ptr, I: tl.constexpr, NS: tl.constexpr, BLOCK: tl.constexpr):
        """parallel: raw p = rshift(a * u, 16) per block, max |p| by atomic max; zero the consumed gate/up sums."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mk = i < I
        n = -16 - (tl.load(E_ptr) + tl.load(ein_ptr))
        ga = tl.load(acc_ptr + i, mask=mk, other=0)
        ua = tl.load(acc_ptr + I + i, mask=mk, other=0)
        tl.store(acc_ptr + i, ga * 0, mask=mk)
        tl.store(acc_ptr + I + i, ua * 0, mask=mk)
        g = _rsh(ga, n)
        u = _rsh(ua, n)
        idx = tl.minimum(tl.maximum(_rshc(g, 6), -NS), NS) + NS
        a = _rshc(g * tl.load(SIG_ptr + idx, mask=mk, other=0), 16)
        p = tl.where(mk, _rshc(a * u, 16), 0)
        tl.store(p_ptr + i, p, mask=mk)
        tl.atomic_max(pmax_ptr, tl.max(tl.abs(p), axis=0), sem="relaxed")

    @triton.jit
    def k_att1d(q_ptr, sh_ptr, K_ptr, S_ptr, Mg_ptr, Cm_ptr, pos_ptr, maxlen, CPP, nch, NCH, NKV: tl.constexpr,
                GRP: tl.constexpr, CH: tl.constexpr, HD: tl.constexpr, GP: tl.constexpr, SC: tl.constexpr):
        """one program per (kv head, CPP chunks), all q heads at once: scores = K @ q^T from int8 digit dots (exact),
        shifted to e = -10, stored; max per head by atomic max."""
        j, c0 = tl.program_id(0), tl.program_id(1) * CPP
        T = tl.load(pos_ptr) + 1
        d = tl.arange(0, HD)
        g = tl.arange(0, GP)
        gm = g < GRP
        q0, q1, q2 = _dig(tl.load(q_ptr + (j * GRP + g[:, None]) * HD + d[None, :], mask=gm[:, None], other=0))
        q0, q1, q2 = tl.trans(q0), tl.trans(q1), tl.trans(q2)  # [HD, GP]
        n = (42 - tl.load(sh_ptr + j * GRP + g, mask=gm, other=0))[None, :]
        PS = NKV * maxlen * HD
        mrun = tl.full([GP], -(2**62), tl.int64)
        for c in range(CPP):
            t = (c0 + c) * CH + tl.arange(0, CH)
            ka = K_ptr + j * maxlen * HD + t[:, None] * HD + d[None, :]
            mk = (t < T)[:, None]
            k0 = tl.load(ka, mask=mk, other=0)
            k1 = tl.load(ka + PS, mask=mk, other=0)
            k2 = tl.load(ka + 2 * PS, mask=mk, other=0)
            acc = _dot9(k0, k1, k2, q0, q1, q2) * SC  # [CH, GP]; |q.K| < 2^44, times SC < 2^16
            r = (acc + ((acc * 0 + 1) << tl.maximum(n - 1, 0))) >> tl.maximum(n, 0)
            r = tl.where(n > 0, r, acc << tl.maximum(-n, 0))
            r = tl.where(mk, r, -(2**60))
            tl.store(S_ptr + (j * GRP + g[None, :]) * maxlen + t[:, None], r, mask=gm[None, :] & (t < nch * CH)[:, None])
            cmax = tl.max(r, axis=0)
            tl.store(Cm_ptr + (j * GRP + g) * NCH + c0 + c, cmax, mask=gm & (c0 + c < nch))  # chunk max per head
            mrun = tl.maximum(mrun, cmax)
        tl.atomic_max(Mg_ptr + j * GRP + g, mrun, mask=gm, sem="relaxed")

    @triton.jit
    def k_att2d(S_ptr, Mg_ptr, Cm_ptr, V_ptr, E_ptr, Oacc_ptr, Zacc_ptr, pos_ptr, maxlen, CPP, nch, NCH, NA: tl.constexpr,
                NKV: tl.constexpr, GRP: tl.constexpr, CH: tl.constexpr, HD: tl.constexpr, GP: tl.constexpr, ZTH: tl.constexpr):
        """w = EXPA[M - s]; Z += sum w; O += w^T @ V from int8 digit dots (exact)."""
        j, c0 = tl.program_id(0), tl.program_id(1) * CPP
        T = tl.load(pos_ptr) + 1
        d = tl.arange(0, HD)
        g = tl.arange(0, GP)
        gm = g < GRP
        mg = tl.load(Mg_ptr + j * GRP + g, mask=gm, other=0)[None, :]
        PS = NKV * maxlen * HD
        oa = tl.zeros([GP, HD], dtype=tl.int64)
        za = tl.zeros([GP], dtype=tl.int64)
        mgv = tl.load(Mg_ptr + j * GRP + g, mask=gm, other=0)
        for c in range(CPP):
            # a chunk whose best score is >= ZTH below the max for every head has all weights EXPA[...] = 0: skip it
            cm = tl.load(Cm_ptr + (j * GRP + g) * NCH + c0 + c, mask=gm & (c0 + c < nch), other=-(2**62))
            if tl.max(tl.where(gm, cm - mgv + ZTH, -1), axis=0) > 0:
                t = (c0 + c) * CH + tl.arange(0, CH)
                va = V_ptr + j * maxlen * HD + t[:, None] * HD + d[None, :]
                mk = (t < T)[:, None]
                v0 = tl.load(va, mask=mk, other=0)
                v1 = tl.load(va + PS, mask=mk, other=0)
                v2 = tl.load(va + 2 * PS, mask=mk, other=0)
                sv = tl.load(S_ptr + (j * GRP + g[None, :]) * maxlen + t[:, None], mask=gm[None, :] & (t < nch * CH)[:, None],
                             other=-(2**60))
                w = tl.load(E_ptr + tl.minimum(mg - sv, NA - 1))  # [CH, GP]; 0 when masked (EXPA[NA-1] = 0)
                za += tl.sum(w, axis=0)
                w0, w1, w2 = _dig(w)
                oa += _dot9(tl.trans(w0), tl.trans(w1), tl.trans(w2), v0, v1, v2)
        tl.atomic_add(Zacc_ptr + j * GRP + g, za.to(tl.uint64, bitcast=True), mask=gm, sem="relaxed")
        tl.atomic_add(Oacc_ptr + (j * GRP + g[:, None]) * HD + d[None, :], oa.to(tl.uint64, bitcast=True), mask=gm[:, None],
                      sem="relaxed")

    class FastInt:
        def __init__(self, bundle, n, log, BUCKET=1024, CH=32, fused=True, APG=192):
            self.tr = None
            z = np.load(bundle)
            cfg = json.loads(bytes(z["cfg"]).decode())
            dev = self.dev = "cuda"
            D, I, Lr = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
            nq, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
            hd, V, grp = D // nq, cfg["vocab_size"], nq // nkv
            self.start, self.V = cfg["bos_token_id"], V
            SC = int(round(2**16 / math.sqrt(hd)))
            NB_ = 1 << (D - 1).bit_length()  # norm block (power of two >= D)
            if not fused:
                assert hd == 64, "the unfused debug path assumes head_dim 64"
            self.BUCKET = BUCKET
            nb = (n + BUCKET) // BUCKET  # positions 0..n-1 need attention length up to n
            maxlen = nb * BUCKET
            Tn = lambda a, dt=L64: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)
            sc = lambda v: torch.tensor(v, dtype=L64, device=dev)

            def mat(nm):
                return Tn(z[nm + ".q"], torch.int8), Tn(z[nm + ".s"], torch.int32), sc(int(z[nm + ".E"]))

            emb_q, emb_s, E_emb = mat("emb")
            layers = [dict(qkv=mat(f"{l}.qkv"), o=mat(f"{l}.o"), gu=mat(f"{l}.gu"), d=mat(f"{l}.d"),
                           b=Tn(z[f"{l}.bqkv"]), g1=Tn(z[f"{l}.g1"]), g2=Tn(z[f"{l}.g2"])) for l in range(Lr)]
            gf = Tn(z["gf"])
            C, S = Tn(z["ROPE_C"]), Tn(z["ROPE_S"])
            EXPA, EXPF, SIG = Tn(z["EXPA"]), Tn(z["EXPF"]), Tn(z["SIG"])
            POW2 = torch.tensor([1 << k for k in range(63)], dtype=L64, device=dev)
            EPS = torch.tensor([(1 << max(0, 48 - 2 * s)) // 10**6 for s in range(64)], dtype=L64, device=dev)
            KV = torch.zeros(Lr, 2, nkv, maxlen, hd, dtype=torch.int32, device=dev)  # unfused path
            KV8 = torch.zeros(Lr, 2, 3, nkv, maxlen, hd, dtype=torch.int8, device=dev)  # fused: three int8 digit planes
            tok, pos = sc(self.start).view(1), sc(0).view(1)
            cum = torch.zeros(V, dtype=L64, device=dev)
            NCH = maxlen // CH
            Sbuf = torch.zeros(nkv * grp, maxlen, dtype=L64, device=dev)
            Mp = torch.zeros(nkv, NCH, grp, dtype=L64, device=dev)
            Op = torch.zeros(nkv, NCH, grp, hd, dtype=L64, device=dev)
            Zp = torch.zeros(nkv, NCH, grp, dtype=L64, device=dev)
            one = sc(1)

            def rshift(x, nn):
                return torch.where(nn > 0, (x + (one << (nn - 1).clamp(min=0))) >> nn.clamp(min=0), x << (-nn).clamp(min=0))

            def bitlen(v):
                return (v[..., None] >= POW2).sum(-1)

            def blockf(x, e):
                sh = (bitlen(x.abs().max()) - 15).clamp(min=0)
                return rshift(x, sh).clamp(-32767, 32767), e + sh

            def norm(h, g):
                sh = (bitlen(h.abs().max()) - 20).clamp(min=0)
                x = rshift(h, sh)
                ms = (x * x).sum() // D + EPS.gather(0, sh.view(1))[0]
                r = torch.sqrt(ms.to(torch.float64)).to(L64)
                for _ in range(2):
                    r = r - (r * r > ms).to(L64)
                    r = r + ((r + 1) * (r + 1) <= ms).to(L64)
                y = torch.div(x * g * (1 << 14), r.clamp(min=1), rounding_mode="trunc")
                return blockf(y, sc(-30))

            def ropef(x, c, s):
                x1, x2 = x[:, : hd // 2], x[:, hd // 2 :]
                return torch.cat([rshift(x1 * c - x2 * s, sc(14)), rshift(x2 * c + x1 * s, sc(14))], dim=1)

            comp = lambda f: torch.compile(f, dynamic=False)

            @comp
            def head(tok, pos, g1):
                row = emb_q.index_select(0, tok)[0].to(L64) * torch.repeat_interleave(emb_s.index_select(0, tok)[0].to(L64), 64)
                h = rshift(row, -24 - E_emb)
                m, e = norm(h, g1)
                return h, m, e, C.index_select(0, pos)[0], S.index_select(0, pos)[0]

            @comp
            def post_qkv(acc, e, E, b, c, s):
                acc = acc.sum(0)
                x = rshift(acc, -24 - (E + e)) + b
                q = ropef(x[: nq * hd].view(nq, hd), c, s)
                k = ropef(x[nq * hd : (nq + nkv) * hd].view(nkv, hd), c, s)
                v = x[(nq + nkv) * hd :].view(nkv, hd)
                lim = 2**22 - 1
                sh = (bitlen(q.abs().amax(-1)) - 15).clamp(min=0)
                mq = rshift(q, sh[:, None]).clamp(-32767, 32767)
                return mq, sh, rshift(k, sc(12)).clamp(-lim, lim).to(torch.int32), rshift(v, sc(12)).clamp(-lim, lim).to(torch.int32)

            @comp
            def gmax(Mpart):
                return Mpart.amax(1)

            @comp
            def post_att(Opart, Zpart):
                O, Z = Opart.sum(1), Zpart.sum(1)[..., None]  # [nkv, grp, hd], [nkv, grp, 1]
                a = O.abs()
                hi = a // Z
                o = torch.sign(O) * ((hi << 12) + ((a - hi * Z) << 12) // Z)
                return blockf(o.reshape(-1), sc(-24))

            @comp
            def mid(h, acc, e, E, g2):
                acc = acc.sum(0)
                h = h + rshift(acc, -24 - (E + e))
                m, e2 = norm(h, g2)
                return h, m, e2

            @comp
            def mlp(acc, e, E):
                acc = acc.sum(0)
                gu = rshift(acc, -16 - (E + e))
                g, u = gu[:I], gu[I:]
                a = rshift(g * SIG[rshift(g, sc(6)).clamp(-NS, NS) + NS], sc(16))
                return blockf(rshift(a * u, sc(16)), sc(-16))

            @comp
            def resid(h, acc, e, E, g):
                acc = acc.sum(0)
                h = h + rshift(acc, -24 - (E + e))
                m, e2 = norm(h, g)
                return h, m, e2

            @comp
            def table(acc, e, E):
                acc = acc.sum(0)
                lg = rshift(acc, -10 - (E + e))
                f = EXPF[(lg.max() - lg).clamp(max=NF - 1)]
                return torch.cumsum(f * (TOTAL - V) // f.sum() + 1, 0)

            bufs = {}

            def mv(W, m, BO=32, TPS=1):
                q, s, E = W
                O, K = q.shape
                SK = triton.cdiv(K // 128, TPS)
                if (O, K) not in bufs:
                    bufs[(O, K)] = torch.empty(SK, O, dtype=L64, device=dev)
                out = bufs[(O, K)]
                k_mv[(triton.cdiv(O, BO), SK)](q, s, m.contiguous(), out, O, K=K, BO=BO, TPS=TPS)
                return out, E

            def forward(Lb):
                nch = Lb // CH
                h, m, e, c, s = head(tok, pos, layers[0]["g1"])
                tr = self.tr
                rec = (lambda k, v: tr.append((k, v.clone()))) if tr is not None else (lambda k, v: None)
                rec("h0", h); rec("n1m", m); rec("n1e", e)
                for l, Ly in enumerate(layers):
                    acc, E = mv(Ly["qkv"], m)
                    mq, sh, kq, vq = post_qkv(acc, e, E, Ly["b"], c, s)
                    KV[l, 0].index_copy_(1, pos, kq[:, None])
                    KV[l, 1].index_copy_(1, pos, vq[:, None])
                    k_att1[(nkv, nch)](mq.contiguous(), sh.contiguous(), KV[l, 0], Sbuf, Mp, pos, maxlen, NCH, GRP=grp, CH=CH, HD=hd)
                    Mg = gmax(Mp[:, :nch])
                    k_att2[(nkv, nch)](Sbuf, Mg, KV[l, 1], EXPA, Op, Zp, pos, maxlen, NCH, NA=NA, GRP=grp, CH=CH, HD=hd)
                    mo, eo = post_att(Op[:, :nch], Zp[:, :nch])
                    if l == 0: rec("kq", kq.to(L64)); rec("vq", vq.to(L64)); rec("mo", mo); rec("eo", eo)
                    acc, E = mv(Ly["o"], mo)
                    h, m, e = mid(h, acc, eo, E, Ly["g2"])
                    if l == 0: rec("h_att", h)
                    acc, E = mv(Ly["gu"], m)
                    mm, em = mlp(acc, e, E)
                    if l == 0: rec("mm", mm); rec("em", em)
                    acc, E = mv(Ly["d"], mm, TPS=2)
                    h, m, e = resid(h, acc, em, E, layers[l + 1]["g1"] if l + 1 < Lr else gf)
                    if l == 0: rec("h_l0", h)
                rec("h_final", h)
                acc, E = mv((emb_q, emb_s, E_emb), m, BO=64, TPS=7)
                cum.copy_(table(acc, e, E))
                pos.add_(1)

            # fused path (the default): 10 launches per layer
            hbuf, mbuf = torch.zeros(D, dtype=L64, device=dev), torch.zeros(D, dtype=L64, device=dev)
            mob, mmb = torch.zeros(D, dtype=L64, device=dev), torch.zeros(I, dtype=L64, device=dev)
            mqb, shb = torch.zeros(nq, hd, dtype=L64, device=dev), torch.zeros(nq, dtype=L64, device=dev)
            e_n, e_o, e_m = (torch.zeros(1, dtype=L64, device=dev) for _ in range(3))
            Mg_acc = torch.zeros(nq, dtype=L64, device=dev)
            Cm = torch.zeros(nq, NCH, dtype=L64, device=dev)  # per-chunk score max
            ZTH = int((EXPA != 0).nonzero().max()) + 1  # EXPA[d] = 0 for every d >= ZTH
            assert bool((EXPA[ZTH:] == 0).all())
            O_acc = torch.zeros(nq, hd, dtype=L64, device=dev)
            Z_acc = torch.zeros(nq, dtype=L64, device=dev)
            O_acc_u, Z_acc_u = O_acc.view(torch.uint64), Z_acc.view(torch.uint64)

            @comp
            def head_h(tok):
                row = emb_q.index_select(0, tok)[0].to(L64) * torch.repeat_interleave(emb_s.index_select(0, tok)[0].to(L64), 64)
                return rshift(row, -24 - E_emb)

            TB = 2048
            NTB = triton.cdiv(V, TB)
            lgbuf, fbuf = torch.zeros(V, dtype=L64, device=dev), torch.zeros(V, dtype=L64, device=dev)
            gmaxb = torch.full((1,), -(2**62), dtype=L64, device=dev)
            Zt = torch.zeros(1, dtype=L64, device=dev)
            Zt_u = Zt.view(torch.uint64)
            bsum = torch.zeros(NTB, dtype=L64, device=dev)
            pbuf = torch.zeros(I, dtype=L64, device=dev)
            pmax = torch.zeros(1, dtype=L64, device=dev)
            outs = {}

            def mv3(W, m, BO, TPS, PRO=False, nw=4):
                q, s_, E = W
                O, K = q.shape
                SK = triton.cdiv(K // 128, TPS)
                if (O, K) not in outs:
                    outs[(O, K)] = torch.zeros(O, dtype=L64, device=dev)
                out = outs[(O, K)]
                k_mv2[(triton.cdiv(O, BO), SK)](q, s_, m, out.view(torch.uint64) if SK > 1 else out, pmax, O, K=K, BO=BO,
                                                TPS=TPS, ATOMIC=SK > 1, PRO=PRO, num_warps=nw)
                return out, E

            def forward3(Lb):
                nch = Lb // CH
                hbuf.copy_(head_h(tok))
                k_norm2[(1,)](hbuf, hbuf, E_emb, e_n, pmax, layers[0]["g1"], EPS, mbuf, e_n, D=D, BLOCK=NB_, ADD=False, EINP=False, num_warps=8)
                for l, Ly in enumerate(layers):
                    acc, E = mv3(Ly["qkv"], mbuf, 32, 1)
                    k_qkv[(1,)](acc, 1, E, e_n, Ly["b"], C, S, pos, KV8[l, 0], KV8[l, 1], maxlen, mqb, shb, Mg_acc,
                                NQ=nq, NKV=nkv, HD=hd, QP=16, ZERO=True, DIG=True, ZB=1 << ((nq + 2 * nkv) * hd - 1).bit_length())
                    cpp = triton.cdiv(nch, APG)  # about APG programs per kv head, each over cpp chunks
                    npg = triton.cdiv(nch, cpp)
                    k_att1d[(nkv, npg)](mqb, shb, KV8[l, 0], Sbuf, Mg_acc, Cm, pos, maxlen, cpp, nch, NCH, NKV=nkv, GRP=grp,
                                        CH=CH, HD=hd, GP=16, SC=SC)
                    k_att2d[(nkv, npg)](Sbuf, Mg_acc, Cm, KV8[l, 1], EXPA, O_acc_u, Z_acc_u, pos, maxlen, cpp, nch, NCH, NA=NA,
                                        NKV=nkv, GRP=grp, CH=CH, HD=hd, GP=16, ZTH=ZTH)
                    k_mergea[(1,)](O_acc, Z_acc, mob, e_o, NQ=nq, HD=hd, QP=16)
                    acc, E = mv3(Ly["o"], mob, 32, 1)
                    k_norm2[(1,)](hbuf, acc, E, e_o, pmax, Ly["g2"], EPS, mbuf, e_n, D=D, BLOCK=NB_, ADD=True, EINP=False, num_warps=8)
                    acc, E = mv3(Ly["gu"], mbuf, 64, 1, nw=2)
                    k_mlp2[(triton.cdiv(I, 1024),)](acc, E, e_n, SIG, pbuf, pmax, I=I, NS=NS, BLOCK=1024)
                    acc, E = mv3(Ly["d"], pbuf, 128, 1, PRO=True)
                    k_norm2[(1,)](hbuf, acc, E, e_n, pmax, layers[l + 1]["g1"] if l + 1 < Lr else gf, EPS, mbuf, e_n,
                                  D=D, BLOCK=NB_, ADD=True, EINP=True, num_warps=8)
                k_mvlog[(triton.cdiv(V, 64),)](emb_q, emb_s, mbuf, lgbuf, E_emb, e_n, gmaxb, V, K=D, BO=64)
                k_tab2[(NTB,)](lgbuf, gmaxb, EXPF, fbuf, Zt_u, V, NF=NF, BLOCK=TB)
                k_tab3[(NTB,)](fbuf, Zt, cum, bsum, V, TOT=TOTAL, BLOCK=TB)
                k_tab4[(NTB,)](cum, bsum, gmaxb, Zt, V, NB=1 << (NTB - 1).bit_length(), BLOCK=TB)
                pos.add_(1)

            if fused:
                forward = forward3

            ids_gpu = torch.zeros(maxlen, dtype=L64, device=dev)
            counts = torch.zeros(maxlen, 3, dtype=L64, device=dev)
            st = torch.zeros(3, dtype=L64, device=dev)
            res = torch.zeros(4, dtype=L64, device=dev)

            def gather3(t):
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
                KV.zero_(); KV8.zero_(); pos.zero_(); tok.fill_(self.start)
                O_acc.zero_(); Z_acc.zero_(); pmax.zero_(); Zt.zero_(); gmaxb.fill_(-(2**62))
                for o_ in outs.values():
                    o_.zero_()
                st.copy_(torch.tensor([0, 0, TOP + 1]))

            t0 = time.time()
            pool = torch.cuda.graph_pool_handle()

            def capture(fn, *a):
                s_ = torch.cuda.Stream()
                s_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_), torch.no_grad():
                    for _ in range(2):
                        reset(); fn(*a)
                torch.cuda.current_stream().wait_stream(s_)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool), torch.no_grad():
                    fn(*a)
                return g

            self.fwd = [capture(forward, (b + 1) * BUCKET) for b in range(nb)]
            self.encg = [capture(enc_step, (b + 1) * BUCKET) for b in range(nb)]
            self.srch = capture(search)
            torch.cuda.synchronize()
            self.t_cap = time.time() - t0
            log(f"{time.strftime('%T')} captured {2 * nb + 1} graphs in {self.t_cap:.1f}s")
            def bench(say):
                """Time each matvec shape and the norm kernel alone, over launch configurations."""
                def tm(fn, reps=100):  # per launch, measured as a CUDA graph of `reps` launches
                    st_ = torch.cuda.Stream()
                    st_.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(st_):
                        fn(); fn()
                    torch.cuda.current_stream().wait_stream(st_)
                    torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        for _ in range(reps):
                            fn()
                    gr.replay(); torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(5):
                        gr.replay()
                    e1.record(); torch.cuda.synchronize()
                    return e0.elapsed_time(e1) / (5 * reps) * 1000
                Ly = layers[0]
                mats = [("qkv", Ly["qkv"], mbuf, False), ("o", Ly["o"], mob, False), ("gu", Ly["gu"], mbuf, False),
                        ("d", Ly["d"], pbuf, True), ("emb", (emb_q, emb_s, E_emb), mbuf, False)]
                for nm, W, mvec, pro in mats:
                    q, s_, E = W
                    O, K = q.shape
                    NT = K // 128
                    res = []
                    for BO in (8, 16, 32, 64, 128):
                        for TPS in sorted({1, 2, 3, 4, 7, NT}):
                            if TPS > NT:
                                continue
                            SK = triton.cdiv(NT, TPS)
                            out = torch.zeros(O, dtype=L64, device=dev)
                            for nw in (2, 4, 8):
                                f = lambda: k_mv2[(triton.cdiv(O, BO), SK)](q, s_, mvec, out.view(torch.uint64) if SK > 1 else out,
                                                                            pmax, O, K=K, BO=BO, TPS=TPS, ATOMIC=SK > 1, PRO=pro,
                                                                            num_warps=nw)
                                try:
                                    res.append((tm(f), BO, TPS, nw))
                                except Exception as ex:
                                    pass
                    res.sort()
                    say(f"mv {nm} {O}x{K}: best " + "; ".join(f"{t:.2f}us BO={b} TPS={tp} nw={w}" for t, b, tp, w in res[:4])
                        + f" | current-ish BO=32: " + "; ".join(f"{t:.2f}us TPS={tp} nw={w}" for t, b, tp, w in res if b == 32 and w == 4)[:120])
                acc0 = torch.zeros(D, dtype=L64, device=dev)
                for nw in (2, 4, 8, 16, 32):
                    f = lambda: k_norm2[(1,)](hbuf, acc0, E_emb, e_o, pmax, Ly["g2"], EPS, mbuf, e_n, D=D, BLOCK=NB_, ADD=True,
                                              EINP=False, num_warps=nw)
                    say(f"norm nw={nw}: {tm(f):.2f}us")
                f = lambda: k_mergea[(1,)](O_acc, Z_acc, mob, e_o, NQ=nq, HD=hd, QP=16)
                say(f"merge: {tm(f):.2f}us")
                f = lambda: k_mlp2[(triton.cdiv(I, 1024),)](outs.get((2 * I, D), torch.zeros(2 * I, dtype=L64, device=dev)), E_emb, e_n,
                                                             SIG, pbuf, pmax, I=I, NS=NS, BLOCK=1024)
                say(f"mlp: {tm(f):.2f}us")
                f = lambda: pmax.add_(0)
                say(f"tiny torch kernel: {tm(f):.2f}us")

            self.bench = bench
            self.forward_eager = forward
            self.__dict__.update(reset=reset, tok=tok, pos=pos, cum=cum, ids_gpu=ids_gpu, counts=counts, st=st, res=res)
            reset()

        def step(self, t):  # for hashes: feed token t, return the table
            k = int(self.pos.item())
            self.tok.fill_(t)
            self.fwd[k // self.BUCKET].replay()
            return self.cum

        def encode(self, ids):
            import torch
            n = len(ids)
            self.reset()
            self.ids_gpu[:n].copy_(torch.tensor(ids))
            torch.cuda.synchronize()
            t0 = time.time()
            for k in range(n):
                self.encg[k // self.BUCKET].replay()
            cnt = self.counts[:n].tolist()
            self.t_gpu = time.time() - t0
            bad = [k for k, (tot, hi, lo) in enumerate(cnt) if not 0 <= lo < hi <= tot]
            if bad:
                raise ValueError(f"bad counts at token {bad[0]}: {cnt[bad[0]]}")
            enc = Encoder()
            for tot, hi, lo in cnt:
                enc.code_counts(tot, hi, lo)
            return struct.pack("<I", n) + enc.finish()

        def decode(self, data):
            import torch
            (n,) = struct.unpack("<I", data[:4])
            self.reset()
            torch.cuda.synchronize()
            dec, out = Decoder(data[4:]), []
            st_h = torch.zeros(3, dtype=torch.long).pin_memory()
            res_h = torch.zeros(4, dtype=torch.long).pin_memory()
            ev = torch.cuda.Event()
            self.fwd[0].replay()
            for k in range(n):
                st_h[0], st_h[1], st_h[2] = dec.value, dec.low, dec.high - dec.low + 1
                self.st.copy_(st_h, non_blocking=True)
                self.srch.replay()
                res_h.copy_(self.res, non_blocking=True)
                ev.record()
                if k + 1 < n:
                    self.fwd[(k + 1) // self.BUCKET].replay()
                ev.synchronize()
                t, hi, lo, tot = res_h.tolist()
                dec.update(tot, hi, lo)
                out.append(t)
            torch.cuda.synchronize()
            return out

    return FastInt


# ---------------------------------------------------------------- Modal
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch==2.8.0", "transformers==4.56.2", "numpy")
             .env({"HF_HOME": "/cache"}))
    app = modal.App("compress13k-int", image=image)
    hfvol = modal.Volume.from_name("compress13k-hf", create_if_missing=True)
    intvol = modal.Volume.from_name("compress13k-int", create_if_missing=True)
    kw = dict(volumes={"/cache": hfvol, "/int": intvol}, timeout=3600)

    def body(task, bundle, repo, text, n, archive, fast=False):
        import traceback
        try:
            import torch
            from transformers import AutoTokenizer
            task, _, targ = task.partition(":")
            log = []
            say = lambda s: (print(s, flush=True), log.append(s))
            say(f"{time.strftime('%T')} GPU: {torch.cuda.get_device_name()}")
            hf = AutoTokenizer.from_pretrained(repo)
            torch.backends.cuda.matmul.allow_tf32 = False
            t0 = time.time()
            ids = encode_text(hf, text) if text is not None else None
            maxlen = (len(ids) if ids else struct.unpack("<I", archive[:4])[0]) + 1
            if fast:
                m = fast_class()(f"/int/{bundle}", maxlen - 1, say, **({"APG": int(targ)} if targ else {}))
            else:
                m = model_class()(f"/int/{bundle}", maxlen, "cuda")
            start = m.start
            say(f"{time.strftime('%T')} loaded in {time.time() - t0:.1f}s")
            res = {"log": log}
            if task == "hashes":
                run_hashes(m, start, ids[:n], say)
            if task == "bench":
                m.bench(say)
            if task == "prof":
                from torch.profiler import profile, ProfilerActivity
                nb = len(m.fwd)
                for bk in sorted({0, nb // 2, nb - 1}):
                    m.reset(); m.pos.fill_(bk * m.BUCKET); torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(100):
                        m.pos.fill_(bk * m.BUCKET)
                        m.fwd[bk].replay()
                    e1.record(); torch.cuda.synchronize()
                    say(f"forward graph at context {(bk + 1) * m.BUCKET}: {e0.elapsed_time(e1) / 100:.3f} ms/step")
                for bk in sorted({0, nb - 1}):  # kernel table at short and full context (5 steps each)
                    p0 = bk * m.BUCKET + m.BUCKET // 2 if bk else 0
                    m.reset(); torch.cuda.synchronize()
                    with torch.no_grad():
                        m.pos.fill_(p0); m.forward_eager((bk + 1) * m.BUCKET); torch.cuda.synchronize()
                        with profile(activities=[ProfilerActivity.CUDA]) as pr:
                            for _ in range(5):
                                m.pos.fill_(p0)
                                m.forward_eager((bk + 1) * m.BUCKET)
                            torch.cuda.synchronize()
                    tbl = f"context {p0 + 1}:\n" + pr.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=40)
                    print(tbl, flush=True); log.append(tbl)
            if task == "debug":  # first token: fast vs reference, stage by stage
                g0 = m.step(torch.tensor(start, device="cuda")).clone()
                say(f"graph-first table hash {table_hash(g0)} total {int(g0[-1])} head {g0[:3].tolist()}")
                g1 = m.step(torch.tensor(ids[0], device="cuda")).clone()
                say(f"graph-second table hash {table_hash(g1)}")
                m.reset()
                g0b = m.step(start).clone()
                say(f"after reset, int arg: hash {table_hash(g0b)}")
                ref = model_class()(f"/int/{bundle}", 8, "cuda")
                ref.tr, m.tr = [], []
                ref.step(torch.tensor(start, device="cuda"))
                m.reset()
                with torch.no_grad():
                    m.forward_eager(m.BUCKET)
                for (k1, a), (k2, b) in zip(ref.tr, m.tr):
                    a, b = a.reshape(-1), b.reshape(-1).to(a.dtype)
                    same = a.shape == b.shape and bool((a == b).all())
                    nd = int((a != b).sum()) if a.shape == b.shape else -1
                    say(f"{k1}/{k2} shape {tuple(a.shape)} {'SAME' if same else f'DIFF in {nd}'} ref {a[:4].tolist()} fast {b[:4].tolist()}")
                c_ref = ref.step  # (already stepped once) -> compare tables
                ref2 = model_class()(f"/int/{bundle}", 8, "cuda")
                t_ref = ref2.step(torch.tensor(start, device="cuda")).clone()
                t_eager = m.cum.clone()
                m.tr = None
                m.reset()
                t_graph = m.step(start).clone()
                for nm, t in (("eager", t_eager), ("graph", t_graph)):
                    nd = int((t != t_ref).sum())
                    i = int((t != t_ref).nonzero()[0]) if nd else -1
                    say(f"table {nm}: {nd} differ; first at {i}: ref {t_ref[max(i,0):max(i,0)+3].tolist()} got {t[max(i,0):max(i,0)+3].tolist()}; totals {int(t_ref[-1])} {int(t[-1])}")
            if task in ("encode", "roundtrip"):
                ids = ids[:n] if n else ids
                t0 = time.time()
                res["archive"] = m.encode(ids) if fast else run_encode(m, start, ids)
                dt = time.time() - t0
                say(f"{time.strftime('%T')} encode: {len(ids)} tokens -> {len(res['archive'])} bytes, {dt:.1f}s ({len(ids) / dt:.0f} tok/s)")
                archive = res["archive"]
                if not fast:
                    m.pos = 0
                    m.K.zero_(); m.Vc.zero_()
            if task in ("decode", "roundtrip"):
                t0 = time.time()
                out = m.decode(archive) if fast else run_decode(m, start, archive)
                dt = time.time() - t0
                res["text"] = hf.decode(out, clean_up_tokenization_spaces=False)
                say(f"{time.strftime('%T')} decode: {len(out)} tokens, {dt:.1f}s ({len(out) / dt:.0f} tok/s)"
                    + (f", round trip {'OK' if out == ids else 'FAILED'}" if task == "roundtrip" else ""))
            return res
        except Exception:
            raise RuntimeError(traceback.format_exc())

    @app.function(gpu="H100", **kw)
    def run_h100(*a):
        return body(*a)

    @app.function(gpu="H200", **kw)
    def run_h200(*a):
        return body(*a)

    @app.local_entrypoint()
    def main(gpu: str = "H100", task: str = "roundtrip", bundle: str = "coder0.5b.npz", repo: str = "Qwen/Qwen2.5-Coder-0.5B",
             file: str = "bjj-case/3-fully-minified.html", n: int = 0, archive: str = "", out: str = "", fast: bool = False):
        fn = {"H100": run_h100, "H200": run_h200}[gpu]
        text = Path(file).read_bytes().decode("utf-8") if task != "decode" else None
        arch = Path(archive).read_bytes() if task == "decode" else None
        res = fn.remote(task, bundle, repo, text, n, arch, fast)
        if out and "archive" in res:
            Path(out).write_bytes(res["archive"])
            print(f"wrote {out}")
        if out and "text" in res and task == "decode":
            Path(out).write_bytes(res["text"].encode("utf-8"))
            print(f"wrote {out}")
        with open(f"llm/int-{gpu}.log", "a") as f:
            f.write(f"--- {gpu} {task}{' fast' if fast else ''} {time.strftime('%F %T')}\n" + "\n".join(res["log"]) + "\n")


if __name__ == "__main__" and sys.argv[1:2] == ["hashes"]:
    import torch
    from transformers import AutoTokenizer
    bundle, src, path, n = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    dev = sys.argv[6] if len(sys.argv) > 6 else "cpu"
    hf = AutoTokenizer.from_pretrained(src)
    ids = encode_text(hf, Path(path).read_bytes().decode("utf-8"))[:n]
    m = model_class()(bundle, n + 1, dev)
    run_hashes(m, m.start, ids, lambda s: print(s, flush=True))
