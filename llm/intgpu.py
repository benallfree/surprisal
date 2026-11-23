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
            self.head = mat("head") if "head.q" in z else self.emb
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

        @staticmethod
        def resc(x, f):  # (x * f + 2^14) >> 15, exact (see intref.resc)
            return (x >> 32) * f * (1 << 17) + (((x & 0xFFFFFFFF) * f + (1 << 14)) >> 15)

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
            qs = q.view(self.nkv, grp, hd) * self.SC  # score scale folded into q (see intref)
            sh = (self.bitlen(qs.abs().amax(-1)) - 15).clamp(min=0)  # [nkv, grp]
            mq = self.rshift(qs, sh[..., None]).clamp(-32767, 32767)
            K, Vv = self.K[l, :, :T], self.Vc[l, :, :T]  # [nkv, T, hd] float64 holding ints
            acc = torch.matmul(K, mq.to(torch.float64).transpose(1, 2)).to(L64)  # [nkv, T, grp]
            s = self.rshift(acc, (42 - sh)[:, None, :])  # [nkv, T, grp]
            for c in range(0, T, 64):  # one pass over fixed 64-key chunks (see intref)
                sc = s[:, c : c + 64]
                mc = sc.amax(1)
                if c == 0:
                    m, Z, O = mc, torch.zeros_like(mc), torch.zeros(self.nkv, grp, hd, dtype=L64, device=self.dev)
                else:
                    mn = torch.maximum(m, mc)
                    f = self.EXPA[(mn - m).clamp(max=NA - 1)]
                    Z, O, m = self.resc(Z, f), self.resc(O, f[..., None]), mn
                w = self.EXPA[(m[:, None, :] - sc).clamp(max=NA - 1)]
                Z = Z + w.sum(1)
                O = O + torch.matmul(w.transpose(1, 2).to(torch.float64), Vv[:, c : c + 64]).to(L64)
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
            lg = self.matvec(self.head, self.norm(h, self.gf), -10)
            f = self.EXPF[(lg.max() - lg).clamp(max=NF - 1)]
            Z = f.sum()
            self.pos += 1
            self.cum_last = torch.cumsum(f * (TOTAL - self.V) // Z + 1, 0)
            return self.cum_last

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
              DIG: tl.constexpr = False, ZB: tl.constexpr = 2048, SCQ: tl.constexpr = 1):
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
        q1 = _rshc(x1 * c - x2 * s, 14) * SCQ  # score scale folded into q before its blockf
        q2 = _rshc(x2 * c + x1 * s, 14) * SCQ
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
                tl.debug_barrier()
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
            acc = _dot9(k0, k1, k2, q0, q1, q2)  # [CH, GP]; |q.K| < 2^44 (the scale is folded into q)
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

    # ---- batched prefill (compression knows every token, so all positions run as matrix-matrix work)
    @triton.jit
    def _store_dig(ptr, PS, idx, x, mask):
        a0, a1, a2 = _dig(x)
        tl.store(ptr + idx, a0, mask=mask)
        tl.store(ptr + PS + idx, a1, mask=mask)
        tl.store(ptr + 2 * PS + idx, a2, mask=mask)

    @triton.jit
    def _store_dig2(ptr, PS, idx, x, mask):  # |x| <= 32767 -> two offset bytes h, l with x = 256 h + l + 128
        xs = x + 32768
        tl.store(ptr + idx, ((xs >> 8) - 128).to(tl.int8), mask=mask)
        tl.store(ptr + PS + idx, ((xs & 255) - 128).to(tl.int8), mask=mask)

    @triton.jit
    def k_gemm(W_ptr, S_ptr, WS_ptr, M_ptr, out_ptr, R, O, PS, K: tl.constexpr, BO: tl.constexpr, BR: tl.constexpr):
        """out[r, o] = sum_g s[o,g] * sum_{k in g} q[o,k] m[r,k]; m given as three int8 digit planes (plane stride PS).
        int8 dots with int32 sums (< 2^20), digits recombined and scaled in int64: exact."""
        NG: tl.constexpr = K // 64
        o = tl.program_id(0) * BO + tl.arange(0, BO)
        r = tl.program_id(1) * BR + tl.arange(0, BR)
        om, rm = o < O, r < R
        kk = tl.arange(0, 64)
        # float64 accumulation is exact: each group term |d * s| < 2^43 and at most 296 groups, so sums < 2^52
        acc = tl.zeros([BO, BR], dtype=tl.float64)
        for g in range(NG):
            w = tl.load(W_ptr + o[:, None] * K + g * 64 + kk[None, :], mask=om[:, None], other=0)
            ma = M_ptr + r[:, None] * K + g * 64 + kk[None, :]
            bh = tl.trans(tl.load(ma, mask=rm[:, None], other=0))
            bl = tl.trans(tl.load(ma + PS, mask=rm[:, None], other=0))
            ws = tl.load(WS_ptr + o * NG + g, mask=om, other=0)  # sum of the group's int8 weights
            d = (tl.dot(w, bh) << 8) + tl.dot(w, bl) + 128 * ws[:, None]  # int32, exact: |d| < 2^29
            sc = tl.load(S_ptr + o * NG + g, mask=om, other=0).to(tl.float64)
            acc += d.to(tl.float64) * sc[:, None]
        tl.store(out_ptr + r[None, :] * O + o[:, None], acc.to(tl.int64), mask=om[:, None] & rm[None, :])

    @triton.jit
    def k_rnorm(h_ptr, acc_ptr, E_ptr, ein_ptr, g_ptr, EPS_ptr, M_ptr, e_ptr, PS, D: tl.constexpr, BLOCK: tl.constexpr,
                ADD: tl.constexpr):
        """per row: optionally h += rshift(acc, to -24); rmsnorm(h, g) -> blockf -> digit planes, exponent."""
        row = tl.program_id(0)
        i = tl.arange(0, BLOCK)
        mk = i < D
        h = tl.load(h_ptr + row * D + i, mask=mk, other=0)
        if ADD:
            n = -24 - (tl.load(E_ptr) + tl.load(ein_ptr + row))
            h = h + _rsh(tl.load(acc_ptr + row * D + i, mask=mk, other=0), n)
            tl.store(h_ptr + row * D + i, h, mask=mk)
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
        qt = (aa.to(tl.float64) / r.to(tl.float64)).to(tl.int64)
        qt = qt - (qt * r > aa).to(tl.int64)
        qt = qt + ((qt + 1) * r <= aa).to(tl.int64)
        qt = qt - (qt * r > aa).to(tl.int64)
        qt = qt + ((qt + 1) * r <= aa).to(tl.int64)
        y = tl.where(a < 0, -qt, qt)
        sh2 = tl.maximum(_bitlen(tl.max(tl.abs(y), axis=0)) - 15, 0)
        _store_dig2(M_ptr, PS, row * D + i, tl.minimum(tl.maximum(_rsh(y, sh2), -32767), 32767), mk)
        tl.store(e_ptr + row, sh2 - 30)

    @triton.jit
    def k_rqkv(acc_ptr, E_ptr, ein_ptr, b_ptr, C_ptr, S_ptr, pos0, K_ptr, V_ptr, maxlen, Q_ptr, sh_ptr, QS,
               NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, QP: tl.constexpr, SCQ: tl.constexpr):
        """per row (position pos0 + row): bias, rope, q blockf per head -> q digit planes; K, V digits into the cache."""
        row = tl.program_id(0)
        O: tl.constexpr = (NQ + 2 * NKV) * HD
        HH: tl.constexpr = HD // 2
        a_ptr = acc_ptr + row * O
        n = -24 - (tl.load(E_ptr) + tl.load(ein_ptr + row))
        pos = pos0 + row
        d2 = tl.arange(0, HH)
        c = tl.load(C_ptr + pos * HH + d2)[None, :]
        s = tl.load(S_ptr + pos * HH + d2)[None, :]
        hq = tl.arange(0, QP)
        mq = (hq < NQ)[:, None]
        i1 = hq[:, None] * HD + d2[None, :]
        x1 = _rsh(tl.load(a_ptr + i1, mask=mq, other=0), n) + tl.load(b_ptr + i1, mask=mq, other=0)
        x2 = _rsh(tl.load(a_ptr + i1 + HH, mask=mq, other=0), n) + tl.load(b_ptr + i1 + HH, mask=mq, other=0)
        q1 = _rshc(x1 * c - x2 * s, 14) * SCQ  # score scale folded into q before its blockf
        q2 = _rshc(x2 * c + x1 * s, 14) * SCQ
        sh = tl.maximum(_bitlen(tl.maximum(tl.max(tl.abs(q1), axis=1), tl.max(tl.abs(q2), axis=1))) - 15, 0)
        qb = row * NQ * HD
        _store_dig(Q_ptr, QS, qb + i1, tl.minimum(tl.maximum(_rsh(q1, sh[:, None]), -32767), 32767), mq)
        _store_dig(Q_ptr, QS, qb + i1 + HH, tl.minimum(tl.maximum(_rsh(q2, sh[:, None]), -32767), 32767), mq)
        tl.store(sh_ptr + row * NQ + hq, sh, mask=hq < NQ)
        hk = tl.arange(0, NKV)
        LIM: tl.constexpr = 2**22 - 1
        PS = NKV * maxlen * HD
        k1i = NQ * HD + hk[:, None] * HD + d2[None, :]
        x1 = _rsh(tl.load(a_ptr + k1i), n) + tl.load(b_ptr + k1i)
        x2 = _rsh(tl.load(a_ptr + k1i + HH), n) + tl.load(b_ptr + k1i + HH)
        k1 = tl.minimum(tl.maximum(_rshc(_rshc(x1 * c - x2 * s, 14), 12), -LIM), LIM)
        k2 = tl.minimum(tl.maximum(_rshc(_rshc(x2 * c + x1 * s, 14), 12), -LIM), LIM)
        dst = hk[:, None] * maxlen * HD + pos * HD + d2[None, :]
        _store_dig(K_ptr, PS, dst, k1, dst >= 0)
        _store_dig(K_ptr, PS, dst + HH, k2, dst >= 0)
        d = tl.arange(0, HD)
        vi = (NQ + NKV) * HD + hk[:, None] * HD + d[None, :]
        v = tl.minimum(tl.maximum(_rshc(_rsh(tl.load(a_ptr + vi), n) + tl.load(b_ptr + vi), 12), -LIM), LIM)
        vd = hk[:, None] * maxlen * HD + pos * HD + d[None, :]
        _store_dig(V_ptr, PS, vd, v, vd >= 0)

    @triton.jit
    def _split16(x, off: tl.constexpr):
        """x + off in [0, 65535] -> two int8 digits h, l with x = 256 h + l + (32896 - off)."""
        xs = x + off
        return ((xs >> 8) - 128).to(tl.int8), ((xs & 255) - 128).to(tl.int8)

    @triton.jit
    def _dot6(ah, al, b0, b1, b2):  # (256 ah + al) @ (b0 + b1 2^8 + b2 2^16), exact (int8 dots, int32 sums < 2^21)
        hi = tl.dot(ah, b0).to(tl.int64) + (tl.dot(ah, b1).to(tl.int64) << 8) + (tl.dot(ah, b2).to(tl.int64) << 16)
        lo = tl.dot(al, b0).to(tl.int64) + (tl.dot(al, b1).to(tl.int64) << 8) + (tl.dot(al, b2).to(tl.int64) << 16)
        return (hi << 8) + lo

    @triton.jit
    def _sum3(b0, b1, b2, axis: tl.constexpr):  # sums of a digit-split tile along an axis, as int64
        return (tl.sum(b0.to(tl.int32), axis=axis).to(tl.int64) + (tl.sum(b1.to(tl.int32), axis=axis).to(tl.int64) << 8)
                + (tl.sum(b2.to(tl.int32), axis=axis).to(tl.int64) << 16))

    @triton.jit
    def _dot9f(a0, a1, a2, b0, b1, b2):  # exact digit-split product as float64 (every term and partial sum < 2^53)
        # int32 partial combines stay below 2^31 (each dot < 2^21 for 128-term sums of 8-bit digits)
        lo = tl.dot(a0, b0) + ((tl.dot(a0, b1) + tl.dot(a1, b0)) << 8)
        hi = tl.dot(a0, b2) + tl.dot(a1, b1) + tl.dot(a2, b0) + ((tl.dot(a1, b2) + tl.dot(a2, b1)) << 8)
        return lo.to(tl.float64) + hi.to(tl.float64) * 65536.0 + tl.dot(a2, b2).to(tl.float64) * 4294967296.0

    @triton.jit
    def k_rattn3(Q_ptr, sh_ptr, QS, K_ptr, V_ptr, E_ptr, out_ptr, pos0, R, maxlen, NQ: tl.constexpr, NKV: tl.constexpr,
                 GRP: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, CH: tl.constexpr, NA: tl.constexpr,
                 ZTH: tl.constexpr):
        """causal attention for BM query rows of one q head, in float64 on exact integers. The digit products give
        q.K exactly (< 2^44); the score rshift(q.K, n) = floor(q.K * 2^-n + 1/2) is exact in float64 (a power-of-two
        scaling and one exactly representable addition); weights come from the exp table (an int32 copy); w and
        w*V sums stay below 2^53, so they accumulate exactly in float64. Pass 1 finds each row's max."""
        rb, h = tl.program_id(0), tl.program_id(1)
        j = h // GRP
        r = rb * BM + tl.arange(0, BM)
        rm = r < R
        pr = pos0 + r
        d = tl.arange(0, HD)
        qa = Q_ptr + (r[:, None] * NQ + h) * HD + d[None, :]
        q0 = tl.load(qa, mask=rm[:, None], other=0)
        q1 = tl.load(qa + QS, mask=rm[:, None], other=0)
        q2 = tl.load(qa + 2 * QS, mask=rm[:, None], other=0)
        n = 42 - tl.load(sh_ptr + r * NQ + h, mask=rm, other=0)
        inv = tl.exp2(-n.to(tl.float64))[:, None]  # 2^-n, exact
        last = pos0 + tl.minimum(rb * BM + BM, R) - 1
        PS = NKV * maxlen * HD
        kb = j * maxlen * HD
        Mf = tl.full([BM], -1e300, tl.float64)
        for c in range(0, last // CH + 1):
            t = c * CH + tl.arange(0, CH)
            ka = K_ptr + kb + t[:, None] * HD + d[None, :]
            km = (t <= last)[:, None]
            k0 = tl.trans(tl.load(ka, mask=km, other=0))
            k1 = tl.trans(tl.load(ka + PS, mask=km, other=0))
            k2 = tl.trans(tl.load(ka + 2 * PS, mask=km, other=0))
            accf = _dot9f(q0, q1, q2, k0, k1, k2)
            Mf = tl.maximum(Mf, tl.max(tl.where(t[None, :] <= pr[:, None], accf, -1e300), axis=1))
        M = tl.floor(Mf[:, None] * inv + 0.5)  # the row's max score (exact)
        Of = tl.zeros([BM, HD], dtype=tl.float64)
        Zf = tl.zeros([BM], dtype=tl.float64)
        for c in range(0, last // CH + 1):
            t = c * CH + tl.arange(0, CH)
            ka = K_ptr + kb + t[:, None] * HD + d[None, :]
            km = (t <= last)[:, None]
            k0 = tl.trans(tl.load(ka, mask=km, other=0))
            k1 = tl.trans(tl.load(ka + PS, mask=km, other=0))
            k2 = tl.trans(tl.load(ka + 2 * PS, mask=km, other=0))
            sc_ = tl.floor(_dot9f(q0, q1, q2, k0, k1, k2) * inv + 0.5)
            dd = tl.where((t[None, :] <= pr[:, None]) & rm[:, None], M - sc_, NA - 1.0)
            if tl.min(tl.min(dd, axis=1), axis=0) < ZTH:  # else every weight in the tile is EXPA[>= ZTH] = 0
                w = tl.load(E_ptr + tl.minimum(dd, NA - 1.0).to(tl.int32))  # [BM, CH] int32
                Zf += tl.sum(w.to(tl.float64), axis=1)
                w0 = (((w + 128) & 255) - 128)
                w1r = (w - w0) >> 8
                w1 = ((w1r + 128) & 255) - 128
                w2 = (w1r - w1) >> 8
                va = V_ptr + kb + t[:, None] * HD + d[None, :]
                v0 = tl.load(va, mask=km, other=0)
                v1 = tl.load(va + PS, mask=km, other=0)
                v2 = tl.load(va + 2 * PS, mask=km, other=0)
                Of += _dot9f(w0.to(tl.int8), w1.to(tl.int8), w2.to(tl.int8), v0, v1, v2)
        Oa = Of.to(tl.int64)
        Zb = Zf.to(tl.int64)[:, None]
        a = tl.abs(Oa)
        hi = a // Zb
        o = (hi << 12) + ((a - hi * Zb) << 12) // Zb
        o = tl.where(Oa < 0, -o, o)
        tl.store(out_ptr + r[:, None] * (NQ * HD) + h * HD + d[None, :], o, mask=rm[:, None])

    @triton.jit
    def k_rblock(x_ptr, M_ptr, e_ptr, PS, ein, N: tl.constexpr, BLOCK: tl.constexpr):
        """per row: blockf -> digit planes and exponent ein + sh."""
        row = tl.program_id(0)
        i = tl.arange(0, BLOCK)
        mk = i < N
        x = tl.load(x_ptr + row * N + i, mask=mk, other=0)
        sh = tl.maximum(_bitlen(tl.max(tl.abs(x), axis=0)) - 15, 0)
        _store_dig2(M_ptr, PS, row * N + i, tl.minimum(tl.maximum(_rsh(x, sh), -32767), 32767), mk)
        tl.store(e_ptr + row, sh + ein)

    @triton.jit
    def k_rmlp(acc_ptr, E_ptr, ein_ptr, SIG_ptr, p_ptr, M_ptr, e_ptr, PS, I: tl.constexpr, NS: tl.constexpr,
               BLOCK: tl.constexpr):
        """per row: gate/up -> silu via table -> p = a*u, then blockf over I (two passes)."""
        row = tl.program_id(0)
        n = -16 - (tl.load(E_ptr) + tl.load(ein_ptr + row))
        mxv = tl.zeros([BLOCK], dtype=tl.int64)
        for b0 in range(0, I, BLOCK):
            i = b0 + tl.arange(0, BLOCK)
            mk = i < I
            g = _rsh(tl.load(acc_ptr + row * 2 * I + i, mask=mk, other=0), n)
            u = _rsh(tl.load(acc_ptr + row * 2 * I + I + i, mask=mk, other=0), n)
            idx = tl.minimum(tl.maximum(_rshc(g, 6), -NS), NS) + NS
            a = _rshc(g * tl.load(SIG_ptr + idx, mask=mk, other=0), 16)
            p = tl.where(mk, _rshc(a * u, 16), 0)
            tl.store(p_ptr + row * I + i, p, mask=mk)
            mxv = tl.maximum(mxv, tl.abs(p))
        sh = tl.maximum(_bitlen(tl.max(mxv, axis=0)) - 15, 0)
        for b0 in range(0, I, BLOCK):
            i = b0 + tl.arange(0, BLOCK)
            mk = i < I
            p = tl.load(p_ptr + row * I + i, mask=mk, other=0)
            _store_dig2(M_ptr, PS, row * I + i, tl.minimum(tl.maximum(_rsh(p, sh), -32767), 32767), mk)
        tl.store(e_ptr + row, sh - 16)

    @triton.jit
    def k_rhead(W_ptr, S_ptr, WS_ptr, M_ptr, PS, E_ptr, ein_ptr, tgt_ptr, Mx_ptr, Z_ptr, tot_ptr, cumt_ptr, ft_ptr, EXPF_ptr,
                R, V, TOT: tl.constexpr, NF: tl.constexpr, K: tl.constexpr, BO: tl.constexpr, BR: tl.constexpr,
                MODE: tl.constexpr):
        """output head tile (vocab x rows) recomputed in each of three passes, so logits are never stored:
        0: max per row; 1: Z = sum EXPF[M - l]; 2: total = sum freq, cum[t] = sum of freq up to t, freq at t."""
        NG: tl.constexpr = K // 64
        o = tl.program_id(0) * BO + tl.arange(0, BO)
        r = tl.program_id(1) * BR + tl.arange(0, BR)
        om, rm = o < V, r < R
        kk = tl.arange(0, 64)
        accf = tl.zeros([BO, BR], dtype=tl.float64)
        for g in range(NG):
            w = tl.load(W_ptr + o[:, None] * K + g * 64 + kk[None, :], mask=om[:, None], other=0)
            ma = M_ptr + r[:, None] * K + g * 64 + kk[None, :]
            bh = tl.trans(tl.load(ma, mask=rm[:, None], other=0))
            bl = tl.trans(tl.load(ma + PS, mask=rm[:, None], other=0))
            ws = tl.load(WS_ptr + o * NG + g, mask=om, other=0)
            dd = (tl.dot(w, bh) << 8) + tl.dot(w, bl) + 128 * ws[:, None]
            accf += dd.to(tl.float64) * tl.load(S_ptr + o * NG + g, mask=om, other=0).to(tl.float64)[:, None]  # exact (< 2^52)
        acc = accf.to(tl.int64)
        lg = _rsh(acc, (-10 - (tl.load(E_ptr) + tl.load(ein_ptr + r, mask=rm, other=0)))[None, :])
        valid = om[:, None] & rm[None, :]
        if MODE == 0:
            tl.atomic_max(Mx_ptr + r, tl.max(tl.where(valid, lg, -(2**62)), axis=0), mask=rm, sem="relaxed")
        else:
            mx = tl.load(Mx_ptr + r, mask=rm, other=0)[None, :]
            f = tl.where(valid, tl.load(EXPF_ptr + tl.minimum(tl.maximum(mx - lg, 0), NF - 1), mask=valid, other=0), 0)
            if MODE == 1:
                tl.atomic_add(Z_ptr + r, tl.sum(f, axis=0).to(tl.uint64, bitcast=True), mask=rm, sem="relaxed")
            else:
                Zr = tl.load(Z_ptr + r, mask=rm, other=1).to(tl.int64, bitcast=True)[None, :]
                fq = tl.where(valid, f * (TOT - V) // Zr + 1, 0)
                t = tl.load(tgt_ptr + r, mask=rm, other=0)[None, :]
                tl.atomic_add(tot_ptr + r, tl.sum(fq, axis=0).to(tl.uint64, bitcast=True), mask=rm, sem="relaxed")
                tl.atomic_add(cumt_ptr + r, tl.sum(tl.where(o[:, None] <= t, fq, 0), axis=0).to(tl.uint64, bitcast=True),
                              mask=rm, sem="relaxed")
                tl.atomic_add(ft_ptr + r, tl.sum(tl.where(o[:, None] == t, fq, 0), axis=0).to(tl.uint64, bitcast=True),
                              mask=rm, sem="relaxed")

    @triton.jit
    def _resc(x, f):  # (x * f + 2^14) >> 15 for |x| < 2^52, 0 <= f <= 2^15, exact in int64
        return (x >> 32) * f * 131072 + (((x & 0xFFFFFFFF) * f + 16384) >> 15)

    @triton.jit
    def _maxop(a, b):
        return tl.maximum(a, b)

    @triton.jit
    def k_rattn4(Q_ptr, sh_ptr, QS, K_ptr, V_ptr, E_ptr, out_ptr, pos0, R, maxlen, NQ: tl.constexpr, NKV: tl.constexpr,
                 GRP: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, NA: tl.constexpr, ZTH: tl.constexpr):
        """one-pass causal attention (the spec's fixed 64-key chunks, ascending, integer online rescaling) for BM
        query rows of one q head. Scores are exact in float64 (see k_rattn3); Z and O are int64."""
        CH: tl.constexpr = 64
        rb, h = tl.program_id(0), tl.program_id(1)
        j = h // GRP
        r = rb * BM + tl.arange(0, BM)
        rm = r < R
        pr = pos0 + r
        d = tl.arange(0, HD)
        qa = Q_ptr + (r[:, None] * NQ + h) * HD + d[None, :]
        q0 = tl.load(qa, mask=rm[:, None], other=0)
        q1 = tl.load(qa + QS, mask=rm[:, None], other=0)
        q2 = tl.load(qa + 2 * QS, mask=rm[:, None], other=0)
        n = 42 - tl.load(sh_ptr + r * NQ + h, mask=rm, other=0)
        inv = tl.exp2(-n.to(tl.float64))[:, None]
        last = pos0 + tl.minimum(rb * BM + BM, R) - 1
        PS = NKV * maxlen * HD
        kb = j * maxlen * HD
        m = tl.full([BM], -1e300, tl.float64)
        Z = tl.zeros([BM], dtype=tl.int64)
        O = tl.zeros([BM, HD], dtype=tl.int64)
        for c in range(0, last // CH + 1):
            t = c * CH + tl.arange(0, CH)
            ka = K_ptr + kb + t[:, None] * HD + d[None, :]
            km = (t <= last)[:, None]
            k0 = tl.trans(tl.load(ka, mask=km, other=0))
            k1 = tl.trans(tl.load(ka + PS, mask=km, other=0))
            k2 = tl.trans(tl.load(ka + 2 * PS, mask=km, other=0))
            sc_ = tl.floor(_dot9f(q0, q1, q2, k0, k1, k2) * inv + 0.5)
            valid = (t[None, :] <= pr[:, None]) & rm[:, None]
            mn = tl.maximum(m, tl.max(tl.where(valid, sc_, -1e300), axis=1))
            need = (mn > m) & (m > -1e299)
            if tl.max(need.to(tl.int32), axis=0) > 0:
                f = tl.where(need, tl.load(E_ptr + tl.minimum(mn - m, NA - 1.0).to(tl.int32)), 32768).to(tl.int64)
                Z = _resc(Z, f)
                O = _resc(O, f[:, None])
            m = mn
            dd = tl.where(valid, m[:, None] - sc_, NA - 1.0)
            if tl.min(tl.min(dd, axis=1), axis=0) < ZTH:
                w = tl.load(E_ptr + tl.minimum(dd, NA - 1.0).to(tl.int32))  # [BM, CH] int32
                Z += tl.sum(w, axis=1).to(tl.int64)
                w0 = (((w + 128) & 255) - 128)
                w1r = (w - w0) >> 8
                w1 = ((w1r + 128) & 255) - 128
                w2 = (w1r - w1) >> 8
                va = V_ptr + kb + t[:, None] * HD + d[None, :]
                v0 = tl.load(va, mask=km, other=0)
                v1 = tl.load(va + PS, mask=km, other=0)
                v2 = tl.load(va + 2 * PS, mask=km, other=0)
                O += _dot9f(w0.to(tl.int8), w1.to(tl.int8), w2.to(tl.int8), v0, v1, v2).to(tl.int64)
        Zb = Z[:, None]
        a = tl.abs(O)
        hi = a // Zb
        o = (hi << 12) + ((a - hi * Zb) << 12) // Zb
        o = tl.where(O < 0, -o, o)
        tl.store(out_ptr + r[:, None] * (NQ * HD) + h * HD + d[None, :], o, mask=rm[:, None])

    @triton.jit
    def k_seg(Cm_ptr, pm_ptr, seg_ptr, Ms_ptr, nch, NCH, MC: tl.constexpr):
        """per q head: running max of the chunk maxima (pm), run ids (seg: +1 whenever the running max rises) and
        each run's max (Ms)."""
        h = tl.program_id(0)
        c = tl.arange(0, MC)
        cv = c < nch
        cm = tl.load(Cm_ptr + h * NCH + c, mask=cv, other=-(2**62))
        pm = tl.associative_scan(cm, 0, _maxop)
        tl.store(pm_ptr + h * NCH + c, pm, mask=cv)
        # running max of the chunks before c: the same scan over the maxima shifted by one (no reload of pm)
        cmp_ = tl.load(Cm_ptr + h * NCH + c - 1, mask=cv & (c >= 1), other=-(2**62))
        prev = tl.associative_scan(cmp_, 0, _maxop)
        inc = ((c >= 1) & (pm > prev)).to(tl.int64)
        sg = tl.cumsum(inc, axis=0)
        tl.store(seg_ptr + h * NCH + c, sg, mask=cv)
        tl.store(Ms_ptr + h * NCH + sg, pm, mask=cv & ((c == 0) | (inc > 0)))

    @triton.jit
    def k_att2s(S_ptr, pm_ptr, seg_ptr, V_ptr, E_ptr, Os_ptr, Zs_ptr, maxlen, CPP, nch, NCH, NA: tl.constexpr,
                NKV: tl.constexpr, GRP: tl.constexpr, CH: tl.constexpr, HD: tl.constexpr, GP: tl.constexpr,
                ZTH: tl.constexpr):
        """per (kv head, CPP chunks): weights relative to the chunk's running max; w and w*V sums added into the
        chunk's run accumulator (integer atomics: runs are summed exactly in any order)."""
        j, c0 = tl.program_id(0), tl.program_id(1) * CPP
        d = tl.arange(0, HD)
        g = tl.arange(0, GP)
        gm = g < GRP
        hrow = j * GRP + g
        PS = NKV * maxlen * HD
        for c in range(CPP):
            cc = c0 + c
            if cc < nch:
                mg = tl.load(pm_ptr + hrow * NCH + cc, mask=gm, other=0)
                sg = tl.load(seg_ptr + hrow * NCH + cc, mask=gm, other=0)
                t = cc * CH + tl.arange(0, CH)
                sv = tl.load(S_ptr + hrow[None, :] * maxlen + t[:, None], mask=gm[None, :], other=-(2**60))
                dd = mg[None, :] - sv
                if tl.min(tl.min(tl.where(gm[None, :], dd, NA), axis=1), axis=0) < ZTH:
                    w = tl.load(E_ptr + tl.minimum(dd, NA - 1))  # [CH, GP]
                    va = V_ptr + j * maxlen * HD + t[:, None] * HD + d[None, :]
                    v0 = tl.load(va)
                    v1 = tl.load(va + PS)
                    v2 = tl.load(va + 2 * PS)
                    w0, w1, w2 = _dig(w)
                    o = _dot9(tl.trans(w0), tl.trans(w1), tl.trans(w2), v0, v1, v2)  # [GP, HD]
                    tl.atomic_add(Zs_ptr + hrow * NCH + sg, tl.sum(w, axis=0).to(tl.uint64, bitcast=True), mask=gm,
                                  sem="relaxed")
                    tl.atomic_add(Os_ptr + (hrow[:, None] * NCH + sg[:, None]) * HD + d[None, :],
                                  o.to(tl.uint64, bitcast=True), mask=gm[:, None], sem="relaxed")

    @triton.jit
    def k_mergeseg(Zs_ptr, Os_ptr, Ms_ptr, seg_ptr, E_ptr, out_ptr, pos_ptr, NCH, NA: tl.constexpr, HD: tl.constexpr):
        """per q head: combine the runs in order (rescale when the max rises), o = O / Z; zero the run sums."""
        h = tl.program_id(0)
        d = tl.arange(0, HD)
        J = tl.load(seg_ptr + h * NCH + tl.load(pos_ptr) // 64) + 1
        Z = tl.load(Zs_ptr + h * NCH)
        O = tl.load(Os_ptr + h * NCH * HD + d)
        mp = tl.load(Ms_ptr + h * NCH)
        for jj in range(1, J):
            mu = tl.load(Ms_ptr + h * NCH + jj)
            f = tl.load(E_ptr + tl.minimum(mu - mp, NA - 1))
            Z = _resc(Z, f) + tl.load(Zs_ptr + h * NCH + jj)
            O = _resc(O, f) + tl.load(Os_ptr + (h * NCH + jj) * HD + d)
            mp = mu
        # zero the run sums only after every thread has read them (values can be held by several threads)
        tl.debug_barrier()
        for jj in range(0, J):
            tl.store(Zs_ptr + h * NCH + jj, tl.zeros([], dtype=tl.int64))
            tl.store(Os_ptr + (h * NCH + jj) * HD + d, tl.zeros([HD], dtype=tl.int64))
        a = tl.abs(O)
        hi = a // Z
        o = (hi << 12) + ((a - hi * Z) << 12) // Z
        tl.store(out_ptr + h * HD + d, tl.where(O < 0, -o, o))

    @triton.jit
    def k_blockf1(x_ptr, m_ptr, e_ptr, ein, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        mk = i < N
        x = tl.load(x_ptr + i, mask=mk, other=0)
        sh = tl.maximum(_bitlen(tl.max(tl.abs(x), axis=0)) - 15, 0)
        tl.store(m_ptr + i, tl.minimum(tl.maximum(_rsh(x, sh), -32767), 32767), mask=mk)
        tl.store(e_ptr, sh + ein)

    class FastInt:
        def __init__(self, bundle, n, log, BUCKET=1024, CH=32, fused=True, APG=192, graphs=True):
            self.tr = None
            z = np.load(bundle)
            cfg = json.loads(bytes(z["cfg"]).decode())
            dev = self.dev = "cuda"
            D, I, Lr = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
            nq, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
            hd, V, grp = D // nq, cfg["vocab_size"], nq // nkv
            self.start, self.V = cfg["bos_token_id"], V
            SC = int(round(2**16 / math.sqrt(hd)))
            QP_ = 1 << (nq - 1).bit_length()  # q heads padded to a power of two
            NB_ = 1 << (D - 1).bit_length()  # norm block (power of two >= D)
            if not fused:
                raise ValueError("the unfused debug path implements the old two-pass attention rule")
            self.BUCKET = BUCKET
            nb = (n + BUCKET) // BUCKET  # positions 0..n-1 need attention length up to n
            maxlen = nb * BUCKET
            Tn = lambda a, dt=L64: torch.from_numpy(np.ascontiguousarray(a)).to(dev, dt)
            sc = lambda v: torch.tensor(v, dtype=L64, device=dev)

            def mat(nm):
                return Tn(z[nm + ".q"], torch.int8), Tn(z[nm + ".s"], torch.int32), sc(int(z[nm + ".E"]))

            emb_q, emb_s, E_emb = mat("emb")
            head_q, head_s, E_head = mat("head") if "head.q" in z else (emb_q, emb_s, E_emb)
            layers = [dict(qkv=mat(f"{l}.qkv"), o=mat(f"{l}.o"), gu=mat(f"{l}.gu"), d=mat(f"{l}.d"),
                           b=Tn(z[f"{l}.bqkv"]), g1=Tn(z[f"{l}.g1"]), g2=Tn(z[f"{l}.g2"])) for l in range(Lr)]
            gf = Tn(z["gf"])
            C, S = Tn(z["ROPE_C"]), Tn(z["ROPE_S"])
            EXPA, EXPF, SIG = Tn(z["EXPA"]), Tn(z["EXPF"]), Tn(z["SIG"])
            EXPA32 = EXPA.to(torch.int32)
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
            NC64 = maxlen // 64
            MC64 = 1 << (NC64 - 1).bit_length()
            PMb, SEGb, MSb = (torch.zeros(nq, NC64, dtype=L64, device=dev) for _ in range(3))
            Zs = torch.zeros(nq, NC64, dtype=L64, device=dev)
            Os = torch.zeros(nq, NC64, hd, dtype=L64, device=dev)
            Zs_u, Os_u = Zs.view(torch.uint64), Os.view(torch.uint64)
            Ob1 = torch.zeros(nq * hd, dtype=L64, device=dev)
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
                                NQ=nq, NKV=nkv, HD=hd, QP=QP_, ZERO=True, DIG=True, ZB=1 << ((nq + 2 * nkv) * hd - 1).bit_length(), SCQ=SC)
                    n64 = Lb // 64  # spec chunks in this bucket
                    cpp = triton.cdiv(n64, APG)
                    npg = triton.cdiv(n64, cpp)
                    k_att1d[(nkv, npg)](mqb, shb, KV8[l, 0], Sbuf, Mg_acc, Cm, pos, maxlen, cpp, n64, NC64, NKV=nkv, GRP=grp,
                                        CH=64, HD=hd, GP=16, SC=SC)
                    k_seg[(nq,)](Cm, PMb, SEGb, MSb, n64, NC64, MC=MC64)
                    k_att2s[(nkv, npg)](Sbuf, PMb, SEGb, KV8[l, 1], EXPA, Os_u, Zs_u, maxlen, cpp, n64, NC64, NA=NA, NKV=nkv,
                                        GRP=grp, CH=64, HD=hd, GP=16, ZTH=ZTH)
                    k_mergeseg[(nq,)](Zs, Os, MSb, SEGb, EXPA, Ob1, pos, NC64, NA=NA, HD=hd)
                    k_blockf1[(1,)](Ob1, mob, e_o, -24, N=nq * hd, BLOCK=1 << (nq * hd - 1).bit_length())
                    if self.tr is not None and l == 0:
                        self.tr.append(("mo", mob.clone())); self.tr.append(("eo", e_o.clone()))
                        self.tr.append(("seg", SEGb[:, :4].clone())); self.tr.append(("pm", PMb[:, :4].clone()))
                    acc, E = mv3(Ly["o"], mob, 32, 1)
                    k_norm2[(1,)](hbuf, acc, E, e_o, pmax, Ly["g2"], EPS, mbuf, e_n, D=D, BLOCK=NB_, ADD=True, EINP=False, num_warps=8)
                    acc, E = mv3(Ly["gu"], mbuf, 64, 1, nw=2)
                    k_mlp2[(triton.cdiv(I, 1024),)](acc, E, e_n, SIG, pbuf, pmax, I=I, NS=NS, BLOCK=1024)
                    acc, E = mv3(Ly["d"], pbuf, 128, 1, PRO=True)
                    k_norm2[(1,)](hbuf, acc, E, e_n, pmax, layers[l + 1]["g1"] if l + 1 < Lr else gf, EPS, mbuf, e_n,
                                  D=D, BLOCK=NB_, ADD=True, EINP=True, num_warps=8)
                k_mvlog[(triton.cdiv(V, 64),)](head_q, head_s, mbuf, lgbuf, E_head, e_n, gmaxb, V, K=D, BO=64)
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
                O_acc.zero_(); Z_acc.zero_(); pmax.zero_(); Zt.zero_(); gmaxb.fill_(-(2**62)); Zs.zero_(); Os.zero_()
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

            if graphs:
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
                f = lambda: k_mergea[(1,)](O_acc, Z_acc, mob, e_o, NQ=nq, HD=hd, QP=QP_)
                say(f"merge: {tm(f):.2f}us")
                f = lambda: k_mlp2[(triton.cdiv(I, 1024),)](outs.get((2 * I, D), torch.zeros(2 * I, dtype=L64, device=dev)), E_emb, e_n,
                                                             SIG, pbuf, pmax, I=I, NS=NS, BLOCK=1024)
                say(f"mlp: {tm(f):.2f}us")
                f = lambda: pmax.add_(0)
                say(f"tiny torch kernel: {tm(f):.2f}us")

            wsums = {}

            def wsum(q):  # per (row, 64-group) sums of the int8 weights, for the offset-byte GEMM inputs
                k = q.data_ptr()
                if k not in wsums:
                    O_, K_ = q.shape
                    wsums[k] = q.view(O_, K_ // 64, 64).to(torch.int32).sum(-1, dtype=torch.int32).contiguous()
                return wsums[k]

            def encode_batched(ids, P=2048, say=None, BM=16, CHa=64, NWa=4, GBO=256, GBR=64, GNW=8, GST=3, HBO=256, HBR=64):
                """Compress with every position as matrix-matrix work: rows of P positions run through all layers,
                attention reads the digit-plane cache. Same spec, so the same counts as the step-by-step path."""
                n = len(ids)
                toks = torch.tensor([self.start] + list(ids[:-1]), dtype=L64, device=dev)
                tg = torch.tensor(list(ids), dtype=L64, device=dev)
                KV8.zero_()
                PSd, PSi, QS = P * D, P * I, P * nq * hd
                Md = torch.empty(2, P, D, dtype=torch.int8, device=dev)  # GEMM inputs: two offset-byte planes
                Mi = torch.empty(2, P, I, dtype=torch.int8, device=dev)
                Mo = torch.empty(2, P, nq * hd, dtype=torch.int8, device=dev)
                Qd = torch.empty(3, P, nq, hd, dtype=torch.int8, device=dev)
                shq = torch.empty(P, nq, dtype=L64, device=dev)
                H = torch.empty(P, D, dtype=L64, device=dev)
                e1, e2 = torch.empty(P, dtype=L64, device=dev), torch.empty(P, dtype=L64, device=dev)
                Oqkv = (nq + 2 * nkv) * hd
                A_qkv = torch.empty(P, Oqkv, dtype=L64, device=dev)
                A_d = torch.empty(P, D, dtype=L64, device=dev)
                A_gu = torch.empty(P, 2 * I, dtype=L64, device=dev)
                Ob = torch.empty(P, nq * hd, dtype=L64, device=dev)
                Pb = torch.empty(P, I, dtype=L64, device=dev)
                Mx, Zr, tot, cumt, ft = (torch.empty(P, dtype=L64, device=dev) for _ in range(5))
                counts = torch.empty(n, 3, dtype=L64, device=dev)
                def gemm(W, Mp, PS, R, out):
                    q, s_, E = W
                    O, K = q.shape
                    k_gemm[(triton.cdiv(O, GBO), triton.cdiv(R, GBR))](q, s_, wsum(q), Mp, out, R, O, PS, K=K, BO=GBO, BR=GBR,
                                                                      num_warps=GNW, num_stages=GST)
                    return E

                for r0 in range(0, n, P):
                    R = min(P, n - r0)
                    tk = toks[r0 : r0 + R]
                    H[:R] = rshift(emb_q[tk].to(L64) * torch.repeat_interleave(emb_s[tk].to(L64), 64, dim=1), -24 - E_emb)
                    k_rnorm[(R,)](H, H, E_emb, e1, layers[0]["g1"], EPS, Md, e1, PSd, D=D, BLOCK=NB_, ADD=False, num_warps=8)
                    for l, Ly in enumerate(layers):
                        E = gemm(Ly["qkv"], Md, PSd, R, A_qkv)
                        k_rqkv[(R,)](A_qkv, E, e1, Ly["b"], C, S, r0, KV8[l, 0], KV8[l, 1], maxlen, Qd, shq, QS,
                                     NQ=nq, NKV=nkv, HD=hd, QP=QP_, SCQ=SC)
                        k_rattn4[(triton.cdiv(R, BM), nq)](Qd, shq, QS, KV8[l, 0], KV8[l, 1], EXPA32, Ob, r0, R, maxlen, NQ=nq,
                                                           NKV=nkv, GRP=grp, HD=hd, BM=BM, NA=NA, ZTH=ZTH, num_warps=NWa)
                        k_rblock[(R,)](Ob, Mo, e2, P * nq * hd, -24, N=nq * hd, BLOCK=1 << (nq * hd - 1).bit_length())
                        E = gemm(Ly["o"], Mo, P * nq * hd, R, A_d)
                        k_rnorm[(R,)](H, A_d, E, e2, Ly["g2"], EPS, Md, e1, PSd, D=D, BLOCK=NB_, ADD=True, num_warps=8)
                        E = gemm(Ly["gu"], Md, PSd, R, A_gu)
                        k_rmlp[(R,)](A_gu, E, e1, SIG, Pb, Mi, e2, PSi, I=I, NS=NS, BLOCK=1024)
                        E = gemm(Ly["d"], Mi, PSi, R, A_d)
                        k_rnorm[(R,)](H, A_d, E, e2, layers[l + 1]["g1"] if l + 1 < Lr else gf, EPS, Md, e1, PSd, D=D,
                                      BLOCK=NB_, ADD=True, num_warps=8)
                    Mx.fill_(-(2**62)); Zr.zero_(); tot.zero_(); cumt.zero_(); ft.zero_()
                    tgr = tg[r0 : r0 + R]
                    for mode in range(3):
                        k_rhead[(triton.cdiv(V, HBO), triton.cdiv(R, HBR))](head_q, head_s, wsum(head_q), Md, PSd, E_head, e1, tgr, Mx,
                                                                           Zr.view(torch.uint64), tot.view(torch.uint64),
                                                                           cumt.view(torch.uint64), ft.view(torch.uint64), EXPF,
                                                                           R, V, TOT=TOTAL, NF=NF, K=D, BO=HBO, BR=HBR,
                                                                           MODE=mode, num_warps=GNW, num_stages=GST)
                    counts[r0 : r0 + R] = torch.stack([tot[:R], cumt[:R], cumt[:R] - ft[:R]], 1)
                    if say:
                        torch.cuda.synchronize()
                        say(f"{time.strftime('%T')}   rows {r0 + R}/{n}")
                cnt = counts.tolist()
                bad = [k for k, (tt, hi, lo) in enumerate(cnt) if not 0 <= lo < hi <= tt]
                if bad:
                    raise ValueError(f"bad counts at token {bad[0]}: {cnt[bad[0]]}")
                enc = Encoder()
                for tt, hi, lo in cnt:
                    enc.code_counts(tt, hi, lo)
                return struct.pack("<I", n) + enc.finish()

            self.encode_batched = encode_batched
            self.dbg = dict(Cm=Cm, PM=PMb, SEG=SEGb, MS=MSb, Zs=Zs, Ob1=Ob1)
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
                kw_ = {"graphs": task != "bencode"}
                if targ and task.startswith("prof"):
                    kw_["APG"] = int(targ)
                m = fast_class()(f"/int/{bundle}", maxlen - 1, say, **kw_)
            else:
                m = model_class()(f"/int/{bundle}", maxlen, "cuda")
            start = m.start
            say(f"{time.strftime('%T')} loaded in {time.time() - t0:.1f}s")
            res = {"log": log}
            if task == "hashes":
                run_hashes(m, start, ids[:n], say)
            if task in ("bencode", "bcheck", "bround"):  # batched compression (bcheck: also step-by-step, compare archives)
                ids_ = ids[:n] if n else ids
                opts = dict(kv.split("=") for kv in targ.split(",") if "=" in kv)
                targ2 = "prof" if "prof" in targ.split(",") else ""
                Pr = int(opts.get("P", 2048))
                ak = dict(BM=int(opts.get("BM", 16)), CHa=int(opts.get("CH", 64)), NWa=int(opts.get("NW", 4)),
                          **{k: int(v) for k, v in opts.items() if k in ("GBO", "GBR", "GNW", "GST", "HBO", "HBR")})
                t0 = time.time()
                m.encode_batched(ids_, P=Pr, **ak)  # first run compiles the kernels (Triton caches them)
                torch.cuda.synchronize()
                say(f"{time.strftime('%T')} first (compiling) batched run: {time.time() - t0:.1f}s")
                torch.cuda.synchronize()
                t0 = time.time()
                arch = m.encode_batched(ids_, P=Pr, **ak)
                torch.cuda.synchronize()
                dt = time.time() - t0
                res["archive"] = arch
                say(f"{time.strftime('%T')} batched encode: {len(ids_)} tokens -> {len(arch)} bytes, {dt:.2f}s ({len(ids_) / dt:.0f} tok/s), P={Pr} {ak}")
                if targ2 == "prof":
                    from torch.profiler import profile, ProfilerActivity
                    with profile(activities=[ProfilerActivity.CUDA]) as pr:
                        m.encode_batched(ids_, P=Pr, **ak)
                        torch.cuda.synchronize()
                    tbl = pr.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=40)
                    print(tbl, flush=True); log.append(tbl)
                if task == "bround":  # decode the batched archive step by step
                    t0 = time.time()
                    out_ = m.decode(arch)
                    dt = time.time() - t0
                    say(f"{time.strftime('%T')} step-by-step decode: {len(out_)} tokens, {dt:.1f}s ({len(out_) / dt:.0f} tok/s), "
                        f"round trip {'OK' if out_ == list(ids_) else 'FAILED'}")
                if task == "bcheck":
                    t0 = time.time()
                    a2 = m.encode(ids_)
                    say(f"{time.strftime('%T')} step-by-step encode: {len(a2)} bytes, {time.time() - t0:.1f}s; "
                        f"archives {'IDENTICAL' if a2 == arch else 'DIFFER'}")
            if task == "attdbg":  # layer-0 attention output at position N: decode path vs plain PyTorch
                P_ = n or 128
                ref = model_class()(f"/int/{bundle}", P_ + 8, "cuda")
                prev = start
                for k in range(P_ + 1):
                    if k == P_:
                        ref.tr = []
                        m.tr = []
                        m.tok.fill_(prev)
                        with torch.no_grad():
                            m.forward_eager(m.BUCKET)
                    else:
                        m.step(torch.tensor(prev, device="cuda"))
                    ref.step(torch.tensor(prev, device="cuda"))
                    prev = ids[k]
                trs = list(m.tr)
                t_eager = m.cum.clone()
                t_ref = ref.cum_last.clone()
                say(f"final table eager {table_hash(t_eager)} ref {table_hash(t_ref)}")
                m.tr = None
                m.reset()
                prev = start
                for k in range(P_ + 1):
                    tg_ = m.step(torch.tensor(prev, device="cuda")).clone()
                    prev = ids[k]
                say(f"final table graph {table_hash(tg_)}")
                rt = dict(ref.tr); ft = dict(trs)
                for k in ("mo", "eo"):
                    a_, b_ = rt[k].reshape(-1), ft[k].reshape(-1)
                    nd = int((a_ != b_).sum())
                    i_ = int((a_ != b_).nonzero()[0]) if nd else -1
                    say(f"{k}: {nd} differ; first {i_}: ref {a_[max(i_,0):max(i_,0)+4].tolist()} fast {b_[max(i_,0):max(i_,0)+4].tolist()}")
                say(f"seg {ft['seg'][:3].tolist()} pm {ft['pm'][:3].tolist()}")
            if task == "racedbg":  # replay one position many times from the same state; count distinct results
                P_ = n or 129
                prev = start
                for k in range(P_):
                    m.step(torch.tensor(prev, device="cuda"))
                    prev = ids[k]
                hs, hob = {}, {}
                for rep in range(30):
                    m.pos.fill_(P_)
                    m.tok.fill_(prev)
                    m.fwd[P_ // m.BUCKET].replay()
                    torch.cuda.synchronize()
                    hs[table_hash(m.cum)] = hs.get(table_hash(m.cum), 0) + 1
                    ob = m.dbg["Ob1"]
                    hob[table_hash(ob)] = hob.get(table_hash(ob), 0) + 1
                say(f"position {P_}: {len(hs)} distinct tables {hs}; last-layer attention outputs {len(hob)} distinct")
            if task == "segdbg":
                prev = start
                for k, t in enumerate(ids[:131]):
                    m.step(torch.tensor(prev, device="cuda"))
                    prev = t
                    if k >= 126:
                        dd = m.dbg
                        say(f"pos {k}: Cm {dd['Cm'][0, :4].tolist()} PM {dd['PM'][0, :4].tolist()} SEG {dd['SEG'][0, :4].tolist()} "
                            f"MS {dd['MS'][0, :4].tolist()} Zs {dd['Zs'][0, :4].tolist()}")
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
                trs = list(m.tr)
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
