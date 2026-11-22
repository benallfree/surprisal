"""fastcodec2.py with the whole decode step compiled: one mx.compile'd function per token (Qwen2 models only).

    .venv/bin/python llm/fastcodec3.py encode <model> <input> <archive.13k>
    .venv/bin/python llm/fastcodec3.py decode <model> <archive.13k> <output>

The forward pass, KV-cache write and frequency table are traced once into a compiled graph, so MLX fuses the small
element-wise kernels (norms, rope, residuals, activations). That needs fixed shapes, so the KV cache is preallocated
in 2K-position buckets; attention reads the whole bucket with a mask, and the step is recompiled when
the cache grows into the next bucket. Encoder and decoder run the identical compiled steps, so they agree bit for bit.
Archives are the same format as fastcodec.py but not byte-identical to it (different kernels).
"""
import struct, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from segcodec import Encoder, encode_text
from fastcodec2 import freq_table, search, Dec

BUCKET = 2048


class Stepper:
    def __init__(self, model):
        self.model = model
        self.n_layers = len(model.model.layers)
        self.size = 0
        self.ks = self.vs = None
        self.fn = mx.compile(self._step)

    def _step(self, tok, pos, ks, vs):
        inner = self.model.model
        h = inner.embed_tokens(tok)
        size = ks[0].shape[2]
        mask = mx.where(mx.arange(size) <= pos, 0.0, -mx.inf).astype(h.dtype)
        nk, nv = [], []
        for layer, K, V in zip(inner.layers, ks, vs):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_proj(x).reshape(1, 1, a.n_heads, -1).transpose(0, 2, 1, 3)
            k = a.k_proj(x).reshape(1, 1, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            v = a.v_proj(x).reshape(1, 1, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q = a.rope(q, offset=pos)
            k = a.rope(k, offset=pos)
            K = mx.slice_update(K, k.astype(K.dtype), pos.reshape(1), axes=(2,))
            V = mx.slice_update(V, v.astype(V.dtype), pos.reshape(1), axes=(2,))
            nk.append(K)
            nv.append(V)
            o = mx.fast.scaled_dot_product_attention(q, K, V, scale=a.scale, mask=mask)
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(1, 1, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = inner.norm(h)
        m = self.model
        logits = m.lm_head(h) if hasattr(m, "lm_head") else inner.embed_tokens.as_linear(h)
        return freq_table(logits[0, 0]), nk, nv

    def __call__(self, tok, pos):
        """tok: int32 [1, 1] array (may be lazy); pos: python int. Returns the lazy cumulative frequency table."""
        if pos >= self.size:
            new = self.size + BUCKET
            a = self.model.model.layers[0].self_attn
            dt = self.model.model.norm.weight.dtype
            hd = a.scale ** -2
            shape = (1, a.n_kv_heads, new - self.size, int(round(hd)))
            pad = [mx.zeros(shape, dtype=dt) for _ in range(self.n_layers)]
            if self.ks is None:
                self.ks, self.vs = pad, [mx.zeros(shape, dtype=dt) for _ in range(self.n_layers)]
            else:
                self.ks = [mx.concatenate([k, p], axis=2) for k, p in zip(self.ks, pad)]
                self.vs = [mx.concatenate([v, p], axis=2) for v, p in zip(self.vs, pad)]
            self.size = new
        cum, self.ks, self.vs = self.fn(tok, mx.array(pos, dtype=mx.int32), self.ks, self.vs)
        return cum


def tok_arr(t):
    return mx.array([[t]], dtype=mx.int32)


if __name__ == "__main__":
    mode, repo, src, dst = sys.argv[1:5]
    t0 = time.time()
    model, tok = load(repo)
    hf = tok._tokenizer
    start = hf.bos_token_id if hf.bos_token_id is not None else hf.eos_token_id
    t_load = time.time() - t0
    t0 = time.time()
    step = Stepper(model)
    if mode == "encode":
        text = Path(src).read_bytes().decode("utf-8")
        ids = encode_text(hf, text)
        enc = Encoder()
        cum = step(tok_arr(start), 0)
        for k, t in enumerate(ids):
            cnt = mx.stack([cum[-1], cum[t], cum[t - 1] if t else mx.array(0, dtype=mx.int32)])
            if k + 1 < len(ids):
                cum = step(tok_arr(t), k + 1)
                mx.async_eval(cnt, cum, step.ks, step.vs)
            tot, hi, lo = cnt.tolist()
            enc.code_counts(tot, hi, lo)
        body = enc.finish()
        Path(dst).write_bytes(struct.pack("<I", len(ids)) + body)
        dt = time.time() - t0
        print(f"encode {repo}: {len(text.encode())} -> {4 + len(body)} bytes, {len(ids)} tokens, {dt:.1f}s "
              f"({len(ids) / dt:.1f} tok/s), load {t_load:.1f}s", flush=True)
    else:
        data = Path(src).read_bytes()
        (n,) = struct.unpack("<I", data[:4])
        dec, ids = Dec(data[4:]), []
        cum = step(tok_arr(start), 0)
        for k in range(n):
            st = mx.array([dec.value, dec.low, dec.high - dec.low + 1], dtype=mx.int64)
            res = search(cum, st)
            if k + 1 < n:
                cum = step(res[:1].astype(mx.int32).reshape(1, 1), k + 1)
                mx.async_eval(res, cum, step.ks, step.vs)
            t, hi, lo, tot = res.tolist()
            dec.update(tot, hi, lo)
            ids.append(t)
        text = hf.decode(ids, clean_up_tokenization_spaces=False)
        Path(dst).write_bytes(text.encode("utf-8"))
        dt = time.time() - t0
        print(f"decode {repo}: {len(data)} -> {len(text.encode())} bytes, {n} tokens, {dt:.1f}s "
              f"({n / dt:.1f} tok/s), load {t_load:.1f}s", flush=True)
