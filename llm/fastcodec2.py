"""fastcodec.py made faster without changing a single coded bit: pipelined, with the token search on the GPU.

    .venv/bin/python llm/fastcodec2.py encode <model> <input> <archive.13k>
    .venv/bin/python llm/fastcodec2.py decode <model> <archive.13k> <output>

Archives are byte-identical to fastcodec.py's (same frequency table, same coder). What changed:

- The frequency table (softmax, floor, cumsum) is one compiled MLX function.
- Decoding: the arithmetic decoder's search (target -> token) runs on the GPU in int64, so only four numbers come back
  per token instead of a 600 KB cumulative row. The found token stays on the GPU and feeds straight into the next
  forward pass, so the next step's graph is built and queued before the CPU even reads the token.
- The coder's few numbers are submitted in their own command buffer before the next forward pass (MLX signals an
  array only when its whole command buffer finishes; sharing one cost 1.6 ms per token).
- Encoding: the next forward pass is queued before the CPU codes the current token (it knows every token already).
"""
import struct, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from segcodec import Encoder, Decoder, encode_text, TOTAL, HALF, QTR


@mx.compile
def freq_table(logits):
    p = mx.softmax(logits.astype(mx.float32))
    q = mx.floor(p * (TOTAL - p.shape[0])).astype(mx.int32) + 1
    return mx.cumsum(q)


@mx.compile
def search(cum, st):
    """st = [value, low, range] (int64). Returns [token, cum[token], cum[token-1], total] (int64)."""
    c = cum.astype(mx.int64)
    total = c[-1]
    target = ((st[0] - st[1] + 1) * total - 1) // st[2]
    t = (c <= target).sum().astype(mx.int64)
    hi = mx.take(c, t)
    lo = mx.where(t > 0, mx.take(c, mx.maximum(t - 1, 0)), 0)
    return mx.stack([t, hi, lo, total])


def table(model, cache, tok):
    """tok: int32 array of shape [1, 1]."""
    return freq_table(model(tok, cache=cache)[0, -1])


class Dec(Decoder):
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
    cache = make_prompt_cache(model)
    if mode == "encode":
        text = Path(src).read_bytes().decode("utf-8")
        ids = encode_text(hf, text)
        enc = Encoder()
        cum = table(model, cache, tok_arr(start))
        for k, t in enumerate(ids):
            cnt = mx.stack([cum[-1], cum[t], cum[t - 1] if t else mx.array(0, dtype=mx.int32)])
            mx.async_eval(cnt)  # its own command buffer, so reading it doesn't wait for the next forward pass
            if k + 1 < len(ids):
                cum = table(model, cache, tok_arr(t))
                mx.async_eval(cum)
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
        cum = table(model, cache, tok_arr(start))
        for k in range(n):
            st = mx.array([dec.value, dec.low, dec.high - dec.low + 1], dtype=mx.int64)
            res = search(cum, st)
            mx.async_eval(res)  # its own command buffer, so reading it doesn't wait for the next forward pass
            if k + 1 < n:
                cum = table(model, cache, res[:1].astype(mx.int32)[None])
                mx.async_eval(cum)
            t, hi, lo, tot = res.tolist()
            dec.update(tot, hi, lo)
            ids.append(t)
        text = hf.decode(ids, clean_up_tokenization_spaces=False)
        Path(dst).write_bytes(text.encode("utf-8"))
        dt = time.time() - t0
        print(f"decode {repo}: {len(data)} -> {len(text.encode())} bytes, {n} tokens, {dt:.1f}s "
              f"({n / dt:.1f} tok/s), load {t_load:.1f}s", flush=True)
