"""Real LLM compressor, one stream, full context, built for speed without changing what it codes.

    .venv/bin/python llm/fastcodec.py encode <model> <input> <archive.13k>
    .venv/bin/python llm/fastcodec.py decode <model> <archive.13k> <output>

Same scheme as realcodec.py (one token per forward pass with a KV cache, Witten-Neal-Cleary arithmetic coding), but the
integer frequency table is built on the GPU (float32 softmax, floor, int32 cumsum) and only what the coder needs comes
back: three numbers per token when encoding, the cumulative row when decoding. Encoder and decoder run the identical
GPU ops on the same shapes, so on the same machine they get identical tables.

Archive: 4-byte token count, then the arithmetic-coded stream.
"""
import struct, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from segcodec import Encoder, Decoder, encode_text, TOTAL


def table(model, cache, tok):
    logits = model(mx.array([[tok]]), cache=cache)[0, -1].astype(mx.float32)
    p = mx.softmax(logits)
    q = mx.floor(p * (TOTAL - p.shape[0])).astype(mx.int32) + 1
    return mx.cumsum(q)


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
        enc, prev = Encoder(), start
        for t in ids:
            cum = table(model, cache, prev)
            tot, hi, lo = cum[-1], cum[t], cum[t - 1] if t else mx.array(0)
            mx.eval(tot, hi, lo)
            enc.code_counts(int(tot.item()), int(hi.item()), int(lo.item()))
            prev = t
        body = enc.finish()
        Path(dst).write_bytes(struct.pack("<I", len(ids)) + body)
        dt = time.time() - t0
        print(f"encode {repo}: {len(text.encode())} -> {4 + len(body)} bytes, {len(ids)} tokens, {dt:.1f}s "
              f"({len(ids) / dt:.1f} tok/s), load {t_load:.1f}s", flush=True)
    else:
        data = Path(src).read_bytes()
        (n,) = struct.unpack("<I", data[:4])
        dec, prev, ids = Decoder(data[4:]), start, []
        for _ in range(n):
            cum = np.array(table(model, cache, prev))
            prev = dec.code(cum)
            ids.append(prev)
        text = hf.decode(ids, clean_up_tokenization_spaces=False)
        Path(dst).write_bytes(text.encode("utf-8"))
        dt = time.time() - t0
        print(f"decode {repo}: {len(data)} -> {len(text.encode())} bytes, {n} tokens, {dt:.1f}s "
              f"({n / dt:.1f} tok/s), load {t_load:.1f}s", flush=True)
