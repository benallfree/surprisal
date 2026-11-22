"""A real LLM compressor: arithmetic coding driven by an MLX model, one token at a time.

    .venv/bin/python llm/realcodec.py encode <hf-repo> <input> <archive.13k>
    .venv/bin/python llm/realcodec.py decode <hf-repo> <archive.13k> <output>

Encoder and decoder run the identical loop: feed one token through the model with a KV cache, turn the
logits into an integer frequency table, code the next token. They must, because arithmetic coding needs
bit-identical probabilities on both ends, and a batched forward pass gives slightly different floats
than single-token steps. So real encoding costs as much as decoding.

Archive: 4-byte token count, then the arithmetic-coded stream (Witten-Neal-Cleary, 32-bit, frequency
total 2^28 with every token given at least 1, which costs a few bytes over the ideal).
"""
import struct, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

PREC = 32
TOP, HALF, QTR = (1 << PREC) - 1, 1 << (PREC - 1), 1 << (PREC - 2)
TOTAL = 1 << 28


def freqs(logits) -> np.ndarray:
    """Cumulative integer frequencies from logits, computed in float64 on the CPU so both ends agree."""
    x = np.asarray(logits.astype(mx.float32), dtype=np.float64)
    p = np.exp(x - x.max())
    p /= p.sum()
    q = np.floor(p * (TOTAL - len(p))).astype(np.int64) + 1
    return np.cumsum(q)


class Model:
    def __init__(self, repo):
        self.model, tok = load(repo)
        self.hf = tok._tokenizer
        self.start = self.hf.bos_token_id if self.hf.bos_token_id is not None else self.hf.eos_token_id
        self.vocab = None

    def encode_text(self, text):
        norm = self.hf.backend_tokenizer.normalizer
        for n in (norm, None):
            self.hf.backend_tokenizer.normalizer = n
            ids = self.hf.encode(text, add_special_tokens=False)
            if self.hf.decode(ids, clean_up_tokenization_spaces=False) == text:
                return ids
        raise ValueError("tokenizer round-trip mismatch")

    def steps(self):
        """Generator: send a token id, get the cumulative frequencies for the next one."""
        cache = make_prompt_cache(self.model)
        tok = self.start
        while True:
            logits = self.model(mx.array([[tok]]), cache=cache)[0, -1]
            tok = yield freqs(logits)


def encode(m, ids):
    out, low, high, pending = [], 0, TOP, 0
    bits = []

    def emit(b):
        nonlocal pending
        bits.append(b)
        bits.extend([1 - b] * pending)
        pending = 0

    g = m.steps()
    cum = next(g)
    for i, t in enumerate(ids):
        total, hi_c, lo_c = int(cum[-1]), int(cum[t]), int(cum[t - 1]) if t else 0
        rng = high - low + 1
        high = low + rng * hi_c // total - 1
        low = low + rng * lo_c // total
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QTR and high < HALF + QTR:
                pending += 1
                low -= QTR
                high -= QTR
            else:
                break
            low, high = 2 * low, 2 * high + 1
        if i + 1 < len(ids):
            cum = g.send(t)
    pending += 1
    emit(0 if low < QTR else 1)
    bits.extend([0] * (-len(bits) % 8))
    return bytes(int("".join(map(str, bits[k : k + 8])), 2) for k in range(0, len(bits), 8))


def decode(m, data, n):
    bits = "".join(f"{b:08b}" for b in data)
    pos = 0

    def bit():
        nonlocal pos
        pos += 1
        return int(bits[pos - 1]) if pos <= len(bits) else 0

    value = 0
    for _ in range(PREC):
        value = 2 * value + bit()
    low, high, ids = 0, TOP, []
    g = m.steps()
    cum = next(g)
    for i in range(n):
        total = int(cum[-1])
        rng = high - low + 1
        target = ((value - low + 1) * total - 1) // rng
        t = int(np.searchsorted(cum, target, side="right"))
        hi_c, lo_c = int(cum[t]), int(cum[t - 1]) if t else 0
        high = low + rng * hi_c // total - 1
        low = low + rng * lo_c // total
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                value -= HALF
                low -= HALF
                high -= HALF
            elif low >= QTR and high < HALF + QTR:
                value -= QTR
                low -= QTR
                high -= QTR
            else:
                break
            low, high, value = 2 * low, 2 * high + 1, 2 * value + bit()
        ids.append(t)
        if i + 1 < n:
            cum = g.send(t)
    return ids


if __name__ == "__main__":
    mode, repo, src, dst = sys.argv[1:5]
    t0 = time.time()
    m = Model(repo)
    t_load = time.time() - t0
    t0 = time.time()
    if mode == "encode":
        text = Path(src).read_bytes().decode("utf-8")
        ids = m.encode_text(text)
        body = encode(m, ids)
        Path(dst).write_bytes(struct.pack("<I", len(ids)) + body)
        dt = time.time() - t0
        print(f"encode {repo}: {len(text.encode())} -> {4 + len(body)} bytes, {len(ids)} tokens, {dt:.0f}s ({len(ids) / dt:.1f} tok/s), load {t_load:.0f}s", flush=True)
    else:
        data = Path(src).read_bytes()
        (n,) = struct.unpack("<I", data[:4])
        ids = decode(m, data[4:], n)
        text = m.hf.decode(ids, clean_up_tokenization_spaces=False)
        Path(dst).write_bytes(text.encode("utf-8"))
        dt = time.time() - t0
        print(f"decode {repo}: {len(data)} -> {len(text.encode())} bytes, {n} tokens, {dt:.0f}s ({n / dt:.1f} tok/s), load {t_load:.0f}s", flush=True)
