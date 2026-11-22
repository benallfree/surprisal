"""Real LLM compressor with N segments stepped in one batched forward pass (nollama dial 2, measured on MLX).

    .venv/bin/python llm/segcodec.py encode <hf-repo> <segments> <input> <archive.13k>
    .venv/bin/python llm/segcodec.py decode <hf-repo> <archive.13k> <output>

The file's tokens are cut into N equal segments. Each step feeds one token per segment as a batch of N rows (one
KV cache with batch N) and codes each row's next token in that segment's own arithmetic-coded stream. A finished
segment keeps feeding the start token so the batch stays the same shape. Encoder and decoder run the identical
sequence of batched steps, so they compute identical probabilities on the same machine.

Archive: u8 N, then N u32 token counts, N u32 stream lengths (little-endian), then the N streams
(Witten-Neal-Cleary, 32-bit, frequency total 2^28, every token at least 1).
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


def freqs(logits):
    """[N, V] logits -> [N, V] cumulative integer frequencies, in float64 on the CPU."""
    x = np.asarray(logits.astype(mx.float32), dtype=np.float64)
    p = np.exp(x - x.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    q = np.floor(p * (TOTAL - x.shape[1])).astype(np.int64) + 1
    return np.cumsum(q, axis=1)


class Encoder:
    def __init__(self):
        self.low, self.high, self.pending, self.bits = 0, TOP, 0, []

    def emit(self, b):
        self.bits.append(b)
        self.bits.extend([1 - b] * self.pending)
        self.pending = 0

    def code(self, cum, t):
        self.code_counts(int(cum[-1]), int(cum[t]), int(cum[t - 1]) if t else 0)

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

    def code(self, cum):
        total = int(cum[-1])
        rng = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // rng
        t = int(np.searchsorted(cum, target, side="right"))
        hi_c, lo_c = int(cum[t]), int(cum[t - 1]) if t else 0
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
        return t


def load_model(repo):
    model, tok = load(repo)
    hf = tok._tokenizer
    start = hf.bos_token_id if hf.bos_token_id is not None else hf.eos_token_id
    return model, hf, start


def encode_text(hf, text):
    norm = hf.backend_tokenizer.normalizer
    for n in (norm, None):
        hf.backend_tokenizer.normalizer = n
        ids = hf.encode(text, add_special_tokens=False)
        if hf.decode(ids, clean_up_tokenization_spaces=False) == text:
            return ids
    raise ValueError("tokenizer round-trip mismatch")


def run(model, start, lengths, choose):
    """Step all segments together. choose(i, k, cum) returns segment i's token at index k given its frequencies."""
    n = len(lengths)
    cache = make_prompt_cache(model)
    feed = [start] * n
    out = [[] for _ in range(n)]
    for k in range(max(lengths)):
        logits = model(mx.array(feed)[:, None], cache=cache)[:, -1]
        cum = freqs(logits)
        for i in range(n):
            if k < lengths[i]:
                t = choose(i, k, cum[i])
                out[i].append(t)
                feed[i] = t
            else:
                feed[i] = start
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    t0 = time.time()
    if mode == "encode":
        repo, n, src, dst = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    else:
        repo, src, dst = sys.argv[2], sys.argv[3], sys.argv[4]
    model, hf, start = load_model(repo)
    t_load = time.time() - t0
    t0 = time.time()
    if mode == "encode":
        text = Path(src).read_bytes().decode("utf-8")
        ids = encode_text(hf, text)
        cut = [len(ids) * i // n for i in range(n + 1)]
        segs = [ids[cut[i] : cut[i + 1]] for i in range(n)]
        enc = [Encoder() for _ in range(n)]

        def choose(i, k, cum):
            enc[i].code(cum, segs[i][k])
            return segs[i][k]

        run(model, start, [len(s) for s in segs], choose)
        streams = [e.finish() for e in enc]
        head = struct.pack("<B", n) + struct.pack(f"<{n}I", *map(len, segs)) + struct.pack(f"<{n}I", *map(len, streams))
        archive = head + b"".join(streams)
        Path(dst).write_bytes(archive)
        dt = time.time() - t0
        print(f"encode {repo} x{n}: {len(text.encode())} -> {len(archive)} bytes, {len(ids)} tokens, {dt:.0f}s "
              f"({len(ids) / dt:.1f} tok/s), load {t_load:.0f}s", flush=True)
    else:
        data = Path(src).read_bytes()
        n = data[0]
        lengths = struct.unpack_from(f"<{n}I", data, 1)
        sizes = struct.unpack_from(f"<{n}I", data, 1 + 4 * n)
        off, dec = 1 + 8 * n, []
        for s in sizes:
            dec.append(Decoder(data[off : off + s]))
            off += s
        segs = run(model, start, list(lengths), lambda i, k, cum: dec[i].code(cum))
        text = hf.decode([t for s in segs for t in s], clean_up_tokenization_spaces=False)
        Path(dst).write_bytes(text.encode("utf-8"))
        dt = time.time() - t0
        print(f"decode {repo} x{n}: {len(data)} -> {len(text.encode())} bytes, {sum(lengths)} tokens, {dt:.0f}s "
              f"({sum(lengths) / dt:.1f} tok/s), load {t_load:.0f}s", flush=True)
