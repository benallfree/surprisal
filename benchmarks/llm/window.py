"""Archive size against a true sliding attention window (nollama dial 1), and against independent segments (dial 2).

    .venv/bin/python llm/window.py <hf-repo> <window> <segments> <file>...   -> appends to llm/out/window.jsonl

Every token is predicted from the start token (an attention sink) plus between WINDOW and WINDOW + WINDOW/4 tokens
right before it, the way a ring KV cache of WINDOW slots would see it (positions renumbered, StreamingLLM-style).
window 0 means the whole file as context. With segments N > 1 the file's tokens are cut into N equal parts, each
coded on its own with no context from the others, and each part pays a 6-byte header and flush.
"""
import json, math, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

CHUNK = 2048
root = Path(__file__).resolve().parent.parent
repo, window, segments, files = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
model, tok = load(repo)
hf = tok._tokenizer
hf.backend_tokenizer.normalizer = None  # Qwen's NFC normalizer is lossy; the byte-level BPE is reversible without it
start = hf.eos_token_id


def bits_of(ids):
    seq = [start] + ids
    n = len(seq)
    W = window if window else n
    B = max(256, W // 4) if window else n
    total = 0.0
    for s in range(1, n, B):
        e = min(s + B, n)
        lo = max(1, s - W)
        full = [start] + seq[lo:e]  # sink, context, targets seq[s:e]
        first = 1 + (s - lo)  # index in full of the first target
        cache = make_prompt_cache(model)
        for a in range(0, len(full) - 1, CHUNK):
            b = min(a + CHUNK, len(full) - 1)
            logits = model(mx.array(full[a:b])[None], cache=cache).astype(mx.float32)
            k0 = max(a, first - 1)  # position k predicts full[k+1]
            if k0 < b:
                lp = logits[0, k0 - a :] - mx.logsumexp(logits[0, k0 - a :], axis=-1, keepdims=True)
                tgt = mx.array(full[k0 + 1 : b + 1])
                total += -float(mx.sum(mx.take_along_axis(lp, tgt[:, None], axis=-1)).item()) / math.log(2)
            mx.eval(cache[0].keys)
    return total


out = root / "llm" / "out" / "window.jsonl"
for f in files:
    text = Path(f).read_bytes().decode("utf-8")
    ids = hf.encode(text, add_special_tokens=False)
    if hf.decode(ids, clean_up_tokenization_spaces=False) != text:
        print(f, "tokenizer round-trip mismatch", flush=True)
        continue
    t0 = time.time()
    cut = [len(ids) * i // segments for i in range(segments + 1)]
    size = sum(math.ceil(bits_of(ids[cut[i] : cut[i + 1]]) / 8) + 6 for i in range(segments))
    rec = {"repo": repo, "window": window, "segments": segments, "file": f, "bytes": size, "tokens": len(ids),
           "raw": len(text.encode()), "s": round(time.time() - t0, 1)}
    with out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(time.strftime("%H:%M:%S"), json.dumps(rec), flush=True)
