"""Model sweep: one model per run, appends a JSON line to llm/out/sweep.jsonl.

    .venv/bin/python llm/sweep.py <hf-repo> <window> <file>...

For each file: exact arithmetic-coded size (sum of -log2 p over tokens + 6 bytes) with a context window of
<window> tokens (fed through in 2,048-token chunks with a KV cache, as in score.py). Also measures
generation speed, which is decompression speed: a decoder must produce the file one token at a time,
so it runs one cached forward step per token. Speed is timed over 256 single-token steps after 1,024
tokens of context.
"""
import json, math, os, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

CHUNK = 2048
root = Path(__file__).resolve().parent.parent
repo, window, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
stride = window * 3 // 4
t_load = time.time()
model, tok = load(repo)
t_load = time.time() - t_load
hf = tok._tokenizer
start_id = hf.bos_token_id if hf.bos_token_id is not None else hf.eos_token_id


def encode(text):
    """Token ids that decode back to exactly text. Use the tokenizer's own normalizer when it round-trips
    (Gemma's turns spaces into its word marker; without it every space becomes a byte-fallback token the
    model never saw). Qwen's NFC normalizer is lossy on some Unicode, so fall back to none."""
    norm = hf.backend_tokenizer.normalizer
    for n in (norm, None):
        hf.backend_tokenizer.normalizer = n
        ids = hf.encode(text, add_special_tokens=False)
        if hf.decode(ids, clean_up_tokenization_spaces=False) == text:
            return ids
    raise ValueError("tokenizer round-trip mismatch")


def bits_of(seq):
    total = 0.0
    for c in range(1, len(seq), stride):
        lo, hi = max(0, c - (window - stride)), min(len(seq), c + stride)
        cache = make_prompt_cache(model)
        for a in range(lo, hi - 1, CHUNK):
            b = min(a + CHUNK, hi - 1)
            logits = model(mx.array(seq[a:b])[None], cache=cache).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            picked = mx.take_along_axis(lp, mx.array(seq[a + 1 : b + 1])[None][..., None], axis=-1)[0, :, 0]
            skip = max(0, c - 1 - a)
            if skip < b - a:
                total += -float(mx.sum(picked[skip:]).item()) / math.log(2)
    return total


row = {"repo": repo, "window": window, "load_s": round(t_load, 1), "files": {}}
seq = None
for p in files:
    text = Path(p).read_bytes().decode("utf-8")
    try:
        ids = encode(text)
    except ValueError as e:
        row["files"][Path(p).name] = {"error": str(e)}
        continue
    seq = [start_id] + ids
    t0 = time.time()
    b = bits_of(seq)
    row["files"][Path(p).name] = {"bytes": math.ceil(b / 8) + 6, "tokens": len(ids), "bpb": round(b / len(text.encode()), 4),
                                  "score_s": round(time.time() - t0, 1)}
    print(repo, Path(p).name, row["files"][Path(p).name], flush=True)

if seq:  # decode speed: 256 single-token cached steps after 1,024 tokens of context
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array(seq[:1024])[None], cache=cache))
    t0 = time.time()
    for i in range(1024, 1280):
        mx.eval(model(mx.array(seq[i : i + 1])[None], cache=cache))
    row["decode_tok_s"] = round(256 / (time.time() - t0), 1)
row["peak_mem_gb"] = round(mx.get_peak_memory() / 1e9, 2)
row["params_b"] = round(sum(v.size for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters())) / 1e9, 3)
print(json.dumps(row), flush=True)
out = root / "llm" / "out" / "sweep.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "a") as f:
    f.write(json.dumps(row) + "\n")
