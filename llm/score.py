"""Score each test game's text part with a language model: archive size = sum of -log2 p(token).

    .venv/bin/python llm/score.py <hf-repo-or-local-path> <label> [adapter-dir]   -> llm/out/<label>.json

This is exactly the size an arithmetic coder driven by the model would produce (plus a few bytes),
assuming both ends run bit-identical inference. Long texts are scored in overlapping windows so each
token sees up to WINDOW-STRIDE tokens of earlier context.
"""
import json, math, sys, time, os
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

WINDOW = int(os.environ.get("LLM_WINDOW", 2048))  # tokens of context per scoring window
STRIDE = WINDOW * 3 // 4
CHUNK = 2048  # tokens per forward pass inside a window
root = Path(__file__).resolve().parent.parent
repo, label = sys.argv[1], sys.argv[2]
adapter = sys.argv[3] if len(sys.argv) > 3 else None  # optional LoRA adapter dir (fine-tuned arm)
model, tok = load(repo, adapter_path=adapter)
hf = tok._tokenizer
# Qwen's tokenizer NFC-normalizes text, which is lossy for some Unicode (combining marks, musical symbols).
# Without the normalizer the byte-level BPE is exactly reversible; already-normalized text tokenizes the same.
hf.backend_tokenizer.normalizer = None
start_id = hf.eos_token_id if hf.eos_token_id is not None else 0


def bits_of(text: str) -> tuple[float, int]:
    ids = hf.encode(text, add_special_tokens=False)
    if hf.decode(ids, clean_up_tokenization_spaces=False) != text:
        raise ValueError("tokenizer round-trip mismatch")
    seq = [start_id] + ids
    total = 0.0
    for c in range(1, len(seq), STRIDE):
        lo = max(0, c - (WINDOW - STRIDE))
        hi = min(len(seq), c + STRIDE)
        # Feed the window through in CHUNK-token pieces with a KV cache: identical result to one forward
        # pass, but attention memory stays bounded, so WINDOW can be Qwen's full 32,768 tokens.
        cache = make_prompt_cache(model)
        for a in range(lo, hi - 1, CHUNK):
            b = min(a + CHUNK, hi - 1)
            logits = model(mx.array(seq[a:b])[None], cache=cache).astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tgt = mx.array(seq[a + 1 : b + 1])[None]
            picked = mx.take_along_axis(lp, tgt[..., None], axis=-1)[0, :, 0]
            skip = max(0, c - 1 - a)  # only score targets >= c (earlier ones were scored by the previous window)
            if skip < b - a:
                total += -float(mx.sum(picked[skip:]).item()) / math.log(2)
            mx.eval(cache[0].keys)
    return total, len(ids)


slugs = json.loads((root / "results.json").read_text())
out_path = root / "llm" / "out" / f"{label}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
res = json.loads(out_path.read_text()) if out_path.exists() else {}
t0 = time.time()
for i, g in enumerate(slugs):
    s = g["slug"]
    if s in res:
        continue
    text = (root / "llm" / "data" / f"{s}.txt").read_bytes().decode("utf-8")
    try:
        b, n = bits_of(text) if text else (0.0, 0)
        res[s] = {"bytes": math.ceil(b / 8) + 6 if text else 0, "tokens": n}
    except ValueError as e:
        res[s] = {"error": str(e)}
    if i % 10 == 0:
        out_path.write_text(json.dumps(res))
        print(time.strftime("%H:%M:%S"), f"{i + 1}/{len(slugs)} {s}: {res[s]}  ({time.time() - t0:.0f}s)", flush=True)
out_path.write_text(json.dumps(res))
ok = [r for r in res.values() if "bytes" in r]
print(time.strftime("%H:%M:%S"), f"done {label}: {len(ok)} scored, {len(res) - len(ok)} errors, total {sum(r['bytes'] for r in ok)} bytes, {sum(r['tokens'] for r in ok)} tokens")
