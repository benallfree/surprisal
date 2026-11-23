"""How many tokens per second can a model decode when N streams step together?

    .venv/bin/python llm/batchbench.py <model> <N>...   -> one JSON line per N in llm/out/batchbench.jsonl

Each of N rows reads 1,024 tokens of the BJJ game as context, then takes 32 one-token steps. A step is what a fast
codec would do: forward pass, integer frequency table on the GPU (softmax, floor, cumsum), and reading back only
each row's total and the two cumulative counts around its token. Reported: steps/s and tokens/s (N x steps/s).
"""
import json, sys, time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

TOTAL = 1 << 28
repo, Ns = sys.argv[1], [int(x) for x in sys.argv[2:]]
model, tok = load(repo)
hf = tok._tokenizer
hf.backend_tokenizer.normalizer = None
text = Path("bjj-case/3-fully-minified.html").read_text()
ids = hf.encode(text, add_special_tokens=False)
out = Path("llm/out/batchbench.jsonl")

for N in Ns:
    cache = make_prompt_cache(model)
    ctx = mx.array([ids[i * 37 : i * 37 + 1024] for i in range(N)])  # different text per row
    for a in range(0, 1024, 256):
        model(ctx[:, a : a + 256], cache=cache)
        mx.eval([c.state for c in cache])
    feed = ctx[:, -1:]
    nxt = mx.array([ids[i * 37 + 1024 + k] for k in range(40) for i in range(N)]).reshape(40, N)
    times = []
    for k in range(36):
        t0 = time.perf_counter()
        logits = model(feed, cache=cache)[:, -1].astype(mx.float32)
        p = mx.softmax(logits, axis=-1)
        q = mx.floor(p * (TOTAL - p.shape[1])).astype(mx.int32) + 1
        cum = mx.cumsum(q, axis=-1)
        t = nxt[k]
        hi = mx.take_along_axis(cum, t[:, None], axis=-1)
        lo = mx.take_along_axis(cum, mx.maximum(t - 1, 0)[:, None], axis=-1)
        tot = cum[:, -1]
        mx.eval(hi, lo, tot)
        _ = (hi.tolist(), lo.tolist(), tot.tolist())
        feed = t[:, None]
        if k >= 4:  # skip warm-up steps
            times.append(time.perf_counter() - t0)
    sps = len(times) / sum(times)
    rec = {"model": repo, "N": N, "steps_s": round(sps, 1), "tok_s": round(N * sps), "peak_gb": round(mx.get_peak_memory() / 1e9, 1)}
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(time.strftime("%H:%M:%S"), json.dumps(rec), flush=True)
    del cache
    mx.clear_cache()
