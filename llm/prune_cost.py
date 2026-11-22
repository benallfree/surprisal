"""Ideal coded size of BJJ with the full vocabulary vs a vocabulary pruned to the training corpus plus an escape.

    .venv/bin/python llm/prune_cost.py <model> [file]

Scores the file in 1,024-token prefill chunks (probabilities differ from the token-by-token codec in the last bits only).
Pruned scheme: the model's logits over the kept tokens plus an ESC symbol with an adaptive probability
(escapes so far + 0.5) / (tokens so far + 1); after ESC, the token is coded from the full distribution restricted
to the tokens outside the kept set. Prints both totals in bytes.
"""
import sys, math, mlx.core as mx, numpy as np
sys.path.insert(0, 'llm')
from mlx_lm import load
from segcodec import encode_text
model, tok = load(sys.argv[1]); hf = tok._tokenizer
src = sys.argv[2] if len(sys.argv) > 2 else 'bjj-case/3-fully-minified.html'
S = set()
for arch in open('llm/train.bin', 'rb').read().split(b'\0'):
    if arch: S.update(hf.encode(arch.decode('utf-8', errors='replace'), add_special_tokens=False))
keep = np.zeros(model.model.embed_tokens.weight.shape[0], bool); keep[sorted(S)] = True
ids = encode_text(hf, open(src).read()); start = hf.bos_token_id or hf.eos_token_id
from mlx_lm.models.cache import make_prompt_cache
inp = [start] + ids[:-1]; cache = make_prompt_cache(model); keepm = mx.array(keep)
LPT, MIN = [], []
for i in range(0, len(inp), 1024):
    logits = model(mx.array([inp[i:i + 1024]]), cache=cache)[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    MIN.append(mx.exp(mx.logsumexp(mx.where(keepm[: lp.shape[1]], lp, -mx.inf), axis=-1)))
    LPT.append(mx.take_along_axis(lp, mx.array(ids[i:i + 1024])[:, None], axis=-1)[:, 0])
    mx.eval(LPT[-1], MIN[-1])
lpt, mass_in = mx.concatenate(LPT), mx.concatenate(MIN)
lpt, mass_in = np.array(lpt, np.float64), np.clip(np.array(mass_in, np.float64), 1e-12, 1 - 1e-12)
full = -lpt.sum() / math.log(2) / 8
bits, esc = 0.0, 0
for k, t in enumerate(ids):
    pe = (esc + 0.5) / (k + 1)
    if keep[t]:
        bits += -(math.log2(1 - pe) + lpt[k] / math.log(2) - math.log2(mass_in[k]))
    else:
        bits += -(math.log2(pe) + lpt[k] / math.log(2) - math.log2(1 - mass_in[k]))
        esc += 1
print(f"{sys.argv[1]}: full vocab {full:,.0f} bytes; pruned to {keep.sum():,} + escape {bits / 8:,.0f} bytes "
      f"({esc} escapes, mean out-of-set mass {1 - mass_in.mean():.4f})")
