"""A small byte-level GPT trained from scratch on js13k 2020-2024 text only (no internet pretraining).

    .venv/bin/python llm/scratch.py train <steps>    -> llm/ckpt/scratch.safetensors (+ val bits/byte as it goes)
    .venv/bin/python llm/scratch.py score            -> llm/out/scratch.json (same format as score.py)

Byte-level (vocab 256), so it needs no tokenizer and can't lean on anything learned elsewhere.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

root = Path(__file__).resolve().parent.parent
CTX, D, LAYERS, HEADS, BATCH = 1024, 384, 8, 6, 24
CKPT = root / "llm" / "ckpt" / "scratch.safetensors"


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc1, self.fc2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
        self.rope = nn.RoPE(D // HEADS)

    def __call__(self, x, mask):
        B, T, _ = x.shape
        q, k, v = mx.split(self.qkv(self.ln1(x)), 3, axis=-1)
        sh = lambda t: t.reshape(B, T, HEADS, D // HEADS).transpose(0, 2, 1, 3)
        q, k, v = self.rope(sh(q)), self.rope(sh(k)), sh(v)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // HEADS) ** -0.5, mask=mask)
        x = x + self.proj(a.transpose(0, 2, 1, 3).reshape(B, T, D))
        return x + self.fc2(nn.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, D)
        self.blocks = [Block() for _ in range(LAYERS)]
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, 256)

    def __call__(self, idx):
        T = idx.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(mx.float32)
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def val_bpb(model, val, n=64, seed=0):
    rng = np.random.default_rng(seed)
    offs = rng.integers(0, len(val) - CTX - 1, n)
    tot = 0.0
    for i in range(0, n, BATCH):
        o = offs[i : i + BATCH]
        x = mx.array(np.stack([val[j : j + CTX] for j in o]).astype(np.int32))
        y = mx.array(np.stack([val[j + 1 : j + CTX + 1] for j in o]).astype(np.int32))
        tot += float(nn.losses.cross_entropy(model(x), y, reduction="sum").item())
    return tot / (n * CTX) / math.log(2)


def train(steps):
    data = np.frombuffer((root / "llm" / "train.bin").read_bytes(), dtype=np.uint8)
    val = np.frombuffer((root / "llm" / "val.bin").read_bytes(), dtype=np.uint8)
    mx.random.seed(0)
    model = GPT()
    mx.eval(model.parameters())
    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(time.strftime("%H:%M:%S"), f"params {nparams / 1e6:.1f}M, train {len(data) / 1e6:.1f} MB, {steps} steps x {BATCH}x{CTX} = {steps * BATCH * CTX / len(data):.1f} epochs", flush=True)
    sched = optim.join_schedules([optim.linear_schedule(1e-5, 6e-4, 200), optim.cosine_decay(6e-4, max(1, steps - 200), 3e-5)], [200])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.1)
    step_fn = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(1)
    t0 = time.time()
    for s in range(1, steps + 1):
        o = rng.integers(0, len(data) - CTX - 1, BATCH)
        x = mx.array(np.stack([data[j : j + CTX] for j in o]).astype(np.int32))
        y = mx.array(np.stack([data[j + 1 : j + CTX + 1] for j in o]).astype(np.int32))
        loss, grads = step_fn(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if s % 100 == 0 or s == 1:
            msg = f"step {s} loss {loss.item() / math.log(2):.3f} bits/byte ({(time.time() - t0) / s:.2f}s/step)"
            if s % 500 == 0 or s == steps:
                msg += f"  val {val_bpb(model, val):.3f} bits/byte"
                CKPT.parent.mkdir(parents=True, exist_ok=True)
                model.save_weights(str(CKPT))
            print(time.strftime("%H:%M:%S"), msg, flush=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(CKPT))


def score():
    model = GPT()
    model.load_weights(str(CKPT))
    stride = CTX * 3 // 4
    res = {}
    out = root / "llm" / "out" / "scratch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(json.loads((root / "results.json").read_text())):
        s = g["slug"]
        seq = [0] + list((root / "llm" / "data" / f"{s}.txt").read_bytes())
        bits = 0.0
        for c in range(1, len(seq), stride):
            lo, hi = max(0, c - (CTX - stride)), min(len(seq), c + stride)
            x = mx.array(seq[lo:hi], dtype=mx.int32)[None]
            lg = model(x[:, :-1]).astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            picked = mx.take_along_axis(lp, x[:, 1:, None], axis=-1)[0, :, 0][c - lo - 1 :]
            bits += -float(mx.sum(picked).item()) / math.log(2)
        res[s] = {"bytes": math.ceil(bits / 8) + 6 if len(seq) > 1 else 0, "tokens": len(seq) - 1}
        if i % 25 == 0:
            print(time.strftime("%H:%M:%S"), f"{i + 1}: {s} {res[s]}", flush=True)
    out.write_text(json.dumps(res))
    print(time.strftime("%H:%M:%S"), f"done scratch: total {sum(r['bytes'] for r in res.values())} bytes")


if __name__ == "__main__":
    train(int(sys.argv[2])) if sys.argv[1] == "train" else score()
