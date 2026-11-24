"""Score a file with Ollama as the predictor: the size an arithmetic coder driven by it would produce.

    python3 nollama/ollama_score.py <ollama-model> <file> <ids.json>

Ollama can't score a given text: it reports logprobs only for tokens it generates, and at most the top 20.
So the codec asks once per token. It sends the raw text before token i (Ollama's prefix cache means
only the new part is read), generates one token, and gets the top 20 next tokens with their logprobs.
The real token (from <ids.json>, written by `DUMP_IDS=dir ./nlscore`, matched here by its bytes) costs
-log2 p if it is in the top 20. Otherwise it costs an escape: the leftover probability, spread evenly
over the rest of the vocabulary. A small floor EPS keeps the escape codable when the top 20 hold
almost all the mass. Encoder and decoder both see the same prefix text, so this is a real lossless
codec, just a weaker one than having the full distribution.
"""
import json, math, sys, time, urllib.request

VOCAB, TOP, EPS = 151936, 20, 1e-4
model, path, ids_path = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(path, "rb").read().decode("utf-8")
ids = json.load(open(ids_path))
# token byte strings, recovered from the text by re-joining: the prefix up to token i is text_bytes[:off[i]]
tb = text.encode("utf-8")


def generate(prompt: str) -> dict:
    body = {"model": model, "prompt": prompt, "raw": True, "stream": False, "logprobs": True, "top_logprobs": TOP,
            "keep_alive": "30m", "options": {"num_predict": 1, "temperature": 0, "num_ctx": 32768}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


# Token byte lengths: ask nothing of Ollama, use the ids' detokenized pieces via the offsets file nlscore wrote.
offs = json.load(open(ids_path.replace(".ids.json", ".offs.json")))
bits = math.log2(VOCAB)  # first token: uniform
miss = split = 0
t0 = time.time()
for i in range(1, len(ids)):
    tok = tb[offs[i] : offs[i + 1]]
    try:
        prefix = tb[: offs[i]].decode("utf-8")
    except UnicodeDecodeError:  # prefix ends inside a character: Ollama can't take it, code the token uniformly
        split += 1
        bits += math.log2(VOCAB)
        continue
    r = generate(prefix)
    top = r["logprobs"][0]["top_logprobs"]
    ps = {bytes(t["bytes"]): math.exp(t["logprob"]) for t in top}
    mass = min(sum(ps.values()), 1.0)
    if tok in ps:
        bits -= math.log2(ps[tok] * (1 - EPS))
    else:
        miss += 1
        bits -= math.log2(((1 - mass) * (1 - EPS) + EPS) / (VOCAB - len(ps)))
    if i % 500 == 0:
        print(time.strftime("%H:%M:%S"), f"{i}/{len(ids)} tokens, {bits / 8:.0f} bytes so far, {miss} escapes, {i / (time.time() - t0):.0f} tok/s", flush=True)
dt = time.time() - t0
print(f"{path.split('/')[-1]}: {math.ceil(bits / 8) + 6} bytes, {len(ids)} tokens, {bits / len(tb):.3f} bits/byte, "
      f"{miss} escapes ({100 * miss / len(ids):.1f}%), {split} split characters, {dt:.0f}s ({len(ids) / dt:.0f} tok/s)", flush=True)
