"""Build mlx_lm.lora training data from llm/train.bin and llm/val.bin (archives separated by 0x00).

    .venv/bin/python llm/ft_data.py     -> llm/ft/train.jsonl, llm/ft/valid.jsonl

Each archive's text is cut into ~5,000-character chunks (about 1,600-2,000 Qwen tokens), so nothing is
lost to the trainer's 2,048-token truncation.
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
CHUNK = 5000
out = root / "llm" / "ft"
out.mkdir(parents=True, exist_ok=True)
for src, dst in [("train.bin", "train.jsonl"), ("val.bin", "valid.jsonl")]:
    n = 0
    with open(out / dst, "w") as f:
        for arch in (root / "llm" / src).read_bytes().split(b"\0"):
            text = arch.decode("utf-8", errors="strict") if arch else ""
            for i in range(0, len(text), CHUNK):
                f.write(json.dumps({"text": text[i : i + CHUNK]}) + "\n")
                n += 1
    print(dst, n, "chunks")
