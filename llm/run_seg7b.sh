#!/bin/sh
# Real encode and decode of the minified BJJ game with Qwen2.5-Coder-7B on MLX, split into N segments stepped together.
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=$PWD/llm/hf-cache
R=mlx-community/Qwen2.5-Coder-7B-bf16
IN=bjj-case/3-fully-minified.html
mkdir -p llm/real
until grep -q "hf-cache" llm/dl7b.log && ! pgrep -f snapshot_download >/dev/null; do echo "$(date +%T) waiting for download"; sleep 30; done
for N in 16 8 4 2; do
  echo "$(date +%T) segments $N"
  .venv/bin/python llm/segcodec.py encode $R $N $IN llm/real/7b-x$N.13k 2>&1 | grep -v Fetch
  .venv/bin/python llm/segcodec.py decode $R llm/real/7b-x$N.13k llm/real/7b-x$N.out 2>&1 | grep -v Fetch
  cmp $IN llm/real/7b-x$N.out && echo "round trip OK" || echo "ROUND TRIP FAILED"
done
echo "$(date +%T) all done"
