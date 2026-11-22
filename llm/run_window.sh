#!/bin/sh
# Size against window (dial 1) and segments (dial 2), Qwen2.5-Coder-0.5B, MLX. Results in llm/out/window.jsonl.
cd "$(dirname "$0")/.."
R=mlx-community/Qwen2.5-Coder-0.5B-bf16
BJJ="bjj-case/3-fully-minified.html bjj-case/1-readable.html"
for W in 1024 4096 8192 16384; do echo "$(date +%T) bjj window $W"; .venv/bin/python llm/window.py $R $W 1 $BJJ; done
echo "$(date +%T) bjj readable full"; .venv/bin/python llm/window.py $R 0 1 bjj-case/1-readable.html
echo "$(date +%T) bjj readable 2048"; .venv/bin/python llm/window.py $R 2048 1 bjj-case/1-readable.html
for N in 2 4 8; do for W in 4096 8192; do echo "$(date +%T) bjj segments $N window $W"; .venv/bin/python llm/window.py $R $W $N $BJJ; done; done
for W in 0 2048 4096 8192; do echo "$(date +%T) corpus window $W"; .venv/bin/python llm/window.py $R $W 1 $(cat llm/window_sample.txt); done
echo "$(date +%T) all done"
