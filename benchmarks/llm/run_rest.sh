#!/bin/sh
# Remaining LLM arms, one GPU job at a time. Each step logs to llm/<step>.log.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
step() { echo "$(date +%T) start $1"; }
step "qwen05 rescore (2 games after normalizer fix)"
$PY llm/score.py mlx-community/Qwen2.5-Coder-0.5B-bf16 qwen05 > llm/qwen05-fix.log 2>&1
step "scratch train 3000"
$PY llm/scratch.py train 3000 > llm/scratch-train.log 2>&1
step "scratch score"
$PY llm/scratch.py score > llm/scratch-score.log 2>&1
step "qwen LoRA 400 iters, lr 1e-5 (1e-4 diverged)"
$PY -m mlx_lm lora --model mlx-community/Qwen2.5-Coder-0.5B-bf16 --train --data llm/ft --fine-tune-type lora \
  --num-layers 24 --batch-size 4 --max-seq-length 2048 --iters 400 --steps-per-report 50 --steps-per-eval 100 \
  --val-batches 20 --learning-rate 1e-5 --adapter-path llm/ckpt/qwen05-lora --grad-checkpoint > llm/lora-train.log 2>&1
step "qwen05ft score"
$PY llm/score.py mlx-community/Qwen2.5-Coder-0.5B-bf16 qwen05ft llm/ckpt/qwen05-lora > llm/qwen05ft.log 2>&1
echo "$(date +%T) all done"
