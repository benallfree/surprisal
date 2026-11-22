#!/bin/sh
# Rescore the test games with Qwen's full 32,768-token context window, pretrained then fine-tuned.
set -e
cd "$(dirname "$0")/.."
export LLM_WINDOW=32768
echo "$(date +%T) start qwen05-32k"
.venv/bin/python llm/score.py mlx-community/Qwen2.5-Coder-0.5B-bf16 qwen05-32k > llm/qwen05-32k.log 2>&1
echo "$(date +%T) start qwen05ft-32k"
.venv/bin/python llm/score.py mlx-community/Qwen2.5-Coder-0.5B-bf16 qwen05ft-32k llm/ckpt/qwen05-lora > llm/qwen05ft-32k.log 2>&1
echo "$(date +%T) all done"
