#!/bin/sh
# Model sweep on the BJJ builds (readable, minified). One model per process; results in llm/out/sweep.jsonl.
# Models not already in the default Hugging Face cache download into llm/hf-cache and are deleted after
# scoring, so only one big model sits on disk at a time.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
F="bjj-case/1-readable.html bjj-case/3-fully-minified.html"
run() { echo "$(date +%T) start $1 (window $2)"; $PY llm/sweep.py "$1" "$2" $F 2>&1 | grep -E '^\{|Error|error' | tail -3; }
new() { HF_HUB_CACHE=$PWD/llm/hf-cache run "$1" "$2"; rm -rf llm/hf-cache; }  # the whole cache: weights live in hf-cache/blobs
# done before the drive filled (llm/out/sweep.jsonl): Qwen2.5-0.5B, Qwen2.5-Coder-0.5B(-Instruct), Qwen3-0.6B(-Base),
# Qwen2.5-1.5B, Qwen2.5-Coder-1.5B, gemma-3-1b-pt. Remaining, smallest first:
new Qwen/Qwen3-1.7B-Base 32768
new HuggingFaceTB/SmolLM2-1.7B 8192
new mlx-community/deepseek-coder-1.3b-base-mlx 16384
new mlx-community/Qwen2.5-Coder-3B-bf16 32768
new mlx-community/Llama-3.2-3B-bf16 32768
new Qwen/Qwen3-4B-Base 32768
new mlx-community/Qwen2.5-Coder-7B-bf16 32768
new mlx-community/Qwen3-8B-Base-bf16 32768
echo "$(date +%T) all done"
