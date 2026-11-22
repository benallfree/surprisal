#!/bin/sh
# Real encode, decode and byte compare of the minified BJJ game with llm/fastcodec.py, every model and precision.
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=$PWD/llm/hf-cache
IN=bjj-case/3-fully-minified.html
mkdir -p llm/real
until grep -q "all done" llm/prep.log; do echo "$(date +%T) waiting for model prep"; sleep 30; done
for s in 0.5b 1.5b 3b 7b; do
  S=$(echo $s | sed 's/b$/B/')
  for v in 4bit 8bit bf16; do
    if [ $s = 0.5b ]; then export HF_HUB_CACHE=$HOME/.cache/huggingface/hub; else export HF_HUB_CACHE=$PWD/llm/hf-cache; fi
    [ $v = bf16 ] && M=mlx-community/Qwen2.5-Coder-$S-bf16 || M=llm/models/coder$s-$v
    A=llm/real/fast-$s-$v
    case $s-$v in 3b-bf16|7b-bf16) continue;; esac
    echo "$(date +%T) $s $v"
    .venv/bin/python llm/fastcodec.py encode $M $IN $A.13k 2>&1 | grep -v Fetch
    .venv/bin/python llm/fastcodec.py decode $M $A.13k $A.out 2>&1 | grep -v Fetch
    cmp $IN $A.out && echo "round trip OK" || echo "ROUND TRIP FAILED"
  done
done
echo "$(date +%T) all done"
