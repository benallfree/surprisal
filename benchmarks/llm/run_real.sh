#!/bin/sh
# Real encode + decode of the minified BJJ build with four models, after the sweep frees the GPU.
# Each step is its own process; the decoded file is compared byte for byte. Log: llm/real.log
cd "$(dirname "$0")/.."
PY=.venv/bin/python
IN=bjj-case/3-fully-minified.html
mkdir -p llm/real
while pgrep -f run_sweep.sh >/dev/null; do sleep 15; done
one() {
  n=$(echo "$1" | tr / _)
  echo "$(date +%T) start $1"
  $PY llm/realcodec.py encode "$1" $IN llm/real/$n.13k 2>&1 | grep -E '^encode|Error'
  $PY llm/realcodec.py decode "$1" llm/real/$n.13k llm/real/$n.out 2>&1 | grep -E '^decode|Error'
  cmp -s $IN llm/real/$n.out && echo "round trip OK" || echo "ROUND TRIP FAILED"
}
one mlx-community/Qwen2.5-Coder-0.5B-bf16
for m in mlx-community/Qwen2.5-Coder-1.5B-bf16 Qwen/Qwen3-4B-Base mlx-community/Qwen2.5-Coder-7B-bf16; do
  HF_HUB_CACHE=$PWD/llm/hf-cache one $m; rm -rf llm/hf-cache
done
echo "$(date +%T) all done"
