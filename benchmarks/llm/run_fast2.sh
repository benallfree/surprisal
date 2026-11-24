#!/bin/sh
# Re-time every full-size row with fastcodec2 (same archives as fastcodec). Decode the existing archives first, then encode.
cd "$(dirname "$0")/.."
F=bjj-case/3-fully-minified.html
rows="0.5b-bf16 0.5b-8bit 0.5b-4bit 1.5b-4bit 1.5b-8bit 1.5b-bf16 3b-4bit 3b-8bit 7b-4bit 7b-8bit"
model() { s=${1%-*}; v=${1#*-}
  if [ "$v" = bf16 ]; then S=$(echo $s | tr b B); echo "mlx-community/Qwen2.5-Coder-$S-bf16"; else echo "llm/models/coder$s-$v"; fi; }
cache() { [ "$1" = 0.5b-bf16 ] && echo "$HOME/.cache/huggingface/hub" || echo "$PWD/llm/hf-cache"; }
mkdir -p llm/real2
for r in $rows; do
  echo "$(date +%T) decode $r"
  HF_HUB_CACHE=$(cache $r) .venv/bin/python llm/fastcodec2.py decode $(model $r) llm/real/fast-$r.13k llm/real2/$r.out 2>&1 | grep -v Fetch
  cmp $F llm/real2/$r.out && echo "round trip OK"
done
for r in $rows; do
  echo "$(date +%T) encode $r"
  HF_HUB_CACHE=$(cache $r) .venv/bin/python llm/fastcodec2.py encode $(model $r) $F llm/real2/$r.13k 2>&1 | grep -v Fetch
  cmp llm/real/fast-$r.13k llm/real2/$r.13k && echo "archive identical to fastcodec"
done
echo "$(date +%T) all done"
