#!/bin/sh
# Coder 0.5B/1.5B/3B/7B in bf16 (downloaded) and 8-bit and 4-bit (converted locally) under llm/models/.
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=$PWD/llm/hf-cache
for s in 0.5B 1.5B 3B 7B; do
  r=mlx-community/Qwen2.5-Coder-$s-bf16
  if [ $s = 0.5B ]; then export HF_HUB_CACHE=$HOME/.cache/huggingface/hub; else export HF_HUB_CACHE=$PWD/llm/hf-cache; fi
  echo "$(date +%T) download $s"
  .venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('$r')" 2>&1 | tail -1
  for b in 8 4; do
    o=llm/models/coder$(echo $s | tr A-Z a-z)-${b}bit
    [ -d $o ] && continue
    echo "$(date +%T) convert $s to $b-bit"
    .venv/bin/python -m mlx_lm convert --hf-path $r --mlx-path $o -q --q-bits $b 2>&1 | tail -1
  done
done
echo "$(date +%T) all done"
