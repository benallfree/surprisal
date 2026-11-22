#!/bin/sh
# Back-to-back A/B: old fastcodec vs pipelined fastcodec2, decoding the same 6 KB slice archive, alternating.
cd "$(dirname "$0")/.."
SL=llm/ab/slice.html; mkdir -p llm/ab; head -c 6000 bjj-case/3-fully-minified.html > $SL
for r in 0.5b-8bit 1.5b-bf16 3b-8bit 7b-8bit; do
  s=${r%-*}; v=${r#*-}
  if [ $v = bf16 ]; then M=mlx-community/Qwen2.5-Coder-$(echo $s | tr b B)-bf16; else M=llm/models/coder$r; fi
  export HF_HUB_CACHE=$PWD/llm/hf-cache
  echo "$(date +%T) $r"
  .venv/bin/python llm/fastcodec.py encode $M $SL llm/ab/$r.13k 2>&1 | grep -v Fetch
  for i in 1 2; do for c in fastcodec fastcodec2; do
    .venv/bin/python llm/$c.py decode $M llm/ab/$r.13k llm/ab/$r.out 2>&1 | grep -v Fetch | sed "s/^/$c /"
    cmp -s $SL llm/ab/$r.out || echo "ROUND TRIP FAILED"
  done; done
done
echo "$(date +%T) all done"
