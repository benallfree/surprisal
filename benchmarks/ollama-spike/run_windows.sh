#!/bin/sh
# Real encode then decode of the minified BJJ game with nollama on Metal (Qwen3-0.6B-Base Q8_0) at several windows.
cd "$(dirname "$0")/.."
M=nollama/models/Qwen3-0.6B-Base-Q8_0.gguf
IN=bjj-case/3-fully-minified.html
mkdir -p nollama/real
while pgrep -f run_seg7b.sh >/dev/null; do echo "$(date +%T) waiting for the 7B segment run"; sleep 60; done
for W in 2048 4096 8192 0; do
  echo "$(date +%T) window $W"
  NLC_WINDOW=$W nollama/codec/nlcodec encode $M $IN nollama/real/w$W.13k
  nollama/codec/nlcodec decode $M nollama/real/w$W.13k nollama/real/w$W.out
  cmp $IN nollama/real/w$W.out && echo "round trip OK" || echo "ROUND TRIP FAILED"
done
echo "$(date +%T) all done"
