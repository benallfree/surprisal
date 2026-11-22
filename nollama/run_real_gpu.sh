#!/bin/sh
# Real encode then decode of the minified BJJ game with nollama on the GPU, then a byte compare.
M=nollama/models/Qwen3-0.6B-Base-Q8_0.gguf
IN=bjj-case/3-fully-minified.html
echo "$(date +%T) encode"
nollama/codec/nlcodec encode $M $IN nollama/real/qwen3-0.6b-q8.13k
echo "$(date +%T) decode"
nollama/codec/nlcodec decode $M nollama/real/qwen3-0.6b-q8.13k nollama/real/qwen3-0.6b-q8.out
cmp $IN nollama/real/qwen3-0.6b-q8.out && echo "round trip OK" || echo "ROUND TRIP FAILED"
echo "$(date +%T) all done"
