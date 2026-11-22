#!/bin/sh
# Score BJJ builds 3 (minified) and 1 (readable) with Qwen3-0.6B-Base (ggml-org Q8_0 GGUF) under both engines.
#   nollama/run_bjj.sh nollama   (CPU, exact teacher forcing)
#   nollama/run_bjj.sh ollama    (Ollama on Metal, top-20 + escape, one request per token)
# Ollama model: ollama create qwen3-0.6b-base-q8 -f nollama/Modelfile.base
set -e
cd "$(dirname "$0")"
GGUF=models/Qwen3-0.6B-Base-Q8_0.gguf
mkdir -p ids
B="../bjj-case/3-fully-minified.html ../bjj-case/1-readable.html"
go build -o nlscore .
DUMP_IDS=ids ./nlscore "$GGUF" $B > /dev/null
case "$1" in
nollama) echo "$(date +%T) start nollama"; ./nlscore "$GGUF" $B; echo "$(date +%T) all done" ;;
ollama) echo "$(date +%T) start ollama"; for f in $B; do python3 ollama_score.py qwen3-0.6b-base-q8 "$f" "ids/$(basename "$f").ids.json"; done; echo "$(date +%T) all done" ;;
esac
