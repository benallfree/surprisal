#!/bin/sh
# Full-file decode of the 7B 4-bit archive: new codec, then old, back to back (does the new loop slow big models at long context?).
cd "$(dirname "$0")/.."
M=llm/models/coder7b-4bit; A=llm/real/fast-7b-4bit.13k
for c in fastcodec2 fastcodec; do
  echo "$(date +%T) $c"
  .venv/bin/python llm/$c.py decode $M $A llm/ab/7b.out 2>&1 | grep -v Fetch
  cmp -s bjj-case/3-fully-minified.html llm/ab/7b.out && echo "round trip OK"
done
echo "$(date +%T) all done"
