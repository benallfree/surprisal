#!/bin/bash
# The BJJ packing matrix in ../SIZES.md: each build compressed with zip, 7z, Brotli and Surprisal -9.
# Needs 7z, advzip, brotli and surprisal on PATH (or SURPRISAL=/path/to/surprisal) with the 7B model installed.
# Prints CSV to stdout and progress to stderr. Surprisal -9 takes several minutes per build on a fast GPU.
#   benchmarks/bjj-case/matrix.sh > matrix.csv
set -u
here=$(cd "$(dirname "$0")" && pwd)
surprisal=${SURPRISAL:-surprisal}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }

echo "build,raw,zip,zip_method,7z,brotli,surprisal9,verify"
for build in 1-readable 1b-html-minified 3-fully-minified 4-roadroller; do
  d=$work/$build
  mkdir -p "$d/src"
  cp "$here/$build.html" "$d/src/index.html" # same stored name in every archive
  raw=$(size "$d/src/index.html")

  echo "$(date +%T) $build: zip, 7z, brotli" >&2
  best=999999999 method=
  for m in Deflate Deflate64 BZip2 LZMA PPMd; do
    (cd "$d/src" && 7z a -tzip -mx=9 -mm=$m "../$m.zip" index.html >/dev/null)
    s=$(size "$d/$m.zip")
    if [ "$s" -lt "$best" ]; then best=$s method=$m; fi
  done
  cp "$d/Deflate.zip" "$d/zopfli.zip"
  advzip -z -4 "$d/zopfli.zip" >/dev/null
  s=$(size "$d/zopfli.zip")
  if [ "$s" -lt "$best" ]; then best=$s method=Zopfli; fi
  (cd "$d/src" && 7z a -t7z -mx=9 -ms=on ../a.7z index.html >/dev/null)
  sevenz=$(size "$d/a.7z")
  brotli=$(brotli -q 11 -w 24 -c "$d/src/index.html" | wc -c | tr -d ' ')

  echo "$(date +%T) $build: surprisal -9" >&2
  (cd "$d/src" && "$surprisal" -9 -o ../out.13k index.html >&2)
  s9=$(size "$d/out.13k")
  echo "$(date +%T) $build: unpack and compare" >&2
  (cd "$d" && "$surprisal" -o unpacked out.13k >&2)
  if cmp -s "$d/src/index.html" "$d/unpacked/index.html"; then verify=ok; else verify=MISMATCH; fi

  echo "$build,$raw,$best,$method,$sevenz,$brotli,$s9,$verify"
  echo "$(date +%T) $build: done" >&2
done
