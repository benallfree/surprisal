#!/bin/bash
# The BJJ packing matrix in ../SIZES.md: `surprisal -9 --compare` on each build, then an unpack-and-compare check.
# Needs surprisal on PATH (or SURPRISAL=/path/to/surprisal) with the 7B model, plus 7z, advzip, gzip, xz and brotli
# for the --compare rows. Surprisal -9 takes several minutes per build on a fast GPU.
#   benchmarks/bjj-case/matrix.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
surprisal=${SURPRISAL:-surprisal}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

for build in 1-readable 1b-html-minified 3-fully-minified 4-roadroller; do
  d=$work/$build
  mkdir -p "$d/src"
  cp "$here/$build.html" "$d/src/index.html" # same stored name in every archive
  echo "== $build ($(date +%T))"
  (cd "$d/src" && "$surprisal" -9 --compare -o ../out.13k index.html)
  (cd "$d" && "$surprisal" -o unpacked out.13k >/dev/null 2>&1)
  if cmp -s "$d/src/index.html" "$d/unpacked/index.html"; then echo "unpack: identical"; else echo "unpack: MISMATCH"; fi
  echo
done
