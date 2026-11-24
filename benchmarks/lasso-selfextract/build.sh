#!/bin/sh
# Zips index.html into dist/game.zip and checks the js13k limit (13,312 bytes).
set -e
cd "$(dirname "$0")"
rm -rf dist && mkdir dist
cp index.html dist/
(cd dist && zip -q -9 -X game.zip index.html)
command -v advzip >/dev/null && advzip -q -z -4 dist/game.zip
size=$(wc -c < dist/game.zip | tr -d ' ')
echo "dist/game.zip: $size / 13312 bytes ($((13312 - size)) left)"
[ "$size" -le 13312 ] || { echo "OVER LIMIT"; exit 1; }
