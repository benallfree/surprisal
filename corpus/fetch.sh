#!/bin/sh
# Download every entry zip listed in slugs.txt into zips/<year>/<slug>.zip.
# Polite: one request at a time with a short pause; skips files already present.
cd "$(dirname "$0")"
ok=0 skip=0 fail=0
while read -r year slugs; do
  mkdir -p "zips/$year"
  for s in $slugs; do
    f="zips/$year/$s.zip"
    if [ -s "$f" ]; then skip=$((skip+1)); continue; fi
    code=$(curl -s -m 20 -A "Mozilla/5.0 (research script)" -o "$f" -w '%{http_code}' "https://js13kgames.com/$year/games/$s.zip")
    if [ "$code" = 200 ]; then ok=$((ok+1)); else fail=$((fail+1)); rm -f "$f"; echo "$(date +%T) FAIL $code $year/$s"; fi
    [ $(( (ok+fail) % 50 )) = 0 ] && echo "$(date +%T) progress ok=$ok fail=$fail skip=$skip"
    sleep 0.2
  done
done < slugs.txt
echo "$(date +%T) done ok=$ok fail=$fail skip=$skip"
