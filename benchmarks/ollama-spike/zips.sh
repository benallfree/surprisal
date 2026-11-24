#!/bin/sh
# Every general-purpose compressor at its best setting, on BJJ builds 1, 3 and 4. Zip sizes are whole .zip
# files holding index.html (what js13k measures); the others are raw streams.
T=$(mktemp -d)
printf '| build | zip Deflate (7-Zip) | zip Zopfli | zip Deflate64 | zip BZip2 | zip LZMA | zip PPMd (best order) | gzip -9 | bzip2 -9 | xz -9e | zstd --ultra -22 | brotli -q11 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for f in ../bjj-case/1-readable.html ../bjj-case/3-fully-minified.html ../bjj-case/4-roadroller.html; do
  cp "$f" $T/index.html; cd $T
  z() { rm -f o.zip; 7z a -tzip -bd -mx=9 "$@" o.zip index.html >/dev/null; stat -f %z o.zip; }
  d=$(z -mm=Deflate -mpass=15 -mfb=258); rm -f o.zip; 7z a -tzip -bd -mx=9 o.zip index.html >/dev/null; advzip -z -4 o.zip >/dev/null; zo=$(stat -f %z o.zip)
  d64=$(z -mm=Deflate64 -mpass=15 -mfb=257); bz=$(z -mm=BZip2 -mpass=10); lz=$(z -mm=LZMA -md=64m -mfb=273)
  pp=999999; for o in 4 6 8 12 16; do s=$(z -mm=PPMd -mo=$o -mmem=256m); [ $s -lt $pp ] && pp=$s; done
  gz=$(gzip -9c index.html | wc -c); b2=$(bzip2 -9c index.html | wc -c); xz=$(xz -9e -c --format=raw index.html | wc -c)
  zs=$(zstd -q --ultra -22 -c index.html | wc -c); br=$(brotli -c -q 11 -w 24 index.html | wc -c)
  cd - >/dev/null
  echo "| $(basename $f .html) | $d | $zo | $d64 | $bz | $lz | $pp | $gz | $b2 | $xz | $zs | $br |" | tr -s ' '
done
rm -rf $T
