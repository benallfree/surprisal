# LASSO and the self-extracting packer (first experiment)

This was the spike's first pass, before the rules were settled. The question then was whether a context-mixing model with its decoder shipped *inside* the page can beat plain zip. It can't for small games: the decoder costs about 720 bytes zipped, and on LASSO's 9 KB script the model saves only about 590 bytes over deflate. The break-even point is roughly 12 KB of raw JS. In the archive.13k setup (see the [top-level README](../README.md)) the contest supplies the codec, so this question no longer applies.

- `index.html`: **LASSO**, a tiny neon arcade game. Steer a comet with the mouse or a finger and cross your own tail to close a loop; every enemy inside explodes. Magenta diamonds drift, orange triangles hunt you, yellow stars cut your tail, and teal hexagons split into three when caught. Touch anything and you're caught. Zips to 4,651 bytes.
- `build.sh`: zips LASSO and checks the 13,312-byte limit.
- `core.js`: the in-page model. It uses integer and IEEE arithmetic only, so a page packed in Node decodes bit-identically in any browser.
- `pack.mjs`: packs LASSO into `dist/packed.html` (self-extracting) and writes `dist/report.html`, a heatmap of the bits each character costs.

```bash
node pack.mjs
```
