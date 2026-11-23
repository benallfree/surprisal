# Model-assisted compression

**Question:** say js13k let you submit an `archive.13k` instead of a `.zip`, where the contest supplies the codec on both ends and only the archive must fit in 13,312 bytes. How much more game would fit if that codec carried a model trained on past entries?

**Answer, from 1,163 real entries:** a small language model is the best predictor by far. **Qwen2.5-Coder-0.5B fine-tuned on 2020–2024 entries**, reading each file with its full 32K-token context window, makes unseen 2025 archives **39.3% smaller than the best legal zip** for each game. At the median game, 13 KiB of archive holds what would take **31.0 KiB as the best possible zip**, well over double. Even with no js13k training at all, the off-the-shelf model reaches −37.0%. See [LLMs as the predictor](#llms-as-the-predictor).

**Without a neural network:** a hand-built context-mixing model pretrained on 2020–2024 entries makes archives **18.6% smaller than the best legal zip** for each game, and 24.8% smaller than the zips entrants actually submitted. At the median game, 13 KiB of archive holds what would take **16.5 KiB** as the best possible zip, about 27% more game. The same model with no training gives only a 7.6% saving, so most of the win comes from training. Off-the-shelf compressors with trained dictionaries reach 12–14%.

**Against every method, legal or not:** pick the best competitor for each game after the fact, from all zip methods, xz, Brotli, zstd, trained Brotli/zstd dictionaries and the untrained model. The trained model is still **5.4% smaller in total**, with a median ratio of 0.922. It wins 129 of 197 games; trained dictionaries win the rest.

## How it works

The idea comes from 3Blue1Brown's "compression is intelligence" video. Any predictor that outputs next-symbol probabilities becomes a lossless compressor through arithmetic coding, and each symbol costs about −log2 p bits, so better prediction means smaller files. The sender and receiver run the same model, so they make identical predictions, and the coded stream only has to say which way reality went.

[codec.mjs](codec.mjs) is that predictor plus a 32-bit binary arithmetic coder:

- **Context models:** byte contexts of order 0 to 12, plus the current identifier/word. Each is a hashed table of bit probabilities, kept in two copies, one fast-adapting and one slow.
- **Match model:** finds the last place the current 6+ byte context appeared and predicts the byte that came next. This is what exploits the snippets js13k games share, like zzfx, shader boilerplate and Roadroller stubs.
- **Logistic mixer:** combines the predictions, with weights selected by the partial byte and the match state.
- **Learning on the fly:** every table keeps learning while it codes, on both ends identically, which is the video's "fine-tune on the fly". **Pretraining** just runs the model over the training corpus first. The codec then ships in that warmed-up state, and each archive starts from a snapshot of it.

Nothing about the training is clever. It's the same online learning, run over 26 MB of old entries.

The LLM arms ([llm/](llm/)) use the same arithmetic-coding idea with a neural predictor, in three variants:
- **Qwen2.5-Coder-0.5B** off the shelf: internet-pretrained, never saw a js13k game.
- **A 14M-parameter byte-level transformer** trained from scratch only on 2020–2024 entries (about 45 minutes on an M1 Max with MLX).
- **Qwen with a LoRA adapter** fine-tuned on 2020–2024 entries (400 steps, about 0.4 of an epoch).

Their sizes are the sum of −log2 p over tokens, assuming bit-identical inference on both ends.

## Method

- **Corpus:** every entry's submitted zip from js13kgames.com for 2020–2025, fetched by [corpus/fetch.sh](corpus/fetch.sh) from the slugs in [corpus/slugs.txt](corpus/slugs.txt). That's 1,164 zips, of which 1,163 parse.
- **Split:** train on 2020–2024 (966 games, 26 MB unpacked), test on 2025 (197 games). Test games are never seen in training.
- **Archive content:** a game's files concatenated with names and lengths ([zipread.mjs](zipread.mjs)). Every method compresses the same bytes.
- **Opponents, each at its best settings:**
  - Legal zips with every method 7-Zip can write: Deflate, Deflate64, BZip2, LZMA, and PPMd at orders 4–16.
  - Deflate recompressed with Google's Zopfli via `advzip -4`.
  - The entrant's own submitted zip.
  - Raw xz, Brotli and zstd, alone and with a dictionary trained on the same 2020–2024 data.
- **Scope: final submissions, not source.** Each entry's GitHub repo is mirrored under `github.com/js13kGames/<slug>`, and about 96% have readable source ([corpus/survey.mjs](corpus/survey.mjs)). But a source tree doesn't run until you rebuild it, and each project builds its assets differently. The submitted zip is the one version of each game known to work, assets included, so that's what every method compresses.
- **Roadroller entries stay in.** Roadroller packs JS into a string that is nearly incompressible for every method here. It accounts for 55 of the 197 test games (28%) and 174 of 966 training games (18%), so it holds every method's numbers down. Those rows are marked RR in the table.
- **Model sizes** are ideal arithmetic-coding sizes plus a 6-byte header. Full encode/decode round-trips on sample games land within 2 bytes of the estimate.

## Findings

1. **A pretrained LLM is the strongest predictor, by a wide margin.** On readable code (the 142 non-Roadroller games), off-the-shelf Qwen gets 1.09 bits per byte, against 1.87 for the trained context mixer and 2.39 for Brotli. The fine-tuned model reaches 1.05. General knowledge of code beats anything learnable from 24 MB of js13k entries.
2. **js13k training still adds something on top.** Fine-tuning Qwen on old entries saved another 4% overall (−37.0% → −39.3% vs the best zip with the full 32K window; −35.6% → −38.1% with a 2K window). The fine-tuned model is smaller on 171 of 197 test games at 32K (179 at 2K). Its validation loss was noisy along the way, rising before it fell, so read this as a modest, consistent gain from a short, untuned fine-tune.
3. **Small trained-from-scratch beats hand-built.** The 14M transformer, trained only on js13k data, beats the context-mixing model trained on exactly the same data by 11% on text (17.5% on readable code). It's also the only predictor that helps on Roadroller text, by about 2%, presumably because it learned Roadroller's output statistics, which a code model never saw. Per game it beats every non-neural method on 188 of 197 games.
4. **Roadroller defeats everything.** Packed Roadroller strings cost about 5.5 bits per byte under every method. They're already compressed, and no predictor can see through them.
5. **Training is what matters for the context mixer.** The untrained model beats the best zip by 7.6%. A model trained on one year of entries reaches 15.7%, and five years reach 18.6%. Returns diminish fast, and most of the gain comes from the first year.
6. **The trained context mixer beats trained dictionaries by about 5 points.** Brotli and zstd with the same training data get −13.6% and −12.2%. A dictionary only helps where whole strings repeat exactly; the model also learns statistics: which tokens follow which, and how minified identifiers get reused. The model gives the smallest archive for 129 of 197 games. Trained Brotli wins the other 66, mostly games that reuse large chunks of earlier entries, which a 16 MB raw dictionary can copy outright. One example is carbons-kitty-care: 72 KiB unpacked, 2.9 KiB with a trained dictionary.
7. **Entrants leave about 8% on the table with zip alone.** Re-zipping each game with the best legal method (PPMd is the strongest single one) comes out 7.7% smaller than the submitted zips in total. Some entrants already use LZMA and PPMd zips.
8. **Brotli is the best untrained general-purpose compressor here** (−5.3% vs the best zip), helped by its built-in web dictionary. xz roughly ties the best zip, and zstd loses to it.
9. **The fine print, in the Hutter Prize spirit:** the trained context mixer's state is about 250 MB, and Qwen is about 1 GB of weights. "Free decoder" rules make that fine. Count the codec's size, as the Hutter Prize does, and none of this wins. The float LLM numbers also assume both ends run bit-identical inference, which in practice means the same GPU type and kernels. `llm/intref.py` removes that: an integer-only forward pass whose archives decode on any hardware with the same model bundle, at no cost in size (7,185 vs 7,183 bytes on the BJJ file) and decompresses the BJJ file in 29.6 s on an H100 (see TIMING.md). The integer codec is frozen as **spec v1** in `SPEC.md` (model manifest, deterministic conversion, archive format, test vectors). **signal** (`cmd/signal`) is the command-line compressor built on it: `signal -c files…` / `signal -d archive.13k`, with nollama running the integer model.
10. **2025 entries were already squeezed to fit 13 KB.** Most are minified and some are Roadroller-packed, which leaves less redundancy. On readable source the gap would be larger.

## Speed: Python reference vs signal

Wall time on the BJJ file (55,854 bytes, 24,126 tokens) with each model of spec v1. Every archive and every decode is
byte-identical across both implementations and both machines. Python is `llm/intref.py` (numpy) on the Mac and
`llm/intgpu.py` (Triton) on the H100; signal is `cmd/signal` with nollama (Metal on the Mac). Compression runs every
position at once (batched); decompression has to go one token at a time.

<!-- speed-table: keep updated when either implementation gets faster -->
<table>
<tr><th>model</th><th>M1 Max (Mac)</th><th>H100</th></tr>
<tr><td><b>0.5B</b></td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">~58 min*</td><td align="right">~58 min*</td></tr>
<tr><td>signal</td><td align="right">53 s</td><td align="right">3 min 52 s</td></tr>
</table>

</td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">2.0 s</td><td align="right">30.0 s</td></tr>
<tr><td>signal</td><td align="right">no CUDA yet</td><td align="right">no CUDA yet</td></tr>
</table>

</td></tr>
<tr><td><b>1.5B</b></td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">~2.9 h*</td><td align="right">~2.9 h*</td></tr>
<tr><td>signal</td><td align="right">2 min 10 s</td><td align="right">measuring</td></tr>
</table>

</td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">3.8 s</td><td align="right">54.4 s</td></tr>
<tr><td>signal</td><td align="right">no CUDA yet</td><td align="right">no CUDA yet</td></tr>
</table>

</td></tr>
<tr><td><b>7B</b></td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">~3.5 days*</td><td align="right">~3.5 days*</td></tr>
<tr><td>signal</td><td align="right">7 min 30 s</td><td align="right">measuring</td></tr>
</table>

</td><td>

<table>
<tr><th></th><th>compress</th><th>decompress</th></tr>
<tr><td>Python</td><td align="right">9.3 s</td><td align="right">123.4 s</td></tr>
<tr><td>signal</td><td align="right">no CUDA yet</td><td align="right">no CUDA yet</td></tr>
</table>

</td></tr>
</table>

\* Estimated from the numpy reference's speed over the first 300, 120 and 30 tokens (6.9, 2.3 and 0.08 tok/s); later
tokens are slower, and the 7B doesn't fit in memory and swaps. signal on the H100 needs nollama's CUDA backend for
integer models, in progress.

## Reproduce

```bash
corpus/fetch.sh
```

```bash
node --max-old-space-size=4096 bench.mjs
```

```bash
node report.mjs
```

LLM arms (needs Python 3.12 and an Apple Silicon Mac for MLX; downloads Qwen2.5-Coder-0.5B, about 1 GB, from Hugging Face):

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python mlx-lm
```

```bash
node --max-old-space-size=4096 llm/split.mjs && node llm/dump_train.mjs && .venv/bin/python llm/ft_data.py
```

```bash
llm/run_rest.sh
```

```bash
node llm/summary.mjs
```

`run_rest.sh` scores pretrained Qwen, trains and scores the from-scratch transformer, then LoRA fine-tunes and scores Qwen: about 1¾ hours in total. `summary.mjs` writes the LLM section below.

`fetch.sh` downloads the corpus (about 15 MB, one request at a time). `bench.mjs` takes about 18 minutes and writes `results.md` and `results.json`. `report.mjs` regenerates the table below. Requires Node 20+, `zstd`, `brotli`, `xz`, `7z`, `zopfli` and `advzip` (all available from Homebrew).

## Files

- [codec.mjs](codec.mjs): the model and the arithmetic coder (`Model`, `encode`, `decode`, `cost`, `pretrain`, `snapshot`/`restore`).
- [bench.mjs](bench.mjs): the benchmark. [report.mjs](report.mjs) builds the README table from its output.
- [zipread.mjs](zipread.mjs): reads zips in memory, with a 7-Zip fallback for LZMA/PPMd/Deflate64 zips.
- [corpus/](corpus/): the slug list and fetch script. The downloaded zips are git-ignored.
- [llm/](llm/): the LLM arms. `split.mjs` splits archives into text and binary, `score.py` scores text with any MLX language model (optionally with a LoRA adapter), `scratch.py` trains and scores the from-scratch transformer, `ft_data.py` builds the LoRA data, `run_rest.sh` chains it all, and `summary.mjs` writes the README section.
- [bjj-case/](bjj-case/): the BJJ builds (readable, JS-minified, fully minified with my minifier and with html-minifier-terser, Roadroller) and the small HTML/CSS minifier.
- [lasso-selfextract/](lasso-selfextract/): the first experiment, a self-extracting page with the decoder counted against the limit.

## Next ideas

- Try bigger models (Qwen2.5-Coder-1.5B and 3B). The video's point that better prediction means better compression should keep paying off.
- Fine-tune longer, on readable entries only, with a proper validation set. The 0.4-epoch LoRA left headroom.
- Let the LLM keep learning while it codes, with deterministic test-time updates, the video's "fine-tune on the fly".
- Find the size crossover beyond 32K tokens: score a 1–10 MB readable corpus against Brotli, xz and the context mixer, carrying context past the window with a sliding KV cache.
- Byte-level hybrid: mix the LLM's prediction with the context mixer's match model, so copied snippets and Roadroller text are handled by whichever is better.
- Train on readable source from the entries' GitHub repos, and pack source instead of minified output.
- Model images separately (decode PNGs and model the pixels) for asset-heavy games.

<!-- llm:start -->
## LLMs as the predictor

The same arithmetic-coding idea, with a neural language model as the predictor. LLM tokenizers only handle text, so each 2025 archive is split into its **text part** (UTF-8 code and markup, 4.63 MB over 197 games) and its **binary part** (images, audio: 0.26 MB). The tables below compare every method on exactly the same text. LLM sizes are the sum of −log2 p over tokens, which is the arithmetic coder's output size, assuming both ends run bit-identical inference. The first three LLM rows score long texts in 2,048-token windows; the 32K rows use Qwen's full 32,768-token context window, which holds all but one test game whole.

### Text part only, all: 197 games, 4525 KiB

| predictor | total KiB | bits per byte | vs trained context mixing |
|---|---:|---:|---:|
| brotli -q11 | 1708.3 | 3.020 | 18.1% |
| brotli + trained dict | 1549.8 | 2.740 | 7.2% |
| context mixing, untrained | 1663.0 | 2.940 | 15.0% |
| context mixing, trained on 2020-24 | 1446.1 | 2.557 | 0.0% |
| Qwen2.5-Coder-0.5B, pretrained, no js13k training | 1114.4 | 1.970 | -22.9% |
| 14M byte-level transformer, trained from scratch on js13k 2020-24 only | 1282.6 | 2.268 | -11.3% |
| Qwen2.5-Coder-0.5B, fine-tuned on js13k 2020-24 | 1065.3 | 1.883 | -26.3% |
| Qwen2.5-Coder-0.5B, pretrained, full 32K window | 1086.5 | 1.921 | -24.9% |
| Qwen2.5-Coder-0.5B, fine-tuned, full 32K window | 1041.9 | 1.842 | -28.0% |

### Text part only, readable (not Roadroller): 142 games, 3684 KiB

| predictor | total KiB | bits per byte | vs trained context mixing |
|---|---:|---:|---:|
| brotli -q11 | 1101.2 | 2.391 | 27.7% |
| brotli + trained dict | 969.4 | 2.105 | 12.4% |
| context mixing, untrained | 1046.3 | 2.272 | 21.3% |
| context mixing, trained on 2020-24 | 862.3 | 1.872 | 0.0% |
| Qwen2.5-Coder-0.5B, pretrained, no js13k training | 500.7 | 1.087 | -41.9% |
| 14M byte-level transformer, trained from scratch on js13k 2020-24 only | 711.1 | 1.544 | -17.5% |
| Qwen2.5-Coder-0.5B, fine-tuned on js13k 2020-24 | 482.2 | 1.047 | -44.1% |
| Qwen2.5-Coder-0.5B, pretrained, full 32K window | 473.9 | 1.029 | -45.0% |
| Qwen2.5-Coder-0.5B, fine-tuned, full 32K window | 458.8 | 0.996 | -46.8% |

### Text part only, Roadroller-packed: 55 games, 841 KiB

| predictor | total KiB | bits per byte | vs trained context mixing |
|---|---:|---:|---:|
| brotli -q11 | 607.2 | 5.778 | 4.0% |
| brotli + trained dict | 580.4 | 5.524 | -0.6% |
| context mixing, untrained | 616.6 | 5.868 | 5.6% |
| context mixing, trained on 2020-24 | 583.9 | 5.556 | 0.0% |
| Qwen2.5-Coder-0.5B, pretrained, no js13k training | 613.8 | 5.841 | 5.1% |
| 14M byte-level transformer, trained from scratch on js13k 2020-24 only | 571.5 | 5.439 | -2.1% |
| Qwen2.5-Coder-0.5B, fine-tuned on js13k 2020-24 | 583.1 | 5.549 | -0.1% |
| Qwen2.5-Coder-0.5B, pretrained, full 32K window | 612.5 | 5.829 | 4.9% |
| Qwen2.5-Coder-0.5B, fine-tuned, full 32K window | 583.1 | 5.549 | -0.1% |

### Whole archive (LLM for text, trained context mixing for binary), all 197 games

| method | total KiB | vs best legal zip |
|---|---:|---:|
| best legal zip per game | 1961.0 | 0.0% |
| brotli + trained dict | 1694.5 | -13.6% |
| context mixing, trained (main result) | 1596.3 | -18.6% |
| Qwen2.5-Coder-0.5B, pretrained, no js13k training + CM for binary | 1263.2 | -35.6% |
| 14M byte-level transformer, trained from scratch on js13k 2020-24 only + CM for binary | 1431.3 | -27.0% |
| Qwen2.5-Coder-0.5B, fine-tuned on js13k 2020-24 + CM for binary | 1214.1 | -38.1% |
| Qwen2.5-Coder-0.5B, pretrained, full 32K window + CM for binary | 1235.2 | -37.0% |
| Qwen2.5-Coder-0.5B, fine-tuned, full 32K window + CM for binary | 1190.6 | -39.3% |

<!-- llm:end -->

## Case study: the BJJ spike (one big readable game)

The `bjj-leverage` spike (a 3D jiu-jitsu game that loads Three.js from a CDN, allowed only in js13k's WebXR category) is 67 KB of readable HTML and JS. Five builds, all in [bjj-case/](bjj-case/), and each one runs. Qwen is scored with its full 32,768-token context window, which holds the whole file (the readable source is 26,528 tokens). The 2K-window column is the earlier, shorter setting, kept for comparison.

| build | raw | best legal zip | brotli | context mixing, trained | Qwen, 32K window | Qwen fine-tuned, 32K window | Qwen, 2K window |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1. readable source | 67,056 | 21,763 (PPMd) | 22,758 | 19,449 | **7,229** | 7,355 | 8,378 |
| 2. JS minified (terser) | 55,926 | 19,276 | 19,695 | 16,573 | 7,197 | 7,079 | 8,146 |
| 3. JS + HTML + CSS minified | 55,854 | 19,247 | 19,723 | 16,537 | 7,187 | 7,071 | 8,134 |
| 3b. same, with html-minifier-terser | 55,812 | 19,230 | 19,722 | 16,541 | 7,187 | **7,069** | 8,140 |
| 4. build 3, JS packed with Roadroller | 27,673 | 18,861 (Zopfli) | 18,427 | 17,703 | 17,883 | 17,115 | 17,937 |

- **Under today's rules it doesn't fit.** The best real js13k build (minify + Roadroller + zip) is 18,861 bytes, 42% over the limit.
- **With an LLM codec it fits easily, unminified.** Qwen compresses the readable source to 7.2 KB, 55% of the limit and a third of the best zip.
- **Minifying is pointless under an LLM.** It saves 12–15% for zip, Brotli and the context mixer, but only 0.6% for Qwen with the full window (3% with the 2K window). Once Qwen sees the whole file, readable names and comments are almost free: it predicts them from their first use.
- **The js13k fine-tune helps minified code and hurts readable code.** Its training data, past js13k entries, is mostly minified, so the fine-tuned Qwen does 1.6% better on the minified builds and 1.7% worse on the readable source. With it, minifying pays off a little (7,355 → 7,069 bytes) but stays far below what it gains for zip and Brotli.
- **Roadroller and the LLM don't mix.** Roadroller is itself a context-mixing compressor, so its output is noise to any later model: Qwen goes from 7.2 KB to 17.9 KB, and the longer window doesn't help. Under an LLM codec you'd ship the readable source.
- **The HTML and CSS were already tight.** A whitespace and comment minifier ([bjj-case/minhtml.mjs](bjj-case/minhtml.mjs)) saved only 72 bytes. html-minifier-terser 7.2.0 with every safe option (build 3b) saved 114, but after compression the two builds are within 17 bytes of each other on every method. Build 4 is Roadroller applied to build 3.

### Does the LLM's advantage hold as files grow?

[llm/sizecurve.py](llm/sizecurve.py) compresses growing prefixes of the readable BJJ source (`LLM_WINDOW=32768` for the full window):

| prefix | Qwen, 32K window | context mixing | brotli | Qwen / brotli, 32K | Qwen / brotli, 2K |
|---:|---:|---:|---:|---:|---:|
| 2 KB | 208 | 748 | 698 | 0.30 | 0.30 |
| 4 KB | 379 | 1,243 | 1,243 | 0.30 | |
| 8 KB | 891 | 2,847 | 2,908 | 0.31 | 0.31 |
| 16 KB | 1,806 | 5,785 | 6,024 | 0.30 | 0.31 |
| 32 KB | 3,812 | 10,971 | 11,767 | 0.32 | 0.36 |
| 67 KB | 7,229 | 20,805 | 22,758 | 0.32 | 0.37 |

With the full window the LLM's edge barely moves: about 0.30–0.32 of Brotli's size from 2 KB to 67 KB. The shrinking edge seen with the 2K window was mostly the window itself: past ~7 KB of code the model could no longer see the start of the file, while Brotli could. There's no sign of an inflection point inside 32K tokens. Finding out what happens beyond it (1–10 MB) needs a long readable corpus and a way to carry context past the window, such as a sliding KV cache.


## Results

<!-- results:start -->
### 2025 test set, all: 197 games, 4782 KiB unpacked

| codec | total KiB | vs submitted zip | vs best legal zip |
|---|---:|---:|---:|
| submitted .zip (as entered) | 2123.7 | 0.0% | 8.3% |
| zip: Deflate (7-Zip, 15 passes) | 2095.2 | -1.3% | 6.8% |
| zip: Deflate via Zopfli (advzip -4) | 2053.8 | -3.3% | 4.7% |
| zip: Deflate64 | 2094.1 | -1.4% | 6.8% |
| zip: BZip2 | 2144.0 | 1.0% | 9.3% |
| zip: LZMA | 2079.4 | -2.1% | 6.0% |
| zip: PPMd (best order) | 2008.6 | -5.4% | 2.4% |
| **best legal zip per game** | 1961.0 | -7.7% | 0.0% |
| zopfli raw deflate stream (no zip headers) | 1960.3 | -7.7% | -0.0% |
| xz -9e (raw lzma2) | 1946.5 | -8.3% | -0.7% |
| brotli -q11 -w24 | 1857.3 | -12.5% | -5.3% |
| zstd -19 | 2002.5 | -5.7% | 2.1% |
| brotli + trained dict (best of 1MB zstd-trained, 16MB raw) | 1694.5 | -20.2% | -13.6% |
| zstd + trained dict (best of 1MB trained, 16MB raw --long) | 1722.6 | -18.9% | -12.2% |
| model, untrained | 1812.4 | -14.7% | -7.6% |
| model, pretrained 1 yr (2020-2020) | 1652.1 | -22.2% | -15.7% |
| model, pretrained 2 yrs (2020-2021) | 1623.5 | -23.6% | -17.2% |
| model, pretrained 3 yrs (2020-2022) | 1606.7 | -24.3% | -18.1% |
| model, pretrained 4 yrs (2020-2023) | 1606.4 | -24.4% | -18.1% |
| model, pretrained 5 yrs (2020-2024) | 1596.3 | -24.8% | -18.6% |

### 2025 test set, mostly code (<20% binary assets): 184 games, 4560 KiB unpacked

| codec | total KiB | vs submitted zip | vs best legal zip |
|---|---:|---:|---:|
| submitted .zip (as entered) | 1980.1 | 0.0% | 8.6% |
| zip: Deflate (7-Zip, 15 passes) | 1953.5 | -1.3% | 7.1% |
| zip: Deflate via Zopfli (advzip -4) | 1916.5 | -3.2% | 5.1% |
| zip: Deflate64 | 1952.4 | -1.4% | 7.0% |
| zip: BZip2 | 1998.0 | 0.9% | 9.5% |
| zip: LZMA | 1936.6 | -2.2% | 6.2% |
| zip: PPMd (best order) | 1868.5 | -5.6% | 2.4% |
| **best legal zip per game** | 1824.1 | -7.9% | 0.0% |
| zopfli raw deflate stream (no zip headers) | 1837.0 | -7.2% | 0.7% |
| xz -9e (raw lzma2) | 1822.6 | -8.0% | -0.1% |
| brotli -q11 -w24 | 1738.1 | -12.2% | -4.7% |
| zstd -19 | 1872.6 | -5.4% | 2.7% |
| brotli + trained dict (best of 1MB zstd-trained, 16MB raw) | 1582.4 | -20.1% | -13.3% |
| zstd + trained dict (best of 1MB trained, 16MB raw --long) | 1606.6 | -18.9% | -11.9% |
| model, untrained | 1692.8 | -14.5% | -7.2% |
| model, pretrained 1 yr (2020-2020) | 1538.3 | -22.3% | -15.7% |
| model, pretrained 2 yrs (2020-2021) | 1511.2 | -23.7% | -17.2% |
| model, pretrained 3 yrs (2020-2022) | 1495.1 | -24.5% | -18.0% |
| model, pretrained 4 yrs (2020-2023) | 1494.8 | -24.5% | -18.1% |
| model, pretrained 5 yrs (2020-2024) | 1485.7 | -25.0% | -18.5% |

### 2025 test set, asset-heavy (>=20% binary): 13 games, 222 KiB unpacked

| codec | total KiB | vs submitted zip | vs best legal zip |
|---|---:|---:|---:|
| submitted .zip (as entered) | 143.7 | 0.0% | 5.0% |
| zip: Deflate (7-Zip, 15 passes) | 141.7 | -1.4% | 3.5% |
| zip: Deflate via Zopfli (advzip -4) | 137.3 | -4.4% | 0.3% |
| zip: Deflate64 | 141.7 | -1.4% | 3.5% |
| zip: BZip2 | 146.0 | 1.7% | 6.7% |
| zip: LZMA | 142.8 | -0.6% | 4.4% |
| zip: PPMd (best order) | 140.1 | -2.5% | 2.4% |
| **best legal zip per game** | 136.9 | -4.7% | 0.0% |
| zopfli raw deflate stream (no zip headers) | 123.3 | -14.2% | -9.9% |
| xz -9e (raw lzma2) | 123.9 | -13.7% | -9.5% |
| brotli -q11 -w24 | 119.2 | -17.0% | -12.9% |
| zstd -19 | 129.9 | -9.6% | -5.1% |
| brotli + trained dict (best of 1MB zstd-trained, 16MB raw) | 112.2 | -21.9% | -18.0% |
| zstd + trained dict (best of 1MB trained, 16MB raw --long) | 116.0 | -19.3% | -15.3% |
| model, untrained | 119.6 | -16.7% | -12.6% |
| model, pretrained 1 yr (2020-2020) | 113.8 | -20.8% | -16.8% |
| model, pretrained 2 yrs (2020-2021) | 112.3 | -21.8% | -17.9% |
| model, pretrained 3 yrs (2020-2022) | 111.6 | -22.3% | -18.4% |
| model, pretrained 4 yrs (2020-2023) | 111.6 | -22.3% | -18.4% |
| model, pretrained 5 yrs (2020-2024) | 110.5 | -23.1% | -19.2% |

Median per-game ratio vs best legal zip: model untrained 0.925, pretrained 5 yrs 0.789, brotli+dict 0.885, zstd+dict 0.906
Capacity: a 13,312-byte archive.13k holds about 16.5 KiB of best-zip-equivalent game with the pretrained model.

Smallest archive per game (wins): brotli +dict 66, zstd +dict 1, model 1, model +trained 129.

Against the best of every other method on each game (an oracle that picks the best competitor per game after the fact), the trained model is -5.4% in total, with a median per-game ratio of 0.922, and it is smaller on 129 of 197 games.

### Every 2025 game, every method

All sizes in KiB (1,024 bytes) for the whole game. The bold cell in each row is the smallest. "zip …" columns are legal js13k zips
built with 7-Zip at max settings (PPMd: best of orders 4-16; Zopfli via `advzip -z -4`). "+dict" means a dictionary trained on
2020-2024 entries, best of a 1 MB `zstd --train` dictionary and the raw last 16 MB of training data. "model" is the codec in
[codec.mjs](codec.mjs) untrained; "model +trained" is the same codec after pretraining on all 2020-2024 entries.
"smallest" names the row's winner. The last column compares the trained model with the best of **every other method** on that game
(all zips, xz, brotli, zstd, both trained dictionaries, and the untrained model), so a positive number means some other method won that game.
RR marks games whose submission is Roadroller-packed. Rows are sorted by the last column.

| # | game | unpacked KiB | submitted zip | zip Deflate | zip Zopfli | zip Deflate64 | zip BZip2 | zip LZMA | zip PPMd | xz -9e | brotli | brotli +dict | zstd | zstd +dict | model | model +trained | smallest | model +trained vs best other method |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | [cat-vs-the-world](https://js13kgames.com/2025/games/cat-vs-the-world) | 25.08 | 7.03 | 6.78 | 6.73 | 6.78 | 6.63 | 6.60 | 5.73 | 6.45 | 5.88 | 5.44 | 6.63 | 5.42 | 5.48 | **4.33** | model +trained | -20.1% |
| 2 | [a-qr-maze](https://js13kgames.com/2025/games/a-qr-maze) | 13.56 | 3.53 | 3.47 | 3.40 | 3.47 | 3.42 | 3.44 | 2.96 | 3.17 | 2.78 | 2.46 | 3.22 | 2.35 | 2.68 | **1.88** | model +trained | -20.0% |
| 3 | [cat-hop-cloud](https://js13kgames.com/2025/games/cat-hop-cloud) | 2.36 | 1.27 | 1.24 | 1.21 | 1.24 | 1.33 | 1.28 | 1.17 | 1.13 | 0.92 | 0.74 | 1.14 | 0.75 | 0.98 | **0.59** | model +trained | -20.0% |
| 4 | [a-hungry-black](https://js13kgames.com/2025/games/a-hungry-black) | 40.18 | 9.85 | 9.56 | 9.51 | 9.55 | 9.73 | 9.24 | 8.33 | 9.10 | 8.44 | 7.68 | 9.40 | 7.57 | 7.69 | **6.07** | model +trained | -19.8% |
| 5 | [midnight-heist](https://js13kgames.com/2025/games/midnight-heist) | 21.98 | 7.20 | 6.02 | 5.95 | 6.02 | 6.16 | 5.87 | 5.21 | 5.56 | 5.09 | 4.36 | 5.69 | 4.22 | 4.71 | **3.40** | model +trained | -19.6% |
| 6 | [fateful-encounter](https://js13kgames.com/2025/games/fateful-encounter) | 4.97 | 2.89 | 2.71 | 2.53 | 2.71 | 2.90 | 2.87 | 2.54 | 2.00 | 1.71 | 1.50 | 1.98 | 1.45 | 1.73 | **1.17** | model +trained | -19.4% |
| 7 | [chickpocalypse-meow](https://js13kgames.com/2025/games/chickpocalypse-meow) | 42.12 | 10.08 | 10.08 | 9.92 | 10.06 | 9.75 | 9.75 | 8.47 | 9.19 | 8.67 | 7.88 | 9.45 | 7.76 | 7.72 | **6.24** | model +trained | -19.1% |
| 8 | [little-adventure-of-mr-cat](https://js13kgames.com/2025/games/little-adventure-of-mr-cat) | 14.90 | 4.85 | 4.73 | 4.67 | 4.73 | 4.62 | 4.60 | 4.04 | 4.45 | 4.00 | 3.71 | 4.57 | 3.69 | 3.80 | **3.00** | model +trained | -18.9% |
| 9 | [shadow-prowler](https://js13kgames.com/2025/games/shadow-prowler) | 11.27 | 4.00 | 4.00 | 3.89 | 4.00 | 4.09 | 4.06 | 3.56 | 3.50 | 3.00 | 2.73 | 3.55 | 2.63 | 3.04 | **2.13** | model +trained | -18.8% |
| 10 | [cat-astrophe](https://js13kgames.com/2025/games/cat-astrophe) | 38.77 | 12.99 | 12.58 | 12.34 | 12.58 | 12.48 | 12.19 | 10.93 | 11.51 | 10.78 | 9.99 | 12.01 | 10.08 | 9.96 | **8.17** | model +trained | -18.0% |
| 11 | [typing-test](https://js13kgames.com/2025/games/typing-test) | 28.33 | 10.87 | 10.50 | 10.33 | 10.50 | 10.83 | 10.28 | 9.61 | 9.80 | 9.14 | 7.96 | 10.08 | 8.01 | 8.60 | **6.53** | model +trained | -17.9% |
| 12 | [cash-cats-13](https://js13kgames.com/2025/games/cash-cats-13) | 44.87 | 7.75 | 7.19 | 7.09 | 7.18 | 6.82 | 6.90 | 5.81 | 6.76 | 6.26 | 5.90 | 6.96 | 5.82 | 5.68 | **4.69** | model +trained | -17.4% |
| 13 | [memorygame](https://js13kgames.com/2025/games/memorygame) | 40.72 | 12.19 | 11.63 | 11.31 | 11.35 | 11.29 | 11.27 | 9.92 | 9.47 | 8.46 | 7.82 | 9.81 | 7.74 | 8.16 | **6.42** | model +trained | -17.1% |
| 14 | [cat-rooftop-adventure](https://js13kgames.com/2025/games/cat-rooftop-adventure) | 9.19 | 3.65 | 3.29 | 3.25 | 3.29 | 3.46 | 3.25 | 2.97 | 3.11 | 2.72 | 2.27 | 3.18 | 2.20 | 2.71 | **1.83** | model +trained | -17.0% |
| 15 | [shadow-network](https://js13kgames.com/2025/games/shadow-network) | 38.15 | 9.35 | 8.68 | 8.60 | 8.58 | 8.30 | 8.39 | 7.13 | 8.24 | 7.54 | 7.11 | 8.40 | 6.95 | 7.02 | **5.78** | model +trained | -16.9% |
| 16 | [mr13lackie](https://js13kgames.com/2025/games/mr13lackie) | 7.10 | 3.27 | 3.22 | 3.18 | 3.22 | 3.29 | 3.26 | 2.94 | 3.11 | 2.78 | 2.34 | 3.12 | 2.30 | 2.71 | **1.91** | model +trained | -16.9% |
| 17 | [cat-and-spikes](https://js13kgames.com/2025/games/cat-and-spikes) | 15.05 | 4.15 | 4.04 | 3.95 | 4.04 | 4.17 | 3.93 | 3.55 | 3.63 | 3.26 | 3.06 | 3.73 | 3.05 | 3.20 | **2.54** | model +trained | -16.7% |
| 18 | [schrodingers-cat-walk](https://js13kgames.com/2025/games/schrodingers-cat-walk) | 28.32 | 12.16 | 8.81 | 8.75 | 8.81 | 8.97 | 8.61 | 7.74 | 8.46 | 7.84 | 7.33 | 8.64 | 7.27 | 7.23 | **6.02** | model +trained | -16.7% |
| 19 | [black-hat-black-cat](https://js13kgames.com/2025/games/black-hat-black-cat) | 35.48 | 11.50 | 11.23 | 11.03 | 11.23 | 11.63 | 10.98 | 10.08 | 10.31 | 9.64 | 8.39 | 10.61 | 8.24 | 9.13 | **6.87** | model +trained | -16.6% |
| 20 | [black-cats-omen](https://js13kgames.com/2025/games/black-cats-omen) | 29.92 | 10.34 | 10.42 | 10.12 | 10.42 | 10.99 | 10.66 | 9.38 | 6.78 | 6.19 | 5.65 | 6.92 | 5.57 | 5.90 | **4.67** | model +trained | -16.1% |
| 21 | [9-lives-an-escape-room](https://js13kgames.com/2025/games/9-lives-an-escape-room) | 48.34 | 10.42 | 10.40 | 10.00 | 10.41 | 10.93 | 10.27 | 9.22 | 7.90 | 7.32 | 6.65 | 8.18 | 6.74 | 6.81 | **5.58** | model +trained | -16.0% |
| 22 | [blackhats-black-cats](https://js13kgames.com/2025/games/blackhats-black-cats) | 37.22 | 12.49 | 12.48 | 12.34 | 12.48 | 12.74 | 12.24 | 11.15 | 12.09 | 11.39 | 10.61 | 12.50 | 10.69 | 10.51 | **8.86** | model +trained | -15.7% |
| 23 | [snowball](https://js13kgames.com/2025/games/snowball) | 11.90 | 4.36 | 4.29 | 4.15 | 4.29 | 4.61 | 4.35 | 3.99 | 3.64 | 3.27 | 2.56 | 3.69 | 2.51 | 3.28 | **2.12** | model +trained | -15.4% |
| 24 | [byte-armada](https://js13kgames.com/2025/games/byte-armada) | 14.57 | 6.92 | 6.78 | 6.62 | 6.78 | 7.13 | 6.93 | 6.32 | 5.82 | 5.37 | 4.64 | 5.87 | 4.53 | 5.07 | **3.83** | model +trained | -15.4% |
| 25 | [darkline-paws](https://js13kgames.com/2025/games/darkline-paws) | 33.87 | 12.75 | 12.22 | 12.01 | 12.22 | 12.82 | 11.99 | 11.16 | 9.93 | 9.45 | 8.37 | 10.46 | 8.31 | 8.89 | **7.03** | model +trained | -15.3% |
| 26 | [the-black-cats-escape](https://js13kgames.com/2025/games/the-black-cats-escape) | 36.19 | 10.82 | 10.47 | 10.34 | 10.47 | 10.62 | 10.05 | 9.07 | 9.58 | 9.08 | 8.42 | 9.89 | 8.42 | 8.44 | **7.13** | model +trained | -15.3% |
| 27 | [catrooms](https://js13kgames.com/2025/games/catrooms) | 24.13 | 6.53 | 6.24 | 6.08 | 6.24 | 6.44 | 6.26 | 5.58 | 5.54 | 5.24 | 4.89 | 5.61 | 4.76 | 4.75 | **4.02** | model +trained | -15.3% |
| 28 | [wake-the-cat](https://js13kgames.com/2025/games/wake-the-cat) | 35.26 | 10.41 | 10.52 | 10.44 | 10.50 | 10.34 | 10.04 | 9.07 | 9.88 | 9.41 | 8.49 | 10.34 | 8.72 | 8.48 | **7.19** | model +trained | -15.2% |
| 29 | [mewsterpiece](https://js13kgames.com/2025/games/mewsterpiece) | 37.52 | 12.18 | 12.32 | 12.19 | 12.32 | 12.71 | 11.92 | 10.96 | 11.35 | 10.80 | 9.75 | 11.82 | 9.74 | 10.09 | **8.26** | model +trained | -15.2% |
| 30 | [cat-black-snack](https://js13kgames.com/2025/games/cat-black-snack) | 8.76 | 3.58 | 3.69 | 3.54 | 3.69 | 3.96 | 3.83 | 3.43 | 3.02 | 2.71 | 1.61 | 3.10 | 1.57 | 2.67 | **1.33** | model +trained | -14.9% |
| 31 | [crazy-cat-crossing](https://js13kgames.com/2025/games/crazy-cat-crossing) | 3.99 | 3.68 | 2.12 | 2.01 | 2.12 | 2.20 | 2.17 | 1.97 | 1.67 | 1.48 | 1.27 | 1.68 | 1.25 | 1.45 | **1.06** | model +trained | -14.8% |
| 32 | [catnfish](https://js13kgames.com/2025/games/catnfish) | 16.52 | 5.30 | 5.22 | 5.08 | 5.22 | 5.56 | 5.16 | 4.81 | 4.46 | 4.16 | 3.53 | 4.60 | 3.48 | 4.05 | **2.97** | model +trained | -14.7% |
| 33 | [clawopatra-and-the-gods-of-egypt](https://js13kgames.com/2025/games/clawopatra-and-the-gods-of-egypt) | 40.76 | 11.90 | 10.98 | 10.76 | 10.97 | 11.64 | 10.61 | 9.67 | 10.46 | 9.71 | 8.42 | 10.77 | 8.30 | 9.07 | **7.09** | model +trained | -14.6% |
| 34 | [black-cats-rise](https://js13kgames.com/2025/games/black-cats-rise) | 34.39 | 12.82 | 12.80 | 12.74 | 12.80 | 13.18 | 12.51 | 11.64 | 12.37 | 11.64 | 9.80 | 12.71 | 9.71 | 10.82 | **8.30** | model +trained | -14.5% |
| 35 | [tower-of-terror](https://js13kgames.com/2025/games/tower-of-terror) | 41.54 | 10.87 | 10.75 | 10.39 | 10.75 | 11.25 | 10.72 | 9.63 | 7.86 | 7.55 | 7.04 | 8.21 | 7.07 | 6.92 | **5.92** | model +trained | -14.5% |
| 36 | [clawdia](https://js13kgames.com/2025/games/clawdia) | 36.92 | 12.87 | 12.65 | 12.50 | 12.64 | 12.79 | 12.40 | 11.17 | 11.80 | 11.12 | 8.72 | 12.12 | 8.67 | 10.25 | **7.43** | model +trained | -14.3% |
| 37 | [catway-xr](https://js13kgames.com/2025/games/catway-xr) | 38.05 | 13.00 | 13.00 | 12.74 | 13.00 | 13.29 | 12.62 | 11.66 | 12.28 | 11.53 | 10.28 | 12.66 | 10.49 | 10.75 | **8.81** | model +trained | -14.3% |
| 38 | [night-dealer](https://js13kgames.com/2025/games/night-dealer) | 37.17 | 12.47 | 12.02 | 11.83 | 12.02 | 12.07 | 11.95 | 10.45 | 10.83 | 10.17 | 9.55 | 11.08 | 9.47 | 9.32 | **8.00** | model +trained | -14.2% |
| 39 | [the-soldier](https://js13kgames.com/2025/games/the-soldier) | 23.28 | 9.14 | 8.95 | 8.88 | 8.95 | 9.04 | 8.80 | 7.95 | 8.37 | 7.91 | 7.37 | 8.58 | 7.36 | 7.44 | **6.32** | model +trained | -14.2% |
| 40 | [cat-mouse-hunt](https://js13kgames.com/2025/games/cat-mouse-hunt) | 12.48 | 5.92 | 5.32 | 5.16 | 5.32 | 5.20 | 5.29 | 4.80 | 4.99 | 4.29 | 3.97 | 5.13 | 4.38 | 4.12 | **3.42** | model +trained | -13.7% |
| 41 | [please-do-not-the-cat](https://js13kgames.com/2025/games/please-do-not-the-cat) | 34.25 | 13.00 | 11.88 | 11.68 | 11.88 | 12.02 | 11.76 | 10.55 | 10.79 | 9.98 | 9.49 | 11.29 | 9.49 | 9.54 | **8.19** | model +trained | -13.7% |
| 42 | [pawker](https://js13kgames.com/2025/games/pawker) | 19.48 | 7.23 | 7.22 | 6.96 | 7.22 | 7.64 | 7.05 | 6.59 | 6.33 | 5.89 | 4.30 | 6.64 | 4.27 | 5.69 | **3.68** | model +trained | -13.7% |
| 43 | [dusk](https://js13kgames.com/2025/games/dusk) | 35.26 | 12.39 | 12.39 | 12.17 | 12.39 | 12.91 | 12.14 | 11.16 | 11.10 | 10.43 | 9.19 | 11.45 | 9.13 | 9.97 | **7.89** | model +trained | -13.6% |
| 44 | [go-home](https://js13kgames.com/2025/games/go-home) | 30.39 | 12.96 | 12.96 | 12.73 | 12.96 | 13.00 | 12.72 | 11.73 | 12.05 | 11.44 | 9.96 | 12.66 | 10.10 | 10.69 | **8.61** | model +trained | -13.5% |
| 45 | [crafty-cauldron](https://js13kgames.com/2025/games/crafty-cauldron) | 27.38 | 12.22 | 11.63 | 11.42 | 11.63 | 11.81 | 11.48 | 10.52 | 10.55 | 9.97 | 8.92 | 10.86 | 8.98 | 9.38 | **7.72** | model +trained | -13.5% |
| 46 | [shadow-alley](https://js13kgames.com/2025/games/shadow-alley) | 32.81 | 10.06 | 9.63 | 9.60 | 9.63 | 9.87 | 9.16 | 8.51 | 9.02 | 8.52 | 7.81 | 9.36 | 7.96 | 8.05 | **6.77** | model +trained | -13.4% |
| 47 | [tom-black-cat](https://js13kgames.com/2025/games/tom-black-cat) | 12.73 | 6.41 | 6.34 | 6.14 | 6.34 | 6.66 | 6.44 | 5.99 | 5.37 | 4.99 | 4.33 | 5.44 | 4.30 | 4.82 | **3.72** | model +trained | -13.3% |
| 48 | [shadows-escape](https://js13kgames.com/2025/games/shadows-escape) | 46.19 | 10.46 | 9.92 | 9.86 | 9.87 | 9.43 | 9.57 | 8.21 | 9.40 | 8.68 | 8.41 | 9.60 | 8.30 | 8.03 | **6.96** | model +trained | -13.3% |
| 49 | [cat-apult](https://js13kgames.com/2025/games/cat-apult) | 25.87 | 10.31 | 10.66 | 10.25 | 10.66 | 10.66 | 10.45 | 9.59 | 8.70 | 8.27 | 7.47 | 9.16 | 7.69 | 7.84 | **6.48** | model +trained | -13.3% |
| 50 | [clawz](https://js13kgames.com/2025/games/clawz) | 24.03 | 8.75 | 8.55 | 8.49 | 8.55 | 8.82 | 8.27 | 7.67 | 8.12 | 7.54 | 6.42 | 8.33 | 6.41 | 7.32 | **5.56** | model +trained | -13.3% |
| 51 | [black-cat-potions](https://js13kgames.com/2025/games/black-cat-potions) | 31.56 | 10.23 | 10.43 | 10.26 | 10.43 | 10.37 | 10.05 | 9.21 | 9.73 | 9.20 | 8.15 | 10.12 | 8.45 | 8.50 | **7.08** | model +trained | -13.2% |
| 52 | [clicker-pawry](https://js13kgames.com/2025/games/clicker-pawry) | 30.85 | 12.00 | 12.20 | 12.00 | 12.20 | 12.29 | 11.93 | 11.02 | 11.26 | 10.63 | 9.72 | 11.86 | 9.74 | 10.07 | **8.44** | model +trained | -13.2% |
| 53 | [dinner-for-two](https://js13kgames.com/2025/games/dinner-for-two) | 28.06 | 10.48 | 10.72 | 10.58 | 10.72 | 10.73 | 10.30 | 9.66 | 10.03 | 9.57 | 8.65 | 10.50 | 8.98 | 8.88 | **7.52** | model +trained | -13.0% |
| 54 | [color-cat](https://js13kgames.com/2025/games/color-cat) | 37.59 | 10.85 | 10.76 | 10.66 | 10.75 | 10.90 | 10.37 | 9.48 | 10.10 | 9.50 | 8.69 | 10.35 | 8.59 | 8.89 | **7.47** | model +trained | -13.0% |
| 55 | [meow-or-never](https://js13kgames.com/2025/games/meow-or-never) | 11.21 | 4.99 | 4.84 | 4.69 | 4.84 | 4.99 | 4.73 | 4.51 | 4.58 | 4.14 | 3.36 | 4.79 | 3.54 | 3.99 | **2.93** | model +trained | -12.9% |
| 56 | [lazy-claws](https://js13kgames.com/2025/games/lazy-claws) | 29.05 | 11.68 | 11.42 | 11.29 | 11.42 | 11.58 | 11.11 | 10.48 | 10.87 | 10.47 | 9.47 | 11.46 | 9.58 | 9.86 | **8.26** | model +trained | -12.8% |
| 57 | [puzzle-black-cat-combat](https://js13kgames.com/2025/games/puzzle-black-cat-combat) | 19.82 | 6.76 | 6.76 | 6.71 | 6.76 | 6.95 | 6.51 | 6.04 | 6.35 | 5.94 | 4.99 | 6.50 | 5.01 | 5.75 | **4.36** | model +trained | -12.7% |
| 58 | [learn-to-code-cat](https://js13kgames.com/2025/games/learn-to-code-cat) | 32.60 | 12.18 | 12.37 | 12.16 | 12.37 | 12.35 | 11.85 | 11.17 | 11.53 | 11.02 | 10.33 | 12.30 | 10.56 | 10.40 | **9.01** | model +trained | -12.7% |
| 59 | [catto](https://js13kgames.com/2025/games/catto) | 25.87 | 10.42 | 10.38 | 10.01 | 10.38 | 10.49 | 10.39 | 9.43 | 8.63 | 8.20 | 7.43 | 9.06 | 7.33 | 7.73 | **6.40** | model +trained | -12.7% |
| 60 | [9-lives](https://js13kgames.com/2025/games/9-lives) | 6.79 | 3.14 | 3.13 | 3.09 | 3.13 | 3.28 | 3.09 | 2.91 | 2.94 | 2.64 | 2.29 | 3.02 | 2.30 | 2.64 | **2.01** | model +trained | -12.5% |
| 61 | [heimao](https://js13kgames.com/2025/games/heimao) | 43.36 | 8.68 | 8.17 | 8.05 | 8.17 | 8.25 | 7.77 | 6.97 | 7.63 | 7.30 | 6.81 | 7.89 | 6.87 | 6.42 | **5.62** | model +trained | -12.4% |
| 62 | [semi-charmed-kind-of-dice](https://js13kgames.com/2025/games/semi-charmed-kind-of-dice) | 32.73 | 12.33 | 11.95 | 11.74 | 11.95 | 12.49 | 11.74 | 10.95 | 10.91 | 10.33 | 8.90 | 11.37 | 8.93 | 9.80 | **7.80** | model +trained | -12.4% |
| 63 | [midnights-journey](https://js13kgames.com/2025/games/midnights-journey) | 37.42 | 12.70 | 12.78 | 12.70 | 12.74 | 13.29 | 12.13 | 11.62 | 11.98 | 11.40 | 10.35 | 12.48 | 10.48 | 10.78 | **9.07** | model +trained | -12.3% |
| 64 | [edgars-mission](https://js13kgames.com/2025/games/edgars-mission) | 35.38 | 11.80 | 11.39 | 11.17 | 11.37 | 11.62 | 10.79 | 10.11 | 10.65 | 10.05 | 8.72 | 11.12 | 9.07 | 9.32 | **7.66** | model +trained | -12.2% |
| 65 | [open-the-door](https://js13kgames.com/2025/games/open-the-door) | 3.34 | 2.49 | 2.36 | 2.11 | 2.36 | 2.54 | 2.47 | 2.27 | 1.40 | 1.19 | 0.97 | 1.38 | 0.93 | 1.20 | **0.82** | model +trained | -12.0% |
| 66 | [enchiladas-day-off](https://js13kgames.com/2025/games/enchiladas-day-off) | 29.08 | 12.12 | 11.68 | 11.45 | 11.68 | 11.89 | 11.70 | 10.68 | 10.17 | 9.90 | 9.19 | 10.79 | 9.31 | 9.22 | **8.09** | model +trained | -11.9% |
| 67 | [grass-tower-defense](https://js13kgames.com/2025/games/grass-tower-defense) | 29.56 | 12.83 | 12.65 | 12.38 | 12.65 | 13.03 | 12.54 | 11.68 | 11.26 | 10.61 | 8.97 | 11.88 | 9.01 | 10.12 | **7.90** | model +trained | -11.9% |
| 68 | [black-cat-maze](https://js13kgames.com/2025/games/black-cat-maze) | 57.68 | 12.86 | 12.86 | 12.62 | 12.78 | 13.29 | 12.16 | 11.58 | 12.01 | 11.49 | 9.76 | 12.56 | 10.00 | 10.72 | **8.60** | model +trained | -11.9% |
| 69 | [catris](https://js13kgames.com/2025/games/catris) | 20.35 | 8.54 | 8.16 | 7.97 | 8.16 | 8.30 | 8.06 | 7.46 | 7.91 | 7.00 | 6.06 | 8.13 | 6.54 | 6.89 | **5.35** | model +trained | -11.7% |
| 70 | [rin-the-cat-sorceresses](https://js13kgames.com/2025/games/rin-the-cat-sorceresses) | 29.56 | 13.00 | 12.37 | 11.89 | 12.37 | 12.58 | 12.28 | 11.35 | 9.26 | 8.90 | 8.25 | 9.78 | 8.36 | 8.37 | **7.29** | model +trained | -11.6% |
| 71 | [kittens-united](https://js13kgames.com/2025/games/kittens-united) | 30.71 | 13.00 | 12.71 | 12.58 | 12.71 | 12.65 | 12.40 | 11.27 | 11.77 | 11.22 | 9.48 | 12.17 | 9.67 | 10.57 | **8.40** | model +trained | -11.5% |
| 72 | [superstition-the-black-cat-game](https://js13kgames.com/2025/games/superstition-the-black-cat-game) | 28.10 | 9.75 | 8.85 | 8.61 | 8.85 | 8.91 | 8.90 | 7.93 | 7.88 | 7.18 | 6.43 | 8.50 | 6.92 | 6.90 | **5.69** | model +trained | -11.4% |
| 73 | [assault-on-city-13](https://js13kgames.com/2025/games/assault-on-city-13) | 41.23 | 12.94 | 12.94 | 12.90 | 12.93 | 12.83 | 12.13 | 11.18 | 11.99 | 11.65 | 10.78 | 12.53 | 11.01 | 10.71 | **9.52** | model +trained | -11.1% |
| 74 | [a-voiding-your-problems](https://js13kgames.com/2025/games/a-voiding-your-problems) | 32.82 | 13.00 | 12.79 | 12.56 | 12.79 | 12.61 | 12.18 | 11.43 | 11.26 | 10.87 | 10.22 | 12.07 | 10.60 | 10.22 | **9.09** | model +trained | -11.1% |
| 75 | [dont-forget-to-feed-the-cat](https://js13kgames.com/2025/games/dont-forget-to-feed-the-cat) | 19.12 | 12.60 | 8.06 | 7.80 | 8.06 | 8.57 | 7.98 | 7.54 | 6.40 | 6.10 | 4.28 | 6.65 | 4.33 | 5.78 | **3.81** | model +trained | -11.1% |
| 76 | [the-black-cross-cat](https://js13kgames.com/2025/games/the-black-cross-cat) | 30.05 | 11.47 | 10.97 | 10.72 | 10.97 | 11.63 | 10.87 | 10.26 | 10.52 | 9.83 | 8.62 | 10.86 | 8.65 | 9.47 | **7.67** | model +trained | -11.0% |
| 77 | [unlucky-street](https://js13kgames.com/2025/games/unlucky-street) | 16.73 | 11.53 | 11.09 | 10.30 | 11.09 | 11.41 | 11.29 | 10.54 | 6.84 | 6.44 | 5.56 | 7.16 | 5.46 | 6.29 | **4.87** | model +trained | -10.8% |
| 78 | [kikis-cafe](https://js13kgames.com/2025/games/kikis-cafe) | 28.92 | 12.66 | 12.52 | 12.13 | 12.52 | 12.80 | 12.36 | 11.36 | 10.68 | 10.14 | 9.22 | 11.13 | 9.45 | 9.70 | **8.25** | model +trained | -10.4% |
| 79 | [cod-black-cat](https://js13kgames.com/2025/games/cod-black-cat) | 32.99 | 10.64 | 10.69 | 10.53 | 10.69 | 11.06 | 10.21 | 9.62 | 9.80 | 9.36 | 7.91 | 10.24 | 8.11 | 8.80 | **7.09** | model +trained | -10.3% |
| 80 | [furball](https://js13kgames.com/2025/games/furball) | 5.89 | 3.62 | 3.53 | 3.38 | 3.53 | 3.62 | 3.58 | 3.35 | 3.00 | 2.78 | 2.45 | 3.15 | 2.42 | 2.73 | **2.17** | model +trained | -10.0% |
| 81 | [pawspaint](https://js13kgames.com/2025/games/pawspaint) | 27.47 | 11.21 | 11.33 | 11.03 | 11.33 | 11.94 | 11.38 | 10.52 | 9.72 | 9.14 | 8.12 | 10.10 | 8.03 | 8.85 | **7.24** | model +trained | -9.8% |
| 82 | [catslap](https://js13kgames.com/2025/games/catslap) | 18.47 | 7.40 | 7.04 | 6.82 | 7.04 | 7.38 | 7.07 | 6.55 | 6.10 | 5.78 | 5.08 | 6.44 | 5.08 | 5.57 | **4.59** | model +trained | -9.8% |
| 83 | [basement-cat-ascends](https://js13kgames.com/2025/games/basement-cat-ascends) | 51.78 | 12.98 | 13.12 | 12.98 | 13.11 | 13.60 | 12.32 | 11.87 | 12.17 | 11.85 | 10.73 | 12.83 | 10.91 | 11.13 | **9.70** | model +trained | -9.6% |
| 84 | [black-cat-caffee](https://js13kgames.com/2025/games/black-cat-caffee) | 24.61 | 12.97 | 13.49 | 12.77 | 13.50 | 14.30 | 14.07 | 12.81 | 7.55 | 6.94 | 6.34 | 7.84 | 6.68 | 6.64 | **5.74** | model +trained | -9.5% |
| 85 | [bike](https://js13kgames.com/2025/games/bike) | 13.19 | 6.02 | 5.83 | 5.64 | 5.83 | 6.25 | 5.96 | 5.50 | 5.20 | 4.92 | 4.64 | 5.42 | 4.69 | 4.71 | **4.21** | model +trained | -9.3% |
| 86 | [mewnbeams-adventure](https://js13kgames.com/2025/games/mewnbeams-adventure) | 28.65 | 13.00 | 13.10 | 13.00 | 13.10 | 13.15 | 12.80 | 11.99 | 12.57 | 12.01 | 11.41 | 13.24 | 11.67 | 11.55 | **10.35** | model +trained | -9.3% |
| 87 | [cat-shooter-ritual-catacombs](https://js13kgames.com/2025/games/cat-shooter-ritual-catacombs) | 39.40 | 12.95 | 12.79 | 12.71 | 12.74 | 13.08 | 12.17 | 11.32 | 11.89 | 11.49 | 10.70 | 12.38 | 10.83 | 10.30 | **9.34** | model +trained | -9.3% |
| 88 | [triskball](https://js13kgames.com/2025/games/triskball) | 42.26 | 12.21 | 12.46 | 12.25 | 12.45 | 13.11 | 11.71 | 11.44 | 11.47 | 11.05 | 9.31 | 12.54 | 9.83 | 10.27 | **8.46** | model +trained | -9.1% |
| 89 | [nine-lives-later-deja-meow](https://js13kgames.com/2025/games/nine-lives-later-deja-meow) | 26.87 | 10.35 | 10.12 | 9.94 | 10.12 | 10.58 | 9.82 | 9.26 | 9.26 | 8.75 | 6.99 | 9.58 | 7.08 | 8.32 | **6.35** | model +trained | -9.1% |
| 90 | [black-cat-shinobi](https://js13kgames.com/2025/games/black-cat-shinobi) | 52.67 | 11.13 | 10.76 | 10.56 | 10.66 | 10.68 | 9.86 | 9.26 | 9.49 | 9.17 | 8.30 | 9.96 | 8.69 | 8.58 | **7.55** | model +trained | -9.0% |
| 91 | [wash-the-cat](https://js13kgames.com/2025/games/wash-the-cat) | 48.33 | 12.00 | 11.46 | 11.30 | 11.44 | 11.47 | 10.55 | 9.98 | 10.21 | 9.84 | 8.56 | 10.91 | 9.05 | 8.95 | **7.79** | model +trained | -9.0% |
| 92 | [mor-chess](https://js13kgames.com/2025/games/mor-chess) | 31.02 | 12.94 | 13.12 | 12.91 | 13.12 | 13.59 | 12.84 | 12.20 | 11.98 | 11.49 | 10.43 | 12.64 | 10.65 | 11.08 | **9.51** | model +trained | -8.8% |
| 93 | [infinite-hunger](https://js13kgames.com/2025/games/infinite-hunger) | 31.20 | 12.99 | 13.13 | 12.98 | 13.13 | 12.80 | 12.68 | 11.70 | 12.12 | 11.67 | 10.39 | 12.84 | 10.92 | 10.86 | **9.48** | model +trained | -8.8% |
| 94 | [cat-fight](https://js13kgames.com/2025/games/cat-fight) | 18.01 | 7.13 | 6.13 | 6.00 | 6.13 | 6.54 | 6.02 | 5.65 | 5.46 | 5.09 | 4.63 | 5.62 | 4.71 | 4.99 | **4.22** | model +trained | -8.8% |
| 95 | [wild-catch](https://js13kgames.com/2025/games/wild-catch) | 41.89 | 13.00 | 14.40 | 14.33 | 14.39 | 14.90 | 13.54 | 12.96 | 13.39 | 12.87 | 11.63 | 13.93 | 11.96 | 12.34 | **10.62** | model +trained | -8.7% |
| 96 | [roni-and-kira](https://js13kgames.com/2025/games/roni-and-kira) | 27.46 | 11.06 | 10.70 | 10.45 | 10.70 | 11.15 | 10.42 | 9.79 | 9.71 | 9.18 | 6.50 | 10.14 | 6.59 | 8.70 | **5.96** | model +trained | -8.4% |
| 97 | [fishing](https://js13kgames.com/2025/games/fishing) | 27.30 | 11.61 | 11.12 | 10.64 | 11.12 | 11.53 | 11.33 | 10.25 | 8.45 | 7.89 | 6.84 | 8.89 | 7.47 | 7.41 | **6.28** | model +trained | -8.2% |
| 98 | [gravity-cat](https://js13kgames.com/2025/games/gravity-cat) | 30.94 | 12.64 | 12.79 | 12.63 | 12.79 | 12.54 | 12.27 | 11.40 | 11.70 | 11.20 | 9.48 | 12.42 | 9.97 | 10.53 | **8.70** | model +trained | -8.2% |
| 99 | [jazz](https://js13kgames.com/2025/games/jazz) | 15.99 | 8.46 | 8.26 | 7.93 | 8.26 | 8.86 | 8.51 | 7.81 | 6.32 | 6.02 | 5.57 | 6.51 | 5.68 | 5.71 | **5.14** | model +trained | -7.8% |
| 100 | [poker-mysterio](https://js13kgames.com/2025/games/poker-mysterio) | 31.65 | 12.98 | 12.75 | 12.57 | 12.75 | 13.27 | 12.35 | 11.62 | 11.81 | 11.23 | 8.89 | 12.20 | 8.98 | 10.67 | **8.22** | model +trained | -7.6% |
| 101 | [encounter](https://js13kgames.com/2025/games/encounter) | 14.55 | 7.71 | 7.29 | 6.97 | 7.29 | 7.59 | 7.54 | 6.80 | 5.41 | 4.98 | 4.82 | 5.66 | 4.92 | 4.70 | **4.35** | model +trained | -7.4% |
| 102 | [alley-kitty](https://js13kgames.com/2025/games/alley-kitty) | 36.32 | 12.62 | 12.75 | 12.66 | 12.74 | 12.79 | 12.08 | 11.52 | 11.93 | 11.50 | 10.93 | 12.78 | 11.63 | 10.51 | **9.74** | model +trained | -7.4% |
| 103 | [black-cat-in-25d-land](https://js13kgames.com/2025/games/black-cat-in-25d-land) | 23.48 | 10.25 | 9.96 | 9.72 | 9.96 | 10.35 | 9.88 | 9.36 | 9.04 | 8.47 | 6.97 | 9.67 | 7.08 | 8.31 | **6.47** | model +trained | -7.2% |
| 104 | [kuro-neko](https://js13kgames.com/2025/games/kuro-neko) | 21.59 | 9.50 | 9.46 | 9.11 | 9.46 | 9.73 | 9.50 | 8.88 | 7.90 | 7.49 | 6.88 | 8.23 | 6.77 | 7.31 | **6.32** | model +trained | -6.7% |
| 105 | [black-cat-survivor](https://js13kgames.com/2025/games/black-cat-survivor) | 40.42 | 12.88 | 13.10 | 12.88 | 13.10 | 13.34 | 12.67 | 11.73 | 11.96 | 11.54 | 8.88 | 12.51 | 9.05 | 10.62 | **8.30** | model +trained | -6.6% |
| 106 | [the-laser-cat](https://js13kgames.com/2025/games/the-laser-cat) | 13.03 | 7.76 | 7.38 | 7.05 | 7.38 | 7.64 | 7.49 | 7.04 | 5.69 | 5.28 | 4.90 | 5.94 | 4.78 | 5.34 | **4.47** | model +trained | -6.6% |
| 107 | [black-cat-on-cybertower](https://js13kgames.com/2025/games/black-cat-on-cybertower) | 31.39 | 9.99 | 8.74 | 8.59 | 8.74 | 9.67 | 8.50 | 8.54 | 8.36 | 7.90 | 7.13 | 8.75 | 7.26 | 7.71 | **6.67** | model +trained | -6.5% |
| 108 | [guardian-cat](https://js13kgames.com/2025/games/guardian-cat) | 45.62 | 11.39 | 11.33 | 11.12 | 11.32 | 12.12 | 10.67 | 10.63 | 10.19 | 9.99 | 9.53 | 10.84 | 9.99 | 9.14 | **8.55** | model +trained | -6.5% |
| 109 | [hungry-cat](https://js13kgames.com/2025/games/hungry-cat) | 21.19 | 12.75 | 12.04 | 11.54 | 12.04 | 12.36 | 12.04 | 11.49 | 9.62 | 9.11 | 7.26 | 10.11 | 7.15 | 8.95 | **6.71** | model +trained | -6.2% |
| 110 | [stray-cat](https://js13kgames.com/2025/games/stray-cat) | 15.86 | 12.60 | 11.18 | 10.56 | 11.18 | 12.27 | 11.28 | 10.63 | 6.13 | 5.67 | 4.88 | 6.47 | 4.95 | 5.72 | **4.60** | model +trained | -5.8% |
| 111 | [the-black-cat](https://js13kgames.com/2025/games/the-black-cat) | 19.21 | 5.58 | 5.58 | 5.40 | 5.58 | 5.80 | 5.29 | 5.12 | 4.71 | 4.52 | 4.15 | 5.08 | 4.28 | 4.52 | **3.94** | model +trained | -5.0% |
| 112 | [swing-a-cat](https://js13kgames.com/2025/games/swing-a-cat) | 34.24 | 12.99 | 12.60 | 12.43 | 12.60 | 13.18 | 12.22 | 11.83 | 11.96 | 11.50 | 10.36 | 12.46 | 10.81 | 10.80 | **9.87** | model +trained | -4.7% |
| 113 | [laserpointer](https://js13kgames.com/2025/games/laserpointer) | 25.90 | 11.54 | 11.63 | 11.54 | 11.63 | 11.83 | 11.36 | 10.76 | 11.10 | 10.52 | 8.67 | 11.67 | 9.02 | 10.18 | **8.27** | model +trained | -4.6% |
| 114 | [clawstruck](https://js13kgames.com/2025/games/clawstruck) | 25.26 | 11.74 | 11.03 | 10.53 | 11.03 | 11.13 | 11.04 | 10.46 | 8.52 | 8.12 | 7.65 | 8.84 | 7.57 | 7.99 | **7.23** | model +trained | -4.5% |
| 115 | [rll](https://js13kgames.com/2025/games/rll) | 23.63 | 12.03 | 11.63 | 11.35 | 11.63 | 12.00 | 11.69 | 11.25 | 10.59 | 10.23 | 10.02 | 10.98 | 10.06 | 10.04 | **9.60** | model +trained | -4.2% |
| 116 | [tangled](https://js13kgames.com/2025/games/tangled) | 30.70 | 9.04 | 8.67 | 8.61 | 8.67 | 8.69 | 8.33 | 7.51 | 8.19 | 7.89 | 7.71 | 8.44 | 7.75 | 7.00 | **6.71** | model +trained | -4.1% |
| 117 | [shadowflame](https://js13kgames.com/2025/games/shadowflame) | 28.04 | 10.47 | 10.55 | 10.27 | 10.55 | 10.88 | 10.52 | 10.10 | 9.30 | 9.01 | 8.52 | 9.56 | 8.43 | 8.80 | **8.09** | model +trained | -4.0% |
| 118 | [pacific-black-cats](https://js13kgames.com/2025/games/pacific-black-cats) | 26.53 | 13.00 | 12.87 | 12.66 | 12.87 | 13.02 | 12.49 | 11.98 | 11.83 | 11.46 | 8.18 | 12.46 | 8.61 | 10.56 | **7.88** | model +trained | -3.6% |
| 119 | [the-giver](https://js13kgames.com/2025/games/the-giver) | 19.86 | 12.96 | 12.71 | 12.49 | 12.71 | 13.10 | 12.80 | 12.38 | 11.77 | 11.28 | 10.72 | 12.17 | 10.67 | 11.38 | **10.37** | model +trained | -2.8% |
| 120 | [bastet-night](https://js13kgames.com/2025/games/bastet-night) | 15.25 | 11.19 | 11.80 | 11.11 | 11.80 | 12.00 | 11.90 | 11.50 | 8.50 | 8.15 | 6.97 | 8.89 | 6.93 | 8.08 | **6.74** | model +trained | -2.7% |
| 121 | [black-lantern](https://js13kgames.com/2025/games/black-lantern) | 27.08 | 12.98 | 12.98 | 12.80 | 12.98 | 13.43 | 11.91 | 12.19 | 11.53 | 11.50 | 10.87 | 12.82 | 11.41 | 11.15 | **10.63** | model +trained | -2.3% |
| 122 | [void-cat](https://js13kgames.com/2025/games/void-cat) RR | 19.71 | 11.87 | 11.61 | 11.37 | 11.61 | 11.96 | 11.84 | 11.68 | 11.18 | 10.66 | 9.82 | 11.53 | 10.06 | 10.70 | **9.64** | model +trained | -1.8% |
| 123 | [black-cats-herbs-brews](https://js13kgames.com/2025/games/black-cats-herbs-brews) RR | 21.61 | 13.00 | 13.00 | 12.77 | 13.00 | 13.30 | 13.20 | 13.13 | 12.77 | 12.24 | 11.57 | 13.05 | 11.74 | 12.30 | **11.39** | model +trained | -1.6% |
| 124 | [mochi-and-the-midnight-escape](https://js13kgames.com/2025/games/mochi-and-the-midnight-escape) | 40.71 | 13.00 | 13.18 | 13.01 | 13.17 | 14.20 | 12.77 | 12.38 | 12.60 | 11.98 | 10.22 | 13.18 | 10.78 | 11.46 | **10.11** | model +trained | -1.1% |
| 125 | [miami-mice](https://js13kgames.com/2025/games/miami-mice) RR | 19.46 | 12.44 | 12.38 | 12.16 | 12.38 | 12.75 | 12.65 | 12.59 | 12.13 | 11.53 | 10.86 | 12.38 | 10.99 | 11.72 | **10.75** | model +trained | -1.0% |
| 126 | [puzzle](https://js13kgames.com/2025/games/puzzle) | 11.70 | 9.53 | 9.51 | 9.27 | 9.51 | 9.76 | 9.70 | 9.36 | 8.66 | 8.22 | 7.91 | 8.86 | 7.85 | 8.38 | **7.81** | model +trained | -0.5% |
| 127 | [yarn-run](https://js13kgames.com/2025/games/yarn-run) RR | 18.39 | 12.96 | 12.96 | 12.72 | 12.96 | 13.40 | 13.07 | 13.42 | 12.92 | 12.29 | 11.84 | 13.10 | 12.00 | 12.59 | **11.79** | model +trained | -0.4% |
| 128 | [boots-and-the-black-cauldron](https://js13kgames.com/2025/games/boots-and-the-black-cauldron) RR | 14.07 | 10.86 | 11.05 | 10.84 | 11.05 | 11.35 | 11.25 | 11.49 | 10.98 | 10.74 | 10.30 | 11.26 | 10.65 | 10.78 | **10.29** | model +trained | -0.0% |
| 129 | [run-from-the-black-cat](https://js13kgames.com/2025/games/run-from-the-black-cat) RR | 7.19 | 5.61 | 5.61 | 5.44 | 5.61 | 5.81 | 5.66 | 5.79 | 5.52 | 5.25 | 4.76 | 5.56 | 4.80 | 5.36 | **4.76** | model +trained | -0.0% |
| 130 | [all-you-have-to-do-is-dream](https://js13kgames.com/2025/games/all-you-have-to-do-is-dream) RR | 18.26 | 12.92 | 12.79 | 12.49 | 12.79 | 13.15 | 12.89 | 13.18 | 12.64 | 12.19 | **11.44** | 13.17 | 11.76 | 12.31 | 11.46 | brotli +dict | 0.1% |
| 131 | [black-cat-chat](https://js13kgames.com/2025/games/black-cat-chat) | 4.77 | 3.66 | 3.76 | 3.68 | 3.76 | 3.93 | 3.85 | 3.90 | 3.71 | 3.55 | **3.23** | 3.71 | 3.25 | 3.60 | 3.24 | brotli +dict | 0.2% |
| 132 | [roll-for-mischief](https://js13kgames.com/2025/games/roll-for-mischief) RR | 16.94 | 12.80 | 12.57 | 12.31 | 12.57 | 12.93 | 12.67 | 13.01 | 12.53 | 11.96 | **11.43** | 12.61 | 11.52 | 12.25 | 11.46 | brotli +dict | 0.2% |
| 133 | [meow-mountain](https://js13kgames.com/2025/games/meow-mountain) | 17.67 | 12.96 | 13.15 | 12.95 | 13.15 | 13.46 | 13.42 | 13.56 | 13.08 | 12.65 | **12.11** | 13.20 | 12.20 | 12.79 | 12.14 | brotli +dict | 0.3% |
| 134 | [kittens-crossing](https://js13kgames.com/2025/games/kittens-crossing) | 16.69 | 12.82 | 12.55 | 12.33 | 12.55 | 12.86 | 12.69 | 13.01 | 12.54 | 12.03 | **11.50** | 12.60 | 11.58 | 12.28 | 11.53 | brotli +dict | 0.3% |
| 135 | [inside-cat](https://js13kgames.com/2025/games/inside-cat) RR | 17.73 | 12.82 | 12.96 | 12.80 | 12.96 | 13.29 | 13.15 | 13.50 | 13.01 | 12.53 | **11.99** | 13.11 | 12.09 | 12.73 | 12.03 | brotli +dict | 0.3% |
| 136 | [midnight-prowler](https://js13kgames.com/2025/games/midnight-prowler) RR | 16.21 | 12.41 | 12.41 | 12.14 | 12.41 | 12.68 | 12.52 | 12.85 | 12.37 | 11.89 | **11.45** | 12.39 | 11.53 | 12.13 | 11.50 | brotli +dict | 0.5% |
| 137 | [happy-birthday-jerry](https://js13kgames.com/2025/games/happy-birthday-jerry) RR | 7.99 | 6.13 | 6.07 | 5.85 | 6.07 | 6.31 | 6.12 | 6.25 | 5.98 | 5.62 | **5.03** | 6.01 | 5.07 | 5.79 | 5.06 | brotli +dict | 0.5% |
| 138 | [meoq-n-shadow](https://js13kgames.com/2025/games/meoq-n-shadow) RR | 17.01 | 12.81 | 13.07 | 12.81 | 13.07 | 13.31 | 13.20 | 13.55 | 13.06 | 12.54 | **12.11** | 13.01 | 12.18 | 12.80 | 12.18 | brotli +dict | 0.5% |
| 139 | [trampofelines](https://js13kgames.com/2025/games/trampofelines) RR | 17.00 | 12.96 | 12.95 | 12.74 | 12.95 | 13.25 | 13.20 | 13.42 | 12.90 | 12.40 | **11.89** | 12.89 | 11.96 | 12.65 | 11.95 | brotli +dict | 0.5% |
| 140 | [order-of-the-crimson-dot](https://js13kgames.com/2025/games/order-of-the-crimson-dot) RR | 14.87 | 12.90 | 13.10 | 12.55 | 13.10 | 13.42 | 13.35 | 13.38 | 11.50 | 11.06 | **10.53** | 12.28 | 11.16 | 11.28 | 10.60 | brotli +dict | 0.7% |
| 141 | [catapoolt](https://js13kgames.com/2025/games/catapoolt) RR | 17.34 | 12.94 | 13.15 | 12.94 | 13.15 | 13.48 | 13.33 | 13.69 | 13.18 | 12.62 | **12.02** | 13.16 | 12.08 | 12.92 | 12.11 | brotli +dict | 0.7% |
| 142 | [nyx-felis-and-lampyris](https://js13kgames.com/2025/games/nyx-felis-and-lampyris) RR | 16.04 | 12.50 | 12.34 | 12.09 | 12.34 | 12.56 | 12.46 | 12.79 | 12.32 | 11.84 | **11.33** | 12.32 | 11.40 | 12.07 | 11.41 | brotli +dict | 0.7% |
| 143 | [mor-chess-2](https://js13kgames.com/2025/games/mor-chess-2) RR | 13.29 | 10.33 | 10.48 | 10.22 | 10.48 | 10.80 | 10.72 | 10.85 | 10.08 | 9.61 | **9.04** | 10.11 | 9.11 | 9.87 | 9.11 | brotli +dict | 0.7% |
| 144 | [cancel-elon](https://js13kgames.com/2025/games/cancel-elon) RR | 16.78 | 12.99 | 12.93 | 12.71 | 12.93 | 13.18 | 13.16 | 13.39 | 12.86 | 12.38 | **11.83** | 12.89 | 11.91 | 12.62 | 11.92 | brotli +dict | 0.7% |
| 145 | [celestial-paws](https://js13kgames.com/2025/games/celestial-paws) RR | 15.98 | 12.74 | 12.27 | 12.05 | 12.27 | 12.52 | 12.42 | 12.75 | 12.28 | 11.85 | **11.36** | 12.23 | 11.39 | 12.04 | 11.45 | brotli +dict | 0.7% |
| 146 | [project-panther](https://js13kgames.com/2025/games/project-panther) RR | 13.97 | 10.72 | 10.72 | 10.56 | 10.72 | 10.93 | 10.92 | 11.14 | 10.77 | 10.45 | **10.05** | 10.69 | 10.10 | 10.58 | 10.13 | brotli +dict | 0.8% |
| 147 | [al13ycat](https://js13kgames.com/2025/games/al13ycat) RR | 17.20 | 12.99 | 13.17 | 13.00 | 13.17 | 13.38 | 13.39 | 13.69 | 13.25 | 12.82 | **12.50** | 13.15 | 12.55 | 13.01 | 12.60 | brotli +dict | 0.8% |
| 148 | [alley-cat-noir](https://js13kgames.com/2025/games/alley-cat-noir) RR | 12.27 | 9.52 | 9.47 | 9.26 | 9.47 | 9.68 | 9.58 | 9.82 | 9.44 | 9.05 | **8.59** | 9.40 | 8.64 | 9.24 | 8.67 | brotli +dict | 0.8% |
| 149 | [non-mewtonian-cat](https://js13kgames.com/2025/games/non-mewtonian-cat) RR | 11.89 | 9.22 | 9.07 | 8.86 | 9.07 | 9.35 | 9.18 | 9.42 | 9.03 | 8.61 | **8.01** | 9.05 | 8.04 | 8.83 | 8.08 | brotli +dict | 0.8% |
| 150 | [biralo](https://js13kgames.com/2025/games/biralo) RR | 17.28 | 12.97 | 13.22 | 12.97 | 13.22 | 13.50 | 13.37 | 13.74 | 13.22 | 12.70 | **12.20** | 13.17 | 12.27 | 12.97 | 12.30 | brotli +dict | 0.9% |
| 151 | [pet-a-cat](https://js13kgames.com/2025/games/pet-a-cat) RR | 13.42 | 10.14 | 10.30 | 10.14 | 10.30 | 10.54 | 10.48 | 10.71 | 10.34 | 10.13 | **9.60** | 10.29 | 9.63 | 10.14 | 9.69 | brotli +dict | 0.9% |
| 152 | [worms-13k](https://js13kgames.com/2025/games/worms-13k) RR | 17.21 | 12.99 | 13.17 | 13.00 | 13.17 | 13.42 | 13.39 | 13.68 | 13.24 | 12.85 | **12.46** | 13.15 | 12.51 | 13.02 | 12.57 | brotli +dict | 0.9% |
| 153 | [lucky-charm](https://js13kgames.com/2025/games/lucky-charm) | 17.50 | 12.87 | 13.14 | 12.88 | 13.14 | 13.58 | 13.28 | 13.66 | 13.14 | 12.63 | **12.05** | 13.19 | 12.27 | 12.87 | 12.16 | brotli +dict | 0.9% |
| 154 | [triska-the-ninja-cat](https://js13kgames.com/2025/games/triska-the-ninja-cat) RR | 17.00 | 12.99 | 13.06 | 12.86 | 13.06 | 13.34 | 13.32 | 13.57 | 13.01 | 12.54 | **12.04** | 12.96 | 12.10 | 12.78 | 12.14 | brotli +dict | 0.9% |
| 155 | [they-took-my-cat](https://js13kgames.com/2025/games/they-took-my-cat) RR | 17.17 | 12.97 | 13.17 | 12.97 | 13.17 | 13.42 | 13.36 | 13.66 | 13.22 | 12.80 | **12.40** | 13.13 | 12.48 | 12.97 | 12.51 | brotli +dict | 0.9% |
| 156 | [midnight-paws](https://js13kgames.com/2025/games/midnight-paws) RR | 16.92 | 12.92 | 13.29 | 13.02 | 13.29 | 13.59 | 13.57 | 13.78 | 12.97 | 12.47 | **11.98** | 12.93 | 12.04 | 12.71 | 12.09 | brotli +dict | 0.9% |
| 157 | [echoes-of-nyx](https://js13kgames.com/2025/games/echoes-of-nyx) RR | 17.21 | 12.99 | 13.18 | 13.00 | 13.18 | 13.43 | 13.40 | 13.70 | 13.25 | 12.90 | **12.46** | 13.16 | 12.51 | 13.02 | 12.57 | brotli +dict | 0.9% |
| 158 | [black-cat-detective-agency](https://js13kgames.com/2025/games/black-cat-detective-agency) RR | 17.08 | 12.99 | 13.20 | 12.98 | 13.20 | 13.47 | 13.41 | 13.72 | 13.14 | 12.75 | **12.33** | 13.03 | 12.37 | 12.91 | 12.44 | brotli +dict | 0.9% |
| 159 | [cat-survivors](https://js13kgames.com/2025/games/cat-survivors) | 17.09 | 12.91 | 13.08 | 12.91 | 13.08 | 13.30 | 13.31 | 13.59 | 13.17 | 12.78 | **12.42** | 13.07 | 12.46 | 12.92 | 12.54 | brotli +dict | 0.9% |
| 160 | [witch-potion](https://js13kgames.com/2025/games/witch-potion) | 16.97 | 12.82 | 13.00 | 12.83 | 13.00 | 13.23 | 13.21 | 13.51 | 13.07 | 12.69 | **12.30** | 12.99 | 12.36 | 12.84 | 12.42 | brotli +dict | 1.0% |
| 161 | [blackcathacker](https://js13kgames.com/2025/games/blackcathacker) RR | 17.17 | 12.96 | 13.13 | 12.96 | 13.13 | 13.33 | 13.36 | 13.65 | 13.22 | 12.79 | **12.48** | 13.11 | 12.54 | 12.98 | 12.60 | brotli +dict | 1.0% |
| 162 | [black-cat-nine-lives-no-mercy](https://js13kgames.com/2025/games/black-cat-nine-lives-no-mercy) RR | 16.81 | 12.67 | 12.91 | 12.67 | 12.91 | 13.16 | 13.06 | 13.38 | 12.91 | 12.45 | **11.97** | 12.87 | 12.03 | 12.67 | 12.08 | brotli +dict | 1.0% |
| 163 | [catculus](https://js13kgames.com/2025/games/catculus) RR | 16.99 | 12.99 | 13.03 | 12.84 | 13.03 | 13.26 | 13.21 | 13.53 | 13.07 | 12.63 | **12.21** | 13.00 | 12.27 | 12.84 | 12.33 | brotli +dict | 1.0% |
| 164 | [clawstrike](https://js13kgames.com/2025/games/clawstrike) RR | 17.28 | 12.99 | 13.25 | 12.99 | 13.25 | 13.48 | 13.39 | 13.75 | 13.25 | 12.76 | **12.24** | 13.18 | 12.29 | 12.99 | 12.36 | brotli +dict | 1.0% |
| 165 | [zabobon](https://js13kgames.com/2025/games/zabobon) | 17.15 | 12.95 | 13.12 | 12.96 | 13.12 | 13.33 | 13.36 | 13.65 | 13.21 | 12.86 | **12.52** | 13.08 | 12.57 | 12.98 | 12.64 | brotli +dict | 1.0% |
| 166 | [flykt](https://js13kgames.com/2025/games/flykt) RR | 16.86 | 13.00 | 12.94 | 12.74 | 12.94 | 13.16 | 13.13 | 13.44 | 12.99 | 12.58 | **12.16** | 12.92 | 12.22 | 12.75 | 12.28 | brotli +dict | 1.0% |
| 167 | [onamazu](https://js13kgames.com/2025/games/onamazu) | 17.21 | 13.00 | 13.18 | 13.00 | 13.18 | 13.41 | 13.41 | 13.70 | 13.27 | 12.88 | **12.49** | 13.18 | 12.54 | 13.03 | 12.61 | brotli +dict | 1.0% |
| 168 | [black-cat-on-a-hot-tin-roof](https://js13kgames.com/2025/games/black-cat-on-a-hot-tin-roof) RR | 15.73 | 11.77 | 12.04 | 11.77 | 12.04 | 12.31 | 12.15 | 12.48 | 12.00 | 11.52 | **10.92** | 12.01 | 11.01 | 11.76 | 11.03 | brotli +dict | 1.0% |
| 169 | [i-got-up-on-the-wrong-foot](https://js13kgames.com/2025/games/i-got-up-on-the-wrong-foot) | 17.15 | 12.93 | 13.17 | 12.93 | 13.17 | 13.34 | 13.31 | 13.65 | 13.16 | 12.76 | **12.36** | 13.13 | 12.43 | 12.92 | 12.49 | brotli +dict | 1.0% |
| 170 | [whiskers-witch-adventure](https://js13kgames.com/2025/games/whiskers-witch-adventure) | 16.91 | 12.78 | 12.95 | 12.79 | 12.95 | 13.18 | 13.17 | 13.46 | 13.03 | 12.65 | **12.29** | 12.93 | 12.34 | 12.79 | 12.42 | brotli +dict | 1.0% |
| 171 | [the-familiar](https://js13kgames.com/2025/games/the-familiar) RR | 15.04 | 12.28 | 12.00 | 11.74 | 12.00 | 12.32 | 12.25 | 12.36 | 11.69 | 11.17 | **10.55** | 12.19 | 11.15 | 11.46 | 10.66 | brotli +dict | 1.0% |
| 172 | [witchcat](https://js13kgames.com/2025/games/witchcat) RR | 17.08 | 12.98 | 13.06 | 12.81 | 13.06 | 13.32 | 13.20 | 13.54 | 13.06 | 12.56 | **12.01** | 13.03 | 12.05 | 12.81 | 12.13 | brotli +dict | 1.1% |
| 173 | [kleptokitty](https://js13kgames.com/2025/games/kleptokitty) RR | 16.51 | 12.98 | 12.81 | 12.61 | 12.81 | 13.07 | 13.04 | 13.31 | 12.78 | 12.42 | **11.96** | 12.94 | 12.32 | 12.55 | 12.09 | brotli +dict | 1.1% |
| 174 | [l1ttl3-paws](https://js13kgames.com/2025/games/l1ttl3-paws) RR | 17.21 | 12.99 | 13.17 | 12.99 | 13.17 | 13.39 | 13.39 | 13.69 | 13.25 | 12.89 | **12.44** | 13.14 | 12.49 | 13.02 | 12.58 | brotli +dict | 1.1% |
| 175 | [snapcat](https://js13kgames.com/2025/games/snapcat) | 16.66 | 12.80 | 13.04 | 12.78 | 13.04 | 13.28 | 13.28 | 13.52 | 12.82 | 12.49 | **12.06** | 12.84 | 12.23 | 12.57 | 12.19 | brotli +dict | 1.1% |
| 176 | [super-benji](https://js13kgames.com/2025/games/super-benji) | 17.18 | 12.98 | 13.16 | 12.99 | 13.16 | 13.39 | 13.39 | 13.66 | 13.24 | 12.89 | **12.46** | 13.14 | 12.52 | 13.00 | 12.60 | brotli +dict | 1.1% |
| 177 | [miss-fortune](https://js13kgames.com/2025/games/miss-fortune) RR | 16.49 | 12.62 | 12.83 | 12.63 | 12.83 | 13.07 | 13.04 | 13.33 | 12.77 | 12.50 | **11.98** | 12.94 | 12.35 | 12.56 | 12.12 | brotli +dict | 1.1% |
| 178 | [i-am-the-night](https://js13kgames.com/2025/games/i-am-the-night) RR | 16.64 | 12.58 | 13.00 | 12.76 | 13.00 | 13.23 | 13.21 | 13.51 | 12.82 | 12.42 | **12.03** | 12.70 | 12.11 | 12.58 | 12.17 | brotli +dict | 1.2% |
| 179 | [pawfect-stack-oracle-of-order](https://js13kgames.com/2025/games/pawfect-stack-oracle-of-order) RR | 16.63 | 12.99 | 13.20 | 13.00 | 13.20 | 13.39 | 13.40 | 13.66 | 13.17 | 12.81 | **12.42** | 13.63 | 13.08 | 12.95 | 12.57 | brotli +dict | 1.2% |
| 180 | [pawfect-stack-oracle-of-chaos](https://js13kgames.com/2025/games/pawfect-stack-oracle-of-chaos) RR | 16.64 | 13.00 | 13.21 | 13.01 | 13.21 | 13.41 | 13.41 | 13.65 | 13.18 | 12.81 | **12.40** | 13.62 | 13.07 | 12.96 | 12.55 | brotli +dict | 1.2% |
| 181 | [paw-of-fate](https://js13kgames.com/2025/games/paw-of-fate) RR | 9.75 | 8.26 | 8.05 | 7.83 | 8.05 | 8.24 | 8.19 | 8.24 | 7.80 | 7.47 | **6.93** | 8.18 | 7.38 | 7.62 | 7.02 | brotli +dict | 1.2% |
| 182 | [the-ninefold-gambit](https://js13kgames.com/2025/games/the-ninefold-gambit) | 15.84 | 12.97 | 13.38 | 12.97 | 13.38 | 13.64 | 13.59 | 13.82 | 12.15 | 11.83 | **11.36** | 12.54 | 11.92 | 11.98 | 11.51 | brotli +dict | 1.3% |
| 183 | [cats-cargo](https://js13kgames.com/2025/games/cats-cargo) RR | 15.84 | 12.93 | 13.36 | 12.93 | 13.36 | 13.61 | 13.61 | 13.78 | 12.06 | 11.71 | **11.15** | 12.45 | 11.70 | 11.89 | 11.29 | brotli +dict | 1.3% |
| 184 | [witches-cauldron](https://js13kgames.com/2025/games/witches-cauldron) RR | 16.58 | 12.97 | 13.17 | 12.98 | 13.17 | 13.42 | 13.37 | 13.63 | 13.13 | 12.87 | **12.30** | 13.60 | 13.00 | 12.93 | 12.46 | brotli +dict | 1.3% |
| 185 | [ashes-of-ulthar](https://js13kgames.com/2025/games/ashes-of-ulthar) RR | 16.01 | 12.94 | 13.19 | 12.94 | 13.19 | 13.50 | 13.31 | 13.62 | 13.06 | 12.65 | **12.08** | 13.93 | 13.08 | 12.86 | 12.24 | brotli +dict | 1.3% |
| 186 | [kuro-neko-market](https://js13kgames.com/2025/games/kuro-neko-market) RR | 17.15 | 12.95 | 13.12 | 12.95 | 13.12 | 13.37 | 13.35 | 13.65 | 13.20 | 12.82 | **12.37** | 13.11 | 12.46 | 12.98 | 12.54 | brotli +dict | 1.3% |
| 187 | [feline-fury](https://js13kgames.com/2025/games/feline-fury) | 13.98 | 11.06 | 11.28 | 11.06 | 11.28 | 11.52 | 11.47 | 11.66 | 11.04 | 10.74 | **10.28** | 11.44 | 10.86 | 10.86 | 10.43 | brotli +dict | 1.4% |
| 188 | [black-cat-squadron](https://js13kgames.com/2025/games/black-cat-squadron) RR | 16.77 | 12.98 | 13.23 | 12.99 | 13.23 | 13.51 | 13.46 | 13.70 | 13.07 | 12.66 | **12.10** | 13.46 | 12.69 | 12.86 | 12.28 | brotli +dict | 1.5% |
| 189 | [oggys-curse](https://js13kgames.com/2025/games/oggys-curse) RR | 15.77 | 12.90 | 13.06 | 12.90 | 13.06 | 13.45 | 13.26 | 13.53 | 13.01 | 12.72 | **12.24** | 13.89 | 13.31 | 12.83 | 12.42 | brotli +dict | 1.5% |
| 190 | [bichita-the-cat](https://js13kgames.com/2025/games/bichita-the-cat) RR | 12.71 | 11.51 | 11.52 | 11.24 | 11.52 | 11.71 | 11.62 | 11.72 | 11.01 | 10.73 | **10.22** | 11.75 | 11.21 | 10.84 | 10.39 | brotli +dict | 1.6% |
| 191 | [cats-puzzle](https://js13kgames.com/2025/games/cats-puzzle) | 12.24 | 10.08 | 10.26 | 10.08 | 10.26 | 10.45 | 10.40 | 10.54 | 10.20 | 9.91 | **9.50** | 10.81 | 10.29 | 10.03 | 9.66 | brotli +dict | 1.7% |
| 192 | [blacky-returns-home](https://js13kgames.com/2025/games/blacky-returns-home) | 31.28 | 11.73 | 11.63 | 11.58 | 11.63 | 11.74 | 11.29 | 10.42 | 11.15 | 10.61 | **5.36** | 11.41 | 5.39 | 9.90 | 5.47 | brotli +dict | 2.1% |
| 193 | [nine-lives](https://js13kgames.com/2025/games/nine-lives) | 28.01 | 12.57 | 12.21 | 12.03 | 12.21 | 12.70 | 11.44 | 11.13 | 11.08 | 11.02 | 10.62 | 11.96 | 11.13 | **10.41** | 10.65 | model | 2.3% |
| 194 | [boing-kat](https://js13kgames.com/2025/games/boing-kat) | 63.37 | 12.23 | 11.70 | 11.36 | 11.54 | 11.53 | 11.29 | 10.33 | 10.35 | 9.85 | **5.21** | 10.91 | 5.22 | 9.23 | 5.34 | brotli +dict | 2.6% |
| 195 | [black-kate](https://js13kgames.com/2025/games/black-kate) RR | 78.41 | 12.25 | 12.56 | 12.28 | 12.60 | 13.57 | 12.45 | 13.35 | 12.05 | 11.71 | **11.26** | 12.28 | 11.54 | 12.06 | 11.67 | brotli +dict | 3.7% |
| 196 | [laser-pointer-cat](https://js13kgames.com/2025/games/laser-pointer-cat) RR | 12.87 | 7.94 | 7.84 | 7.54 | 7.84 | 8.15 | 7.81 | 7.99 | 7.48 | 7.21 | **6.56** | 7.83 | 6.84 | 7.32 | 6.92 | brotli +dict | 5.6% |
| 197 | [carbons-kitty-care](https://js13kgames.com/2025/games/carbons-kitty-care) | 72.53 | 12.79 | 11.10 | 10.95 | 10.94 | 11.06 | 10.52 | 9.30 | 9.62 | 8.97 | 2.98 | 9.91 | **2.93** | 8.12 | 3.25 | zstd +dict | 11.2% |
<!-- results:end -->
