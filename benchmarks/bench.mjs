// Benchmark: does a codec pretrained on past js13k entries fit more game into archive.13k?
//   node bench.mjs        (needs corpus/zips from corpus/fetch.sh; writes corpus/blobs, results.json)
// Train years 2020-2024, test year 2025: test games are never seen during training.
import fs from 'node:fs'
import zlib from 'node:zlib'
import { execFileSync } from 'node:child_process'
import { Model, cost, encode, decode } from './codec.mjs'

const dir = new URL('.', import.meta.url).pathname
const T = () => new Date().toTimeString().slice(0, 8)
const log = (...a) => console.log(T(), ...a)

import { unzip, unzip7z, blobOf, writeFiles } from "./zipread.mjs"

const BIN = /\.(png|gif|jpe?g|webp|mp3|ogg|wav|zip|woff2?|ico|bmp|glb)$/i

// --- load corpus ---
const games = []
for (const year of fs.readdirSync(dir + 'corpus/zips').sort()) {
  for (const f of fs.readdirSync(`${dir}corpus/zips/${year}`).sort()) {
    const buf = fs.readFileSync(`${dir}corpus/zips/${year}/${f}`)
    try {
      let files
      try { files = unzip(buf) } catch { files = unzip7z(`${dir}corpus/zips/${year}/${f}`) } // LZMA/PPMd/Deflate64 zips
      const blob = blobOf(files)
      const binBytes = files.filter((x) => BIN.test(x.name)).reduce((a, x) => a + x.data.length, 0)
      games.push({ year: +year, slug: f.slice(0, -4), zip: buf.length, raw: blob.length, binFrac: binBytes / blob.length, blob, files })
    } catch (err) {
      log('skip', year, f, err.message)
    }
  }
}
const train = games.filter((g) => g.year < 2025), test = games.filter((g) => g.year === 2025)
log(`corpus: ${games.length} games (${train.length} train 2020-24, ${test.length} test 2025), train bytes ${(train.reduce((a, g) => a + g.raw, 0) / 1e6).toFixed(1)} MB`)

// --- zstd baselines (CLI): plain and with a dictionary trained on the train years ---
fs.rmSync(dir + 'corpus/blobs', { recursive: true, force: true })
for (const g of games) {
  fs.mkdirSync(`${dir}corpus/blobs/${g.year}`, { recursive: true })
  fs.writeFileSync((g.file = `${dir}corpus/blobs/${g.year}/${g.slug}.bin`), g.blob)
}
const dict = dir + 'corpus/zstd.dict'
log('training zstd dictionary (1 MB max)')
execFileSync('zstd', ['-q', '-f', '--train', ...train.map((g) => g.file), '--maxdict=1048576', '-o', dict], { stdio: 'inherit' })
// raw-content dictionary: the most recent 16 MB of training data, verbatim
const raw16 = dir + 'corpus/raw16.dict'
{ const all = Buffer.concat(train.map((g) => g.blob)); fs.writeFileSync(raw16, all.subarray(Math.max(0, all.length - (16 << 20)))) }
const run = (cmd, args) => execFileSync(cmd, args, { maxBuffer: 1 << 27 }).length
for (const [i, g] of test.entries()) {
  g.zopfli = run('zopfli', ['-c', '--deflate', '--i15', g.file]) // Google: optimal DEFLATE (zip's format, minus headers)
  g.xz = run('xz', ['-c', '--format=raw', '--lzma2=preset=9e', g.file])
  g.brotli = run('brotli', ['-c', '-q', '11', '-w', '24', g.file]) // Google: Brotli (has a built-in web dictionary)
  g.brotliDict = Math.min(run('brotli', ['-c', '-q', '11', '-w', '24', '-D', dict, g.file]), run('brotli', ['-c', '-q', '11', '-w', '24', '-D', raw16, g.file]))
  g.zstd = run('zstd', ['-q', '-c', '-19', g.file])
  g.zstdDict = Math.min(run('zstd', ['-q', '-c', '-19', '-D', dict, g.file]), run('zstd', ['-q', '-c', '--ultra', '-22', '--long=27', '-D', raw16, g.file]))
  // Legal js13k zips with every method 7-Zip can write, max settings; best per game is the real opponent.
  const fd = `${dir}corpus/files/${g.slug}`, zp = `${dir}corpus/tmp.zip`
  writeFiles(fd, g.files)
  const z7 = (...a) => { fs.rmSync(zp, { force: true }); execFileSync('7z', ['a', '-tzip', '-bd', '-mx=9', ...a, zp, '.'], { cwd: fd, stdio: 'ignore' }); return fs.statSync(zp).size }
  g.zDeflate = z7('-mm=Deflate', '-mfb=258', '-mpass=15')
  try { execFileSync('advzip', ['-q', '-z', '-4', zp]); g.zZopfli = fs.statSync(zp).size } catch { g.zZopfli = g.zDeflate }
  g.zDeflate64 = z7('-mm=Deflate64', '-mfb=257', '-mpass=15')
  g.zBzip2 = z7('-mm=BZip2')
  g.zLzma = z7('-mm=LZMA', '-mfb=273')
  g.zPpmd = Math.min(...[4, 6, 8, 12, 16].map((o) => z7('-mm=PPMd', '-mo=' + o, '-mmem=256m')))
  g.zBest = Math.min(g.zip, g.zDeflate, g.zZopfli, g.zDeflate64, g.zBzip2, g.zLzma, g.zPpmd)
  delete g.files
  if (i % 25 === 0) log(`baselines ${i}/${test.length}`)
}
log('baselines done')

// --- model: untrained, then after each additional training year ---
const model = new Model()
const evalAll = (label) => {
  const snap = model.snapshot()
  let t0 = Date.now()
  for (const g of test) {
    model.restore(snap)
    g[label] = Math.ceil(cost(model, g.blob) / 8) + 6 // + length header + coder flush
  }
  model.restore(snap)
  log(`${label}: evaluated ${test.length} games in ${((Date.now() - t0) / 1000).toFixed(0)}s`)
}
evalAll('m0')
const labels = ['m0']
for (const y of [2020, 2021, 2022, 2023, 2024]) {
  const t0 = Date.now()
  for (const g of train.filter((g) => g.year === y)) model.pretrain(g.blob)
  log(`pretrained on ${y} (${((Date.now() - t0) / 1000).toFixed(0)}s, history ${(model.pos / 1e6).toFixed(1)} MB)`)
  const l = 'm' + (y - 2019)
  evalAll(l)
  labels.push(l)
}

// --- prove the numbers are real: full encode/decode round-trip on a few test games ---
const snap = model.snapshot()
for (const g of test.slice(0, 3)) {
  model.restore(snap); const code = encode(model, g.blob)
  model.restore(snap); const back = decode(model, code)
  log(`round-trip ${g.slug}: ${code.length} bytes (estimate ${g.m5}) ${Buffer.compare(Buffer.from(back), g.blob) ? 'MISMATCH' : 'ok'}`)
}

// --- summary ---
const sum = (k, gs = test) => gs.reduce((a, g) => a + g[k], 0)
const rows = [['submitted .zip (as entered)', 'zip'], ['zip: Deflate (7-Zip, 15 passes)', 'zDeflate'], ['zip: Deflate via Zopfli (advzip -4)', 'zZopfli'], ['zip: Deflate64', 'zDeflate64'], ['zip: BZip2', 'zBzip2'], ['zip: LZMA', 'zLzma'], ['zip: PPMd (best order)', 'zPpmd'], ['**best legal zip per game**', 'zBest'],
  ['zopfli raw deflate stream (no zip headers)', 'zopfli'], ['xz -9e (raw lzma2)', 'xz'], ['brotli -q11 -w24', 'brotli'], ['zstd -19', 'zstd'],
  ['brotli + trained dict (best of 1MB zstd-trained, 16MB raw)', 'brotliDict'], ['zstd + trained dict (best of 1MB trained, 16MB raw --long)', 'zstdDict'], ['model, untrained', 'm0'],
  ...[1, 2, 3, 4, 5].map((n) => [`model, pretrained ${n} yr${n > 1 ? 's' : ''} (2020-${2019 + n})`, 'm' + n])]
const groups = { all: test, 'mostly code (<20% binary assets)': test.filter((g) => g.binFrac < 0.2), 'asset-heavy (>=20% binary)': test.filter((g) => g.binFrac >= 0.2) }
let md = ''
for (const [name, gs] of Object.entries(groups)) {
  md += `\n### 2025 test set, ${name}: ${gs.length} games, ${(sum('raw', gs) / 1024).toFixed(0)} KiB unpacked\n\n| codec | total KiB | vs submitted zip | vs best legal zip |\n|---|---:|---:|---:|\n`
  const z = sum('zip', gs), zb = sum('zBest', gs), pct = (a, b) => (((a - b) / b) * 100).toFixed(1) + '%'
  for (const [lab, k] of rows) md += `| ${lab} | ${(sum(k, gs) / 1024).toFixed(1)} | ${pct(sum(k, gs), z)} | ${pct(sum(k, gs), zb)} |\n`
}
const med = (f) => test.map(f).sort((a, b) => a - b)[test.length >> 1].toFixed(3)
md += `\nMedian per-game ratio vs best legal zip: model untrained ${med((g) => g.m0 / g.zBest)}, pretrained 5 yrs ${med((g) => g.m5 / g.zBest)}, brotli+dict ${med((g) => g.brotliDict / g.zBest)}, zstd+dict ${med((g) => g.zstdDict / g.zBest)}\n`
md += `Capacity: a 13,312-byte archive.13k holds about ${(13312 / +med((g) => g.m5 / g.zBest) / 1024).toFixed(1)} KiB of best-zip-equivalent game with the pretrained model.\n`
console.log(md)
fs.writeFileSync(dir + 'results.md', md)
fs.writeFileSync(dir + 'results.json', JSON.stringify(test.map(({ blob, file, ...g }) => g), null, 1))
log('wrote results.md, results.json')
