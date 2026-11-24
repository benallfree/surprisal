// Wall clock and size for every non-LLM compressor on one file: compress, decompress, verify, median of 3 runs.
//   node time_classic.mjs <file>   -> markdown table on stdout, JSON in timing-classic.json
// Times are whole-process wall clock (spawn included) for the CLI tools, in-process for the context mixer.
// The trained context mixer's pretraining (a one-time setup a shipped codec would load as a snapshot) is timed apart.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { Model, encode, decode } from './codec.mjs'

const src = path.resolve(process.argv[2])
const input = fs.readFileSync(src)
const T = fs.mkdtempSync(path.join(os.tmpdir(), 'tc-'))
fs.copyFileSync(src, path.join(T, 'index.html'))
const sh = (cmd) => execFileSync('/bin/sh', ['-c', cmd], { cwd: T, maxBuffer: 1 << 28 })
const med = (a) => [...a].sort((x, y) => x - y)[a.length >> 1]
const time = (fn) => { const t = performance.now(); fn(); return (performance.now() - t) / 1000 }

const zip = (m) => `rm -f o.zip && 7z a -tzip -bd -mx=9 ${m} o.zip index.html >/dev/null`
const tools = [
  ['zip Deflate (7-Zip, 15 passes)', zip('-mm=Deflate -mpass=15 -mfb=258'), 'o.zip', 'unzip -p o.zip index.html'],
  ['zip Zopfli (advzip -4)', `${zip('')} && advzip -q -z -4 o.zip`, 'o.zip', 'unzip -p o.zip index.html'],
  ['zip LZMA (7-Zip)', zip('-mm=LZMA -md=64m -mfb=273'), 'o.zip', '7z e -so o.zip index.html 2>/dev/null'],
  ['zip PPMd order 6 (7-Zip)', zip('-mm=PPMd -mo=6 -mmem=256m'), 'o.zip', '7z e -so o.zip index.html 2>/dev/null'],
  ['gzip -9', 'gzip -9c index.html > o.gz', 'o.gz', 'gzip -dc o.gz'],
  ['bzip2 -9', 'bzip2 -9c index.html > o.bz2', 'o.bz2', 'bzip2 -dc o.bz2'],
  ['xz -9e', 'xz -9e -c index.html > o.xz', 'o.xz', 'xz -dc o.xz'],
  ['zstd --ultra -22', 'zstd -q -f --ultra -22 index.html -o o.zst', 'o.zst', 'zstd -q -dc o.zst'],
  ['brotli -q11', 'brotli -f -q 11 -w 24 index.html -o o.br', 'o.br', 'brotli -dc o.br'],
]

const rows = []
for (const [name, comp, out, dec] of tools) {
  const c = [], d = []
  let size = 0
  for (let i = 0; i < 3; i++) {
    c.push(time(() => sh(comp)))
    size = fs.statSync(path.join(T, out)).size
    let back
    d.push(time(() => { back = sh(dec) }))
    if (!back.equals(input)) throw new Error(`${name}: round trip failed`)
  }
  rows.push({ name, bytes: size, compress_s: med(c), decompress_s: med(d) })
  process.stderr.write(`${name}: ${size} bytes\n`)
}

// Context mixer, untrained and pretrained on 2020-2024 js13k entries.
const cm = (name, model, setup) => {
  const snap = model.snapshot(), c = [], d = []
  let code
  for (let i = 0; i < 3; i++) {
    model.restore(snap); c.push(time(() => { code = encode(model, input) }))
    let back
    model.restore(snap); d.push(time(() => { back = decode(model, code) }))
    if (!Buffer.from(back).equals(input)) throw new Error(`${name}: round trip failed`)
  }
  rows.push({ name, bytes: code.length, compress_s: med(c), decompress_s: med(d), setup_s: setup })
  process.stderr.write(`${name}: ${code.length} bytes\n`)
}
cm('context mixer, untrained', new Model(), 0)
const trained = new Model()
const blobs = [2020, 2021, 2022, 2023, 2024].flatMap((y) =>
  fs.readdirSync(`corpus/blobs/${y}`).sort().map((f) => fs.readFileSync(`corpus/blobs/${y}/${f}`)))
const setup = time(() => { for (const b of blobs) trained.pretrain(b) })
cm('context mixer, trained on 2020-24', trained, setup)

fs.rmSync(T, { recursive: true, force: true })
fs.writeFileSync('timing-classic.json', JSON.stringify({ file: process.argv[2], raw: input.length, rows }, null, 1))
const f = (s) => (s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(1)} s`)
console.log(`| compressor | archive bytes | compress | decompress |\n|---|---:|---:|---:|`)
for (const r of [...rows].sort((a, b) => a.bytes - b.bytes))
  console.log(`| ${r.name}${r.setup_s ? ` (one-time pretraining ${f(r.setup_s)})` : ''} | ${r.bytes.toLocaleString()} | ${f(r.compress_s)} | ${f(r.decompress_s)} |`)
