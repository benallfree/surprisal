// Split each 2025 test archive into a text part (UTF-8 code/markup, LLM-scorable) and a binary part,
// and cost both with the context-mixing codec (untrained and pretrained on 2020-2024) plus brotli.
//   node --max-old-space-size=4096 llm/split.mjs   -> llm/data/<slug>.txt, llm/split.json
import fs from 'node:fs'
import { execFileSync } from 'node:child_process'
import { unzip, unzip7z, blobOf } from '../zipread.mjs'
import { Model, cost } from '../codec.mjs'

const root = new URL('..', import.meta.url).pathname
const log = (...a) => console.log(new Date().toTimeString().slice(0, 8), ...a)
const BIN = /\.(png|gif|jpe?g|webp|mp3|ogg|wav|zip|woff2?|ico|bmp|glb|wasm|bin)$/i
const utf8 = new TextDecoder('utf-8', { fatal: true })
const isText = (f) => { if (BIN.test(f.name)) return false; try { utf8.decode(f.data); return true } catch { return false } }

const test = JSON.parse(fs.readFileSync(root + 'results.json', 'utf8')).map((g) => g.slug)
fs.mkdirSync(root + 'llm/data', { recursive: true })
const games = test.map((slug) => {
  const zp = `${root}corpus/zips/2025/${slug}.zip`
  let files
  try { files = unzip(fs.readFileSync(zp)) } catch { files = unzip7z(zp) }
  const text = blobOf(files.filter(isText)), bin = blobOf(files.filter((f) => !isText(f)))
  fs.writeFileSync(`${root}llm/data/${slug}.txt`, text)
  return { slug, text, bin }
})
log(`split ${games.length} games: text ${(games.reduce((a, g) => a + g.text.length, 0) / 1e6).toFixed(2)} MB, binary ${(games.reduce((a, g) => a + g.bin.length, 0) / 1e6).toFixed(2)} MB`)

const raw16 = root + 'corpus/raw16.dict'
const brotli = (file, dict) => execFileSync('brotli', ['-c', '-q', '11', '-w', '24', ...(dict ? ['-D', dict] : []), file], { maxBuffer: 1 << 27 }).length
const bytesOf = (bits) => Math.ceil(bits / 8)
const out = {}
const m = new Model()
let snap = m.snapshot()
for (const g of games) {
  m.restore(snap)
  const f = `${root}llm/data/${g.slug}.txt`
  out[g.slug] = { textBytes: g.text.length, binBytes: g.bin.length, text_m0: bytesOf(cost(m, g.text)), text_brotli: brotli(f), text_brotliDict: brotli(f, raw16) }
  m.restore(snap); out[g.slug].bin_m0 = g.bin.length ? bytesOf(cost(m, g.bin)) : 0
}
log('untrained costs done; pretraining on 2020-2024')
for (const y of ['2020', '2021', '2022', '2023', '2024'])
  for (const f of fs.readdirSync(`${root}corpus/blobs/${y}`).sort()) m.pretrain(fs.readFileSync(`${root}corpus/blobs/${y}/${f}`))
snap = m.snapshot()
for (const g of games) {
  m.restore(snap); out[g.slug].text_m5 = bytesOf(cost(m, g.text))
  m.restore(snap); out[g.slug].bin_m5 = g.bin.length ? bytesOf(cost(m, g.bin)) : 0
}
fs.writeFileSync(root + 'llm/split.json', JSON.stringify(out, null, 1))
const tot = (k) => Object.values(out).reduce((a, r) => a + r[k], 0)
log(`text part totals (bytes): m0 ${tot('text_m0')}, m5 ${tot('text_m5')}, brotli ${tot('text_brotli')}, brotli+dict ${tot('text_brotliDict')}`)
