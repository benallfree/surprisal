// Dump the text parts of all 2020-2024 archives (same split rule as split.mjs) for LLM training.
//   node llm/dump_train.mjs   -> llm/train.bin (archives separated by a 0x00 byte), llm/val.bin (last 3%)
import fs from 'node:fs'
import { unzip, unzip7z, blobOf } from '../zipread.mjs'

const root = new URL('..', import.meta.url).pathname
const BIN = /\.(png|gif|jpe?g|webp|mp3|ogg|wav|zip|woff2?|ico|bmp|glb|wasm|bin)$/i
const utf8 = new TextDecoder('utf-8', { fatal: true })
const isText = (f) => { if (BIN.test(f.name)) return false; try { utf8.decode(f.data); return true } catch { return false } }
const parts = []
for (const y of ['2020', '2021', '2022', '2023', '2024'])
  for (const f of fs.readdirSync(`${root}corpus/zips/${y}`).sort()) {
    const zp = `${root}corpus/zips/${y}/${f}`
    let files
    try { files = unzip(fs.readFileSync(zp)) } catch { try { files = unzip7z(zp) } catch { continue } }
    parts.push(blobOf(files.filter(isText)), Buffer.from([0]))
  }
// shuffle archive order deterministically so validation isn't all one year
const pairs = []
for (let i = 0; i < parts.length; i += 2) pairs.push(parts[i])
let s = 12345
for (let i = pairs.length - 1; i > 0; i--) { s = (s * 1103515245 + 12345) >>> 0; const j = s % (i + 1); [pairs[i], pairs[j]] = [pairs[j], pairs[i]] }
const nVal = Math.round(pairs.length * 0.03)
const join = (ps) => Buffer.concat(ps.flatMap((p) => [p, Buffer.from([0])]))
fs.writeFileSync(root + 'llm/train.bin', join(pairs.slice(nVal)))
fs.writeFileSync(root + 'llm/val.bin', join(pairs.slice(0, nVal)))
console.log(`archives ${pairs.length} (val ${nVal}); train ${fs.statSync(root + 'llm/train.bin').size} bytes, val ${fs.statSync(root + 'llm/val.bin').size} bytes`)
