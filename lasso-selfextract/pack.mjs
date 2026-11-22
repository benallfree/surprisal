// Model-assisted compression: context-mixing predictor + arithmetic coder.
//   node pack.mjs              pack index.html -> dist/packed.html, write dist/report.html, compare zips
//   node pack.mjs tune         grid-search the model's three knobs on index.html's script
//   node pack.mjs stats FILE   bits/char for any file (round-trip checked)
import fs from 'node:fs'
import zlib from 'node:zlib'
import { execFileSync } from 'node:child_process'

const dir = new URL('.', import.meta.url).pathname
const CORE = fs.readFileSync(dir + 'core.js', 'utf8').replace(/\/\*[\s\S]*?\*\//, '').trim()
export const DEFAULTS = { R1: 24000, R2: 0.75, R3: 30, R4: 'c', R5: '[0,1,2,3,4,6]' } // weight init, mixer learning rate, count cap

const core = (k) => CORE.replace(/R[1-5]/g, (r) => k[r])
const Zof = (k) => new Function(core(k) + ';return Z')()

export function encode(bytes, k = DEFAULTS) {
  const out = [], cost = new Float64Array(bytes.length)
  let L = 0, U = 0xffffffff, b = 0
  Zof(k)(bytes.length, (p) => {
    const y = (bytes[b >> 3] >> (7 - (b & 7))) & 1
    cost[b >> 3] -= Math.log2(y ? p / 4096 : 1 - p / 4096) // report only, never shipped
    b++
    const m = L + ((U - L) >>> 12) * p
    y ? (U = m) : (L = m + 1)
    while (!((L ^ U) >>> 24)) out.push(U >>> 24), (L = (L << 8) >>> 0), (U = ((U << 8) | 255) >>> 0)
    return y
  })
  out.push(L & 0xffffff ? (L >>> 24) + 1 : L >>> 24)
  return { out: Uint8Array.from(out), cost }
}

export function decode(code, n, k = DEFAULTS) {
  const g = (i) => code[i] | 0
  let L = 0, U = 0xffffffff, x = 0, q = 0
  while (q < 4) x = ((x << 8) | g(q++)) >>> 0
  return Zof(k)(n, (p) => {
    const m = L + ((U - L) >>> 12) * p, y = x <= m
    y ? (U = m) : (L = m + 1)
    while (!((L ^ U) >>> 24)) (L = (L << 8) >>> 0), (U = ((U << 8) | 255) >>> 0), (x = ((x << 8) | g(q++)) >>> 0)
    return +y
  })
}

const roundtrip = (bytes, k) => {
  const { out, cost } = encode(bytes, k)
  const back = decode(out, bytes.length, k)
  if (Buffer.compare(Buffer.from(back), Buffer.from(bytes))) throw new Error('round-trip mismatch')
  return { out, cost }
}

// The decoder that ships: base64 payload -> same core -> TextDecoder -> indirect eval.
const decoderJs = (b64, n, k) =>
  `D=atob("${b64}");g=i=>D.charCodeAt(i)|0;for(X=q=0;q<4;)X=(X<<8|g(q++))>>>0;L=0;U=-1>>>0;` +
  core(k) +
  `;(0,eval)(new TextDecoder().decode(Z(${n},p=>{let m=L+(U-L>>>12)*p,y=X<=m;y?U=m:L=m+1;for(;!((L^U)>>>24);)L=L<<8>>>0,U=(U<<8|255)>>>0,X=(X<<8|g(q++))>>>0;return y})))`

const zipSize = (file) => {
  const z = file.replace(/\.html$/, '.zip')
  fs.rmSync(z, { force: true })
  fs.copyFileSync(file, dir + 'dist/index.html')
  execFileSync('zip', ['-q', '-9', '-X', '-j', z, dir + 'dist/index.html'])
  try { execFileSync('advzip', ['-q', '-z', '-4', z]) } catch {}
  fs.rmSync(dir + 'dist/index.html')
  return fs.statSync(z).size
}

function report(text, cost, title) {
  const esc = (s) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`)
  let bi = 0, html = ''
  for (const ch of text) {
    const nb = Buffer.byteLength(ch)
    let bits = 0
    for (let i = 0; i < nb; i++) bits += cost[bi++]
    const heat = Math.min(1, bits / 8)
    html += `<span style="background:hsla(${(1 - heat) * 220},90%,50%,${0.08 + heat * 0.8})" title="${bits.toFixed(2)} bits">${ch == '\n' ? '⏎\n' : esc(ch)}</span>`
  }
  const total = cost.reduce((a, b) => a + b, 0)
  return `<!doctype html><meta charset=utf-8><title>${title}</title><style>body{background:#0b0b12;color:#ddd;font:13px/1.45 ui-monospace,monospace;margin:24px}pre{white-space:pre-wrap;word-break:break-all}h1{font-size:16px}.k span{padding:0 6px}</style>
<h1>${title}: ${(total / 8) | 0} bytes for ${text.length} chars, ${(total / text.length).toFixed(2)} bits/char</h1>
<p class=k>cost per character: <span style="background:hsla(220,90%,50%,.1)">~0 bits</span><span style="background:hsla(110,90%,50%,.5)">4 bits</span><span style="background:hsla(0,90%,50%,.9)">8+ bits</span> (hover for exact)</p><pre>${html}</pre>`
}

const scriptOf = (html) => {
  const m = html.match(/<script>([\s\S]*?)<\/script>/)
  if (!m) throw new Error('no <script> in input')
  return m
}

const [cmd, arg] = process.argv.slice(2)
if (import.meta.url.includes('?')) {
  // imported by an experiment script: export only
} else if (cmd === 'stats') {
  // Size-only comparison for any file: model payload + shipped decoder vs. deflate of the raw file.
  const bytes = fs.readFileSync(arg)
  const { out } = roundtrip(bytes, DEFAULTS)
  const defl = zlib.deflateRawSync(bytes, { level: 9 }).length
  const dec = zlib.deflateRawSync(decoderJs('', bytes.length, DEFAULTS), { level: 9 }).length
  const tot = out.length + dec // base64 adds ~1% after deflate; ignored here
  console.log(`${arg}: ${bytes.length} bytes, round-trip ok`)
  console.log(`  deflate -9          ${defl}`)
  console.log(`  model payload       ${out.length}  (${((out.length * 8) / bytes.length).toFixed(3)} bits/byte, ${(((out.length - defl) / defl) * 100).toFixed(1)}% vs deflate)`)
  console.log(`  + decoder (zipped)  ${tot}  (${(((tot - defl) / defl) * 100).toFixed(1)}% vs deflate)`)
} else if (cmd === 'tune') {
  const bytes = Buffer.from(scriptOf(fs.readFileSync(dir + 'index.html', 'utf8'))[1])
  let best
  for (const R1 of [8000, 16000, 24000, 32000])
    for (const R2 of [0.5, 0.75, 1, 2])
      for (const R3 of [20, 30, 60, 127]) {
        const n = encode(bytes, { ...DEFAULTS, R1, R2, R3 }).out.length
        if (!best || n < best.n) (best = { n, R1, R2, R3 }), console.log(JSON.stringify(best))
      }
} else {
  const html = fs.readFileSync(dir + 'index.html', 'utf8')
  const m = scriptOf(html)
  const text = m[1], bytes = Buffer.from(text)
  console.log(`${new Date().toTimeString().slice(0, 8)} encoding ${bytes.length} bytes of JS`)
  const { out, cost } = roundtrip(bytes, DEFAULTS)
  console.log(`model: ${bytes.length} -> ${out.length} bytes (${((out.length * 8) / text.length).toFixed(3)} bits/char), round-trip ok`)
  const js = decoderJs(Buffer.from(out).toString('base64'), bytes.length, DEFAULTS)
  fs.mkdirSync(dir + 'dist', { recursive: true })
  const packed = html.slice(0, m.index) + '<script>' + js + '</script>' + html.slice(m.index + m[0].length)
  fs.writeFileSync(dir + 'dist/packed.html', packed)
  fs.copyFileSync(dir + 'index.html', dir + 'dist/plain.html')
  fs.writeFileSync(dir + 'dist/report.html', report(text, cost, 'LASSO under the model'))
  const zp = zipSize(dir + 'dist/plain.html'), zk = zipSize(dir + 'dist/packed.html')
  console.log(`decoder code: ${core(DEFAULTS).length + 330} bytes (approx, unzipped)`)
  console.log(`plain.zip   ${zp} bytes`)
  console.log(`packed.zip  ${zk} bytes  (${(((zk - zp) / zp) * 100).toFixed(1)}% vs zip)`)
}
