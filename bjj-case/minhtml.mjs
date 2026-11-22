// Minimal HTML+CSS minifier: leaves <script> bodies alone (JS is already terser'd).
//   node minhtml.mjs in.html out.html
import fs from 'node:fs'
const [, , inp, out] = process.argv
let h = fs.readFileSync(inp, 'utf8')
const keep = []
h = h.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, (m) => `\u0000${keep.push(m) - 1}\u0000`)
const css = (c) => c
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/\s+/g, ' ')
  .replace(/\s*([{}:;,>+~])\s*/g, '$1')
  .replace(/;}/g, '}')
  .replace(/:0(px|em|rem|%)/g, ':0')
  .trim()
h = h.replace(/(<style\b[^>]*>)([\s\S]*?)(<\/style>)/gi, (m, a, c, b) => a + css(c) + b)
h = h.replace(/\sstyle="([^"]*)"/gi, (m, c) => ` style="${css(c)}"`)
h = h
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/>\s+</g, '><')
  .replace(/\s{2,}/g, ' ')
  .replace(/\s*\n\s*/g, ' ')
  .trim()
h = h.replace(/\u0000(\d+)\u0000/g, (m, i) => keep[+i])
fs.writeFileSync(out, h)
console.log(`${inp} ${fs.statSync(inp).size} -> ${out} ${h.length}`)
