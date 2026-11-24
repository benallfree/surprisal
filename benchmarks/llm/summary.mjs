// Merge LLM scores (llm/out/*.json) with the text/binary split costs (llm/split.json) and the main
// results (results.json); write a section into STUDY.md between <!-- llm:start --> and <!-- llm:end -->.
//   node llm/summary.mjs
import fs from 'node:fs'

const root = new URL('..', import.meta.url).pathname
const split = JSON.parse(fs.readFileSync(root + 'llm/split.json', 'utf8'))
const main = Object.fromEntries(JSON.parse(fs.readFileSync(root + 'results.json', 'utf8')).map((g) => [g.slug, g]))
const rr = new Set(fs.readFileSync(root + 'corpus/roadroller.txt', 'utf8').split('\n').filter((l) => l.startsWith('2025/')).map((l) => l.slice(5)))
const arms = [
  ['qwen05', 'Qwen2.5-Coder-0.5B, pretrained, no js13k training'],
  ['scratch', '14M byte-level transformer, trained from scratch on js13k 2020-24 only'],
  ['qwen05ft', 'Qwen2.5-Coder-0.5B, fine-tuned on js13k 2020-24'],
  ['qwen05-32k', 'Qwen2.5-Coder-0.5B, pretrained, full 32K window'],
  ['qwen05ft-32k', 'Qwen2.5-Coder-0.5B, fine-tuned, full 32K window'],
].filter(([k]) => fs.existsSync(`${root}llm/out/${k}.json`))
const llm = Object.fromEntries(arms.map(([k]) => [k, JSON.parse(fs.readFileSync(`${root}llm/out/${k}.json`, 'utf8'))]))

// only games every arm scored
const slugs = Object.keys(split).filter((s) => arms.every(([k]) => llm[k][s] && 'bytes' in llm[k][s]))
const groups = { all: slugs, 'readable (not Roadroller)': slugs.filter((s) => !rr.has(s)), 'Roadroller-packed': slugs.filter((s) => rr.has(s)) }
const textRows = [
  ['brotli -q11', (s) => split[s].text_brotli],
  ['brotli + trained dict', (s) => split[s].text_brotliDict],
  ['context mixing, untrained', (s) => split[s].text_m0],
  ['context mixing, trained on 2020-24', (s) => split[s].text_m5],
  ...arms.map(([k, name]) => [name, (s) => llm[k][s].bytes]),
]
const sum = (f, ss) => ss.reduce((a, s) => a + f(s), 0)
let md = `<!-- llm:start -->\n## LLMs as the predictor\n\nThe same arithmetic-coding idea, with a neural language model as the predictor. LLM tokenizers only handle text, so each 2025 archive is split into its **text part** (UTF-8 code and markup, ${(sum((s) => split[s].textBytes, slugs) / 1e6).toFixed(2)} MB over ${slugs.length} games) and its **binary part** (images, audio: ${(sum((s) => split[s].binBytes, slugs) / 1e6).toFixed(2)} MB). The tables below compare every method on exactly the same text. LLM sizes are the sum of −log2 p over tokens, which is the arithmetic coder's output size, assuming both ends run bit-identical inference. The first three LLM rows score long texts in 2,048-token windows; the 32K rows use Qwen's full 32,768-token context window, which holds all but one test game whole.\n`
for (const [g, ss] of Object.entries(groups)) {
  if (!ss.length) continue
  md += `\n### Text part only, ${g}: ${ss.length} games, ${(sum((s) => split[s].textBytes, ss) / 1024).toFixed(0)} KiB\n\n| predictor | total KiB | bits per byte | vs trained context mixing |\n|---|---:|---:|---:|\n`
  const base = sum(textRows[3][1], ss), raw = sum((s) => split[s].textBytes, ss)
  for (const [name, f] of textRows) { const t = sum(f, ss); md += `| ${name} | ${(t / 1024).toFixed(1)} | ${((t * 8) / raw).toFixed(3)} | ${(((t - base) / base) * 100).toFixed(1)}% |\n` }
}
// full archive: LLM on text + trained context mixing on binary, vs the main table
md += `\n### Whole archive (LLM for text, trained context mixing for binary), all ${slugs.length} games\n\n| method | total KiB | vs best legal zip |\n|---|---:|---:|\n`
const zb = sum((s) => main[s].zBest, slugs)
const full = [['best legal zip per game', (s) => main[s].zBest], ['brotli + trained dict', (s) => main[s].brotliDict], ['context mixing, trained (main result)', (s) => main[s].m5],
  ...arms.map(([k, name]) => [name + ' + CM for binary', (s) => llm[k][s].bytes + split[s].bin_m5])]
for (const [name, f] of full) { const t = sum(f, slugs); md += `| ${name} | ${(t / 1024).toFixed(1)} | ${(((t - zb) / zb) * 100).toFixed(1)}% |\n` }
const fits = (f) => slugs.filter((s) => f(s) <= 13312).length
md += `\n<!-- llm:end -->`

const readme = fs.readFileSync(root + 'STUDY.md', 'utf8')
const out = readme.includes('<!-- llm:start -->')
  ? readme.replace(/<!-- llm:start -->[\s\S]*<!-- llm:end -->/, md)
  : readme.replace('## Results\n', md + '\n\n## Results\n')
fs.writeFileSync(root + 'STUDY.md', out)
console.log(md.replace(/<!--.*?-->/g, ''))
