// Inject the benchmark results (results.json + results.md from bench.mjs) into STUDY.md
// between <!-- results:start --> and <!-- results:end -->.
//   node report.mjs
import fs from 'node:fs'

const dir = new URL('.', import.meta.url).pathname
const games = JSON.parse(fs.readFileSync(dir + 'results.json', 'utf8'))
const summary = fs.readFileSync(dir + 'results.md', 'utf8').trim()

// Every method at its best settings. Sizes are bytes of the whole archive.
const cols = [
  ['zip', 'submitted zip'],
  ['zDeflate', 'zip Deflate'],
  ['zZopfli', 'zip Zopfli'],
  ['zDeflate64', 'zip Deflate64'],
  ['zBzip2', 'zip BZip2'],
  ['zLzma', 'zip LZMA'],
  ['zPpmd', 'zip PPMd'],
  ['xz', 'xz -9e'],
  ['brotli', 'brotli'],
  ['brotliDict', 'brotli +dict'],
  ['zstd', 'zstd'],
  ['zstdDict', 'zstd +dict'],
  ['m0', 'model'],
  ['m5', 'model +trained'],
]
const kb = (n) => (n / 1024).toFixed(2)
const rr = new Set(fs.existsSync(dir + 'corpus/roadroller.txt') ? fs.readFileSync(dir + 'corpus/roadroller.txt', 'utf8').split('\n') : [])
const others = cols.filter(([k]) => k !== 'm5')
const bestOther = (g) => Math.min(...others.map(([k]) => g[k]))
let t = `| # | game | unpacked KiB | ${cols.map((c) => c[1]).join(' | ')} | smallest | model +trained vs best other method |\n`
t += `|---:|---|---:|${cols.map(() => '---:').join('|')}|---|---:|\n`
games
  .slice()
  .sort((a, b) => a.m5 / bestOther(a) - b.m5 / bestOther(b))
  .forEach((g, i) => {
    const best = Math.min(...cols.map(([k]) => g[k]))
    const cells = cols.map(([k]) => (g[k] === best ? `**${kb(g[k])}**` : kb(g[k])))
    const d = ((g.m5 - bestOther(g)) / bestOther(g)) * 100
    const win = cols.find(([k]) => g[k] === best)[1]
    t += `| ${i + 1} | [${g.slug}](https://js13kgames.com/2025/games/${g.slug})${rr.has('2025/' + g.slug) ? ' RR' : ''} | ${kb(g.raw)} | ${cells.join(' | ')} | ${win} | ${d.toFixed(1)}% |\n`
  })

// how often each method wins
const wins = Object.fromEntries(cols.map(([k]) => [k, 0]))
for (const g of games) { const best = Math.min(...cols.map(([k]) => g[k])); for (const [k] of cols) if (g[k] === best) { wins[k]++; break } }
const winLine = cols.filter(([k]) => wins[k]).map(([k, n]) => `${n} ${wins[k]}`).join(', ')
const tot = (f) => games.reduce((a, g) => a + f(g), 0)
const vsAll = ((tot((g) => g.m5) - tot(bestOther)) / tot(bestOther)) * 100
const medAll = games.map((g) => g.m5 / bestOther(g)).sort((a, b) => a - b)[games.length >> 1]
const beats = games.filter((g) => g.m5 < bestOther(g)).length
console.log(`vs best other: total ${vsAll.toFixed(1)}%, median ratio ${medAll.toFixed(3)}, wins ${beats}/${games.length}`)

const block = `<!-- results:start -->
${summary}

Smallest archive per game (wins): ${winLine}.

Against the best of every other method on each game (an oracle that picks the best competitor per game after the fact), the trained model is ${vsAll.toFixed(1)}% in total, with a median per-game ratio of ${medAll.toFixed(3)}, and it is smaller on ${beats} of ${games.length} games.

### Every 2025 game, every method

All sizes in KiB (1,024 bytes) for the whole game. The bold cell in each row is the smallest. "zip …" columns are legal js13k zips
built with 7-Zip at max settings (PPMd: best of orders 4-16; Zopfli via \`advzip -z -4\`). "+dict" means a dictionary trained on
2020-2024 entries, best of a 1 MB \`zstd --train\` dictionary and the raw last 16 MB of training data. "model" is the codec in
[codec.mjs](codec.mjs) untrained; "model +trained" is the same codec after pretraining on all 2020-2024 entries.
"smallest" names the row's winner. The last column compares the trained model with the best of **every other method** on that game
(all zips, xz, brotli, zstd, both trained dictionaries, and the untrained model), so a positive number means some other method won that game.
RR marks games whose submission is Roadroller-packed. Rows are sorted by the last column.

${t}<!-- results:end -->`

const readme = fs.readFileSync(dir + 'STUDY.md', 'utf8')
const out = readme.includes('<!-- results:start -->')
  ? readme.replace(/<!-- results:start -->[\s\S]*<!-- results:end -->/, block)
  : readme.trimEnd() + '\n\n## Results\n\n' + block + '\n'
fs.writeFileSync(dir + 'STUDY.md', out)
console.log(`STUDY.md: wrote ${games.length} rows`)
