// Classify each entry's source repo (from trees.json) by how usable its hand-written source is.
//   node corpus/survey.mjs
// Exported `sourceFiles(tree)` is the rule set later used to build source archives.
import fs from 'node:fs'

const SRC = /\.(js|mjs|cjs|ts|tsx|jsx|html?|css|scss|glsl|frag|vert|hx)$/i
const SKIP_DIR = /(^|\/)(node_modules|\.website|\.github|\.vscode|\.husky|\.idea|\.yarn|dist|build|builds|out|release|releases|bin|docs?|coverage|tmp|temp|vendors?|libs?|third[_-]?party|external|tools?|scripts|editor|level[_-]?editor|tests?|__tests__|submission|zip)\//i
const SKIP_FILE = /(\.min\.(js|css)|\.d\.ts|\.bundle\.js|\.map)$|(^|\/)((webpack|vite|rollup|esbuild|babel|eslint|prettier|jest|postcss|tailwind|svelte|terser)[^/]*\.(c|m)?js|\.eslintrc[^/]*|gulpfile\.js|gruntfile\.js|[^/]*roadroller[^/]*)$/i
const MAX_FILE = 100 * 1024

export const sourceFiles = (tree) =>
  (tree.files || []).filter(([p, s]) => SRC.test(p) && !SKIP_DIR.test(p) && !SKIP_FILE.test(p) && s <= MAX_FILE)

const dir = new URL('.', import.meta.url).pathname
if (import.meta.url === `file://${process.argv[1]}`) {
  const trees = JSON.parse(fs.readFileSync(dir + 'trees.json', 'utf8'))
  const rr = new Set(fs.existsSync(dir + 'roadroller.txt') ? fs.readFileSync(dir + 'roadroller.txt', 'utf8').split('\n') : [])
  const rows = {}
  for (const [key, t] of Object.entries(trees)) {
    const y = key.slice(0, 4)
    const r = (rows[y] ||= { entries: 0, error: 0, 'no source (<2 KB)': 0, 'huge (>250 KB)': 0, usable: 0, 'usable & roadroller zip': 0 })
    r.entries++
    if (t.error) { r.error++; continue }
    const bytes = sourceFiles(t).reduce((a, [, s]) => a + s, 0)
    if (bytes < 2048) r['no source (<2 KB)']++
    else if (bytes > 250 * 1024) r['huge (>250 KB)']++
    else { r.usable++; if (rr.has(key)) r['usable & roadroller zip']++ }
  }
  console.table(rows)
}
