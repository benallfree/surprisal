// Fetch the file tree (paths + sizes) of every entry's source repo in the js13kGames GitHub org.
//   node corpus/trees.mjs     -> corpus/trees.json   (uses `gh api`; resumable)
import fs from 'node:fs'
import { execFileSync } from 'node:child_process'

const dir = new URL('.', import.meta.url).pathname
const out = dir + 'trees.json'
const trees = fs.existsSync(out) ? JSON.parse(fs.readFileSync(out, 'utf8')) : {}
const entries = fs.readFileSync(dir + 'slugs.txt', 'utf8').trim().split('\n').flatMap((l) => { const [y, ...s] = l.split(' '); return s.map((slug) => ({ y, slug })) })
let n = 0
for (const { y, slug } of entries) {
  const key = `${y}/${slug}`
  if (trees[key]) continue
  try {
    const json = execFileSync('gh', ['api', `repos/js13kGames/${slug}/git/trees/HEAD?recursive=1`], { timeout: 30000, maxBuffer: 1 << 26 })
    const t = JSON.parse(json)
    trees[key] = { truncated: !!t.truncated, files: t.tree.filter((e) => e.type === 'blob').map((e) => [e.path, e.size]) }
  } catch (e) {
    trees[key] = { error: String(e.message).slice(0, 200) }
  }
  if (++n % 50 === 0) { fs.writeFileSync(out, JSON.stringify(trees)); console.log(new Date().toTimeString().slice(0, 8), `fetched ${n}`) }
}
fs.writeFileSync(out, JSON.stringify(trees))
console.log(new Date().toTimeString().slice(0, 8), `done: ${Object.keys(trees).length} trees`)
