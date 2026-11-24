// In-memory zip reader + archive payload format, shared by bench.mjs and experiments.
import zlib from 'node:zlib'

// --- read a zip in memory (stored + deflate only); nothing is extracted to disk ---
export function unzip(buf) {
  let e = buf.length - 22
  while (e >= 0 && buf.readUInt32LE(e) !== 0x06054b50) e--
  if (e < 0) throw new Error('no EOCD')
  const count = buf.readUInt16LE(e + 10)
  let p = buf.readUInt32LE(e + 16)
  const files = []
  for (let i = 0; i < count; i++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error('bad central dir')
    const method = buf.readUInt16LE(p + 10), csize = buf.readUInt32LE(p + 20)
    const nl = buf.readUInt16LE(p + 28), xl = buf.readUInt16LE(p + 30), cl = buf.readUInt16LE(p + 32)
    const lo = buf.readUInt32LE(p + 42), name = buf.toString('utf8', p + 46, p + 46 + nl)
    p += 46 + nl + xl + cl
    if (name.endsWith('/') || name.startsWith('__MACOSX') || name.endsWith('.DS_Store')) continue
    const ds = lo + 30 + buf.readUInt16LE(lo + 26) + buf.readUInt16LE(lo + 28)
    const raw = buf.subarray(ds, ds + csize)
    if (method === 0) files.push({ name, data: raw })
    else if (method === 8) files.push({ name, data: zlib.inflateRawSync(raw) })
    else throw new Error('method ' + method)
  }
  return files.sort((a, b) => (a.name < b.name ? -1 : 1))
}

// archive payload: "name\nlength\n" + bytes, per file
export const blobOf = (files) => Buffer.concat(files.flatMap((f) => [Buffer.from(`${f.name}\n${f.data.length}\n`), f.data]))

// Fallback for zip methods we don't decode (LZMA 14, Deflate64 9, PPMd 98, ...): let 7-Zip extract
// into a scratch dir, read the files back, delete the dir.
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
export function unzip7z(zipPath) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'j13k-'))
  try {
    execFileSync('7z', ['x', '-y', '-bd', '-o' + tmp, zipPath], { stdio: 'ignore' })
    const files = []
    const walk = (d, rel) => {
      for (const e of fs.readdirSync(d, { withFileTypes: true })) {
        const r = rel ? rel + '/' + e.name : e.name
        if (e.isDirectory()) walk(path.join(d, e.name), r)
        else if (!r.startsWith('__MACOSX') && !r.endsWith('.DS_Store')) files.push({ name: r, data: fs.readFileSync(path.join(d, e.name)) })
      }
    }
    walk(tmp, '')
    return files.sort((a, b) => (a.name < b.name ? -1 : 1))
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
}

// Write files under dir (names sanitized), for re-zipping with other tools.
export function writeFiles(dir, files) {
  fs.rmSync(dir, { recursive: true, force: true })
  for (const f of files) {
    const safe = f.name.split('/').filter((s) => s && s !== '..' && s !== '.').join('/')
    if (!safe) continue
    fs.mkdirSync(path.dirname(path.join(dir, safe)), { recursive: true })
    fs.writeFileSync(path.join(dir, safe), f.data)
  }
}
