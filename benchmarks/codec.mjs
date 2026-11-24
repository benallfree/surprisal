// Shared codec for archive.13k: an adaptive context-mixing model + binary arithmetic coder.
//
// Both ends run this same code, so the model (and anything it was pretrained on) is free;
// only the coded bytes count toward the 13,312-byte limit.
//
// Model: order-0 plus hashed order-1..12 contexts (each with a fast and a slow probability table), a "current word" context, and a match model
// (finds the last occurrence of the current 6+ byte context and predicts the byte that followed).
// A logistic mixer combines them; every table keeps learning while coding, on both ends.
// pretrain(bytes) runs the model over data without coding, so a codec can be shipped already
// "warmed up" on a corpus; snapshot()/restore() reset to that state between archives.

const ST = new Float64Array(4096) // stretch(p) = ln(p/(1-p)) for 12-bit p
for (let i = 0; i < 4096; i++) { const p = (i + 0.5) / 4096; ST[i] = Math.log(p / (1 - p)) }

export class Model {
  constructor({ B = 22, orders = [1, 2, 3, 4, 5, 6, 8, 12], cap = 3, cap2 = 60, lr = 0.004, hist = 96 << 20 } = {}) {
    this.B = B; this.orders = orders; this.cap = cap; this.cap2 = cap2; this.lr = lr
    this.N = orders.length + 2 // + order-0 + word
    this.NI = 2 * this.N + 2 // mixer inputs: fast + slow table per context, match, bias
    this.P = new Uint16Array(this.N << B).fill(32768)
    this.P2 = new Uint16Array(this.N << B).fill(32768) // slow-adapting twin of P
    this.C = new Uint8Array(this.N << B)
    this.W = new Float64Array(this.NI * 256 * 3).fill(0.12) // weight set by (partial byte, match state)
    this.MP = new Uint16Array(64 * 2).fill(32768); this.MC = new Uint8Array(64 * 2)
    this.hist = new Uint8Array(hist); this.pos = 0
    this.MT = new Int32Array(1 << 22) // hash of last 6 bytes -> position after them
    this.mptr = 0; this.mlen = 0; this.word = 0
    this.H = new Int32Array(this.N)
    this.s = new Float64Array(this.NI); this.idx = new Int32Array(this.N)
  }

  snapshot() {
    return { P: this.P.slice(), P2: this.P2.slice(), C: this.C.slice(), W: this.W.slice(), MP: this.MP.slice(), MC: this.MC.slice(), MT: this.MT.slice(), pos: this.pos, mptr: this.mptr, mlen: this.mlen, word: this.word }
  }
  restore(s) {
    this.P.set(s.P); this.P2.set(s.P2); this.C.set(s.C); this.W.set(s.W); this.MP.set(s.MP); this.MC.set(s.MC); this.MT.set(s.MT)
    this.pos = s.pos; this.mptr = s.mptr; this.mlen = s.mlen; this.word = s.word
  }

  // Byte-boundary bookkeeping: context hashes, word hash, match model.
  _ctx() {
    const h = this.hist, p = this.pos, H = this.H
    H[0] = 0
    let t = 0, k = 1, oi = 0
    for (const o of this.orders) {
      for (; k <= o; k++) t = Math.imul(t ^ (p - k >= 0 ? h[p - k] : 0), 0x2f0b4ac7) + k
      H[++oi] = t + o * 0x1000193
    }
    H[this.N - 1] = this.word * 0x3b9aca07 + 77
    // match model
    if (this.mlen > 0 && this.mptr < p) { /* continues, verified in _byte */ }
    if (p >= 6) {
      let mh = 0
      for (let i = 1; i <= 6; i++) mh = Math.imul(mh ^ h[p - i], 0x9e3779b1)
      mh >>>= 10
      if (this.mlen === 0) {
        const cand = this.MT[mh]
        if (cand > 0) {
          let L = 0
          while (L < 32 && cand - 1 - L >= 0 && h[cand - 1 - L] === h[p - 1 - L]) L++
          if (L >= 6) (this.mptr = cand), (this.mlen = L)
        }
      }
      this.MT[mh] = p
    }
  }

  _byte(c) {
    const h = this.hist
    if (this.mlen > 0) { if (h[this.mptr] === c) this.mlen++, this.mptr++; else this.mlen = 0 }
    if (this.mlen > 65535) this.mlen = 65535
    h[this.pos++] = c
    const w = (c >= 48 && c <= 57) || (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c === 95 || c === 36
    this.word = w ? Math.imul(this.word + c + 1, 0x6f4f2a35) : 0
  }

  // Run the model over n bytes. f(p12) gets P(bit=1) in 1..4095 and returns the actual bit.
  process(n, f) {
    const { B, N, NI, P, P2, C, W, MP, MC, s, idx, H, cap, cap2, lr } = this
    const out = new Uint8Array(n)
    for (let i = 0; i < n; i++) {
      this._ctx()
      const exp = this.mlen > 0 ? this.hist[this.mptr] : -1
      let c0 = 1
      for (let bit = 7; bit >= 0; bit--) {
        for (let m = 0; m < N; m++) {
          const j = (Math.imul(H[m] + c0 * 0x9e37, 0x2c9277b5) >>> (32 - B)) + (m << B)
          idx[m] = j
          s[m] = ST[P[j] >> 4]
          s[N + m] = ST[P2[j] >> 4]
        }
        // match input: only while the expected byte agrees with the bits seen so far
        let mi = -1
        if (exp >= 0 && (exp + 256) >> (bit + 1) === c0) {
          const eb = (exp >> bit) & 1, lb = Math.min(this.mlen, 31)
          mi = lb * 2 + eb
          s[2 * N] = ST[MP[mi] >> 4]
        } else s[2 * N] = 0
        s[2 * N + 1] = 0.25
        const ms = mi < 0 ? 0 : this.mlen < 16 ? 1 : 2
        const wo = (c0 * 3 + ms) * NI
        let dot = 0
        for (let m = 0; m < NI; m++) dot += W[wo + m] * s[m]
        let p = 1 / (1 + Math.exp(-dot))
        let p12 = Math.round(p * 4096)
        p12 = p12 < 1 ? 1 : p12 > 4095 ? 4095 : p12
        const y = f(p12)
        const err = (y - p12 / 4096) * lr
        for (let m = 0; m < NI; m++) W[wo + m] += err * s[m]
        for (let m = 0; m < N; m++) {
          const j = idx[m]
          const n = C[j], t = y ? 65535 : 0
          P[j] += (t - P[j]) / ((n < cap ? n : cap) + 1.5) | 0
          P2[j] += (t - P2[j]) / (n + 1.5) | 0
          if (n < cap2) C[j]++
        }
        if (mi >= 0) { MP[mi] += ((y ? 65535 : 0) - MP[mi]) / (MC[mi] + 1.5) | 0; if (MC[mi] < 255) MC[mi]++ }
        c0 = c0 * 2 + y
      }
      out[i] = c0 & 255
      this._byte(c0 & 255)
    }
    return out
  }

  // Learn from bytes without coding them.
  pretrain(bytes) {
    let b = 0
    this.process(bytes.length, () => (bytes[b >> 3] >> (7 - (b++ & 7))) & 1)
  }
}

// Ideal coded size in bits (arithmetic coding lands within a couple of bytes of this).
export function cost(model, bytes) {
  let b = 0, bits = 0
  model.process(bytes.length, (p) => {
    const y = (bytes[b >> 3] >> (7 - (b++ & 7))) & 1
    bits -= Math.log2(y ? p / 4096 : 1 - p / 4096)
    return y
  })
  return bits
}

// Real arithmetic coding: archive.13k = 4-byte length + code bytes.
export function encode(model, bytes) {
  const out = [bytes.length >>> 24, (bytes.length >> 16) & 255, (bytes.length >> 8) & 255, bytes.length & 255]
  let L = 0, U = 0xffffffff, b = 0
  model.process(bytes.length, (p) => {
    const y = (bytes[b >> 3] >> (7 - (b++ & 7))) & 1
    const m = L + ((U - L) >>> 12) * p
    y ? (U = m) : (L = m + 1)
    while (!((L ^ U) >>> 24)) out.push(U >>> 24), (L = (L << 8) >>> 0), (U = ((U << 8) | 255) >>> 0)
    return y
  })
  out.push(L & 0xffffff ? (L >>> 24) + 1 : L >>> 24)
  return Uint8Array.from(out)
}

export function decode(model, code) {
  const n = ((code[0] << 24) | (code[1] << 16) | (code[2] << 8) | code[3]) >>> 0
  let L = 0, U = 0xffffffff, x = 0, q = 4
  const g = () => code[q++] | 0
  for (let i = 0; i < 4; i++) x = ((x << 8) | g()) >>> 0
  return model.process(n, (p) => {
    const m = L + ((U - L) >>> 12) * p, y = x <= m
    y ? (U = m) : (L = m + 1)
    while (!((L ^ U) >>> 24)) (L = (L << 8) >>> 0), (U = ((U << 8) | 255) >>> 0), (x = ((x << 8) | g()) >>> 0)
    return +y
  })
}
