"""The converted model file: a GGUF (v3), written deterministically, so its SHA-256 identifies the model.

Contents (see SPEC.md §1.3):
- metadata, in this order: general.architecture = "qwen2-intcodec", general.alignment = 32, intcodec.spec_version = 1,
  the config (intcodec.<key> for the keys in CONFIG_KEYS), the matrix exponents (intcodec.exp.<matrix>, int32), then
  the tokenizer (tokenizer.ggml.model = "gpt2", .pre = "qwen2", .tokens, .token_type, .merges);
- tensors as GGUF integer types (I8 = 24, I16 = 25, I32 = 26, I64 = 27), dims in GGUF order (dims[0] contiguous),
  each tensor's data aligned to 32 bytes, in the spec's fixed order.

Nothing in the file depends on where the weights came from, so the official Qwen repo and a faithful copy give the
same bytes.
"""
import hashlib, json, struct
from pathlib import Path
import numpy as np

ALIGN = 32
GTYPE = {np.dtype(np.int8): 24, np.dtype(np.int16): 25, np.dtype(np.int32): 26, np.dtype(np.int64): 27}
NPTYPE = {v: k for k, v in GTYPE.items()}
CONFIG_KEYS = ("bos_token_id", "hidden_size", "intermediate_size", "max_position_embeddings", "num_attention_heads",
               "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "rope_theta", "tie_word_embeddings",
               "vocab_size")
U32, I32, F64, BOOL, STR, ARR = 4, 5, 12, 7, 8, 9


def _s(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key, typ, val):
    out = _s(key) + struct.pack("<I", typ)
    if typ == U32:
        return out + struct.pack("<I", val)
    if typ == I32:
        return out + struct.pack("<i", val)
    if typ == F64:
        return out + struct.pack("<d", val)
    if typ == BOOL:
        return out + struct.pack("<B", 1 if val else 0)
    if typ == STR:
        return out + _s(val)
    et, items = val  # array
    out += struct.pack("<IQ", et, len(items))
    if et == STR:
        return out + b"".join(_s(x) for x in items)
    if et == I32:
        return out + np.asarray(items, dtype="<i4").tobytes()
    raise ValueError(et)


def config_kvs(cfg):
    kv = []
    for k in CONFIG_KEYS:
        v = cfg[k]
        if isinstance(v, bool):
            kv.append(("intcodec." + k, BOOL, v))
        elif isinstance(v, float):
            kv.append(("intcodec." + k, F64, v))
        else:
            kv.append(("intcodec." + k, U32, int(v)))
    return kv


def tokenizer_kvs(tokenizer_json, vocab_size):
    """GGUF tokenizer arrays from a Hugging Face tokenizer.json, padded to the model's vocabulary size."""
    t = json.loads(Path(tokenizer_json).read_text())
    tokens = [None] * vocab_size
    types = [1] * vocab_size  # 1 normal
    for s, i in t["model"]["vocab"].items():
        tokens[i] = s
    for a in t["added_tokens"]:
        tokens[a["id"]] = a["content"]
        types[a["id"]] = 3 if a["special"] else 4  # 3 control, 4 user-defined
    for i in range(vocab_size):
        if tokens[i] is None:
            tokens[i], types[i] = f"[PAD{i}]", 5  # 5 unused
    merges = [m if isinstance(m, str) else " ".join(m) for m in t["model"]["merges"]]
    return [("tokenizer.ggml.model", STR, "gpt2"), ("tokenizer.ggml.pre", STR, "qwen2"),
            ("tokenizer.ggml.tokens", ARR, (STR, tokens)), ("tokenizer.ggml.token_type", ARR, (I32, types)),
            ("tokenizer.ggml.merges", ARR, (STR, merges))]


def write(path, kvs, tensors, arch="qwen2-intcodec"):
    """kvs: [(key, type, value)] after the fixed general.* keys; tensors: [(name, int array)] in file order."""
    kvs = [("general.architecture", STR, arch), ("general.alignment", U32, ALIGN)] + list(kvs)
    head = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(kvs))] + [_kv(*kv) for kv in kvs]
    off, infos, datas = 0, [], []
    for name, a in tensors:
        a = np.asarray(a)
        a = np.ascontiguousarray(a) if a.ndim else a.reshape(1)
        dims = list(reversed(a.shape))
        infos.append(_s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
                     + struct.pack("<IQ", GTYPE[a.dtype], off))
        b = a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
        datas.append(b + b"\0" * (-len(b) % ALIGN))
        off += len(datas[-1])
    hb = b"".join(head + infos)
    with open(path, "wb") as f:
        f.write(hb + b"\0" * (-len(hb) % ALIGN))
        for d in datas:
            f.write(d)


class _R:
    def __init__(self, b):
        self.b, self.o = b, 0

    def take(self, fmt):
        v = struct.unpack_from(fmt, self.b, self.o)
        self.o += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    def s(self):
        n = self.take("<Q")
        v = self.b[self.o : self.o + n].decode("utf-8")
        self.o += n
        return v

    def val(self, t):
        fm = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}
        if t == 8:
            return self.s()
        if t == 9:
            et, n = self.take("<IQ")
            return [self.val(et) for _ in range(n)]
        v = self.take(fm[t])
        return bool(v) if t == 7 else v


def read(path, tensors_only=False):
    """-> (metadata dict, {name: array in row-major shape}). Arrays view the file's bytes."""
    b = Path(path).read_bytes()
    assert b[:4] == b"GGUF", "not a GGUF"
    r = _R(b)
    r.o = 4
    ver, nt, nkv = r.take("<IQQ")
    meta = {}
    for _ in range(nkv):
        k = r.s()
        t = r.take("<I")
        if tensors_only and k.startswith("tokenizer."):
            meta[k] = None
            r.val(t)  # parse past it
        else:
            meta[k] = r.val(t)
    infos = []
    for _ in range(nt):
        name = r.s()
        nd = r.take("<I")
        dims = [r.take("<Q") for _ in range(nd)]
        typ, off = r.take("<IQ")
        infos.append((name, dims, typ, off))
    align = meta.get("general.alignment", 32)
    base = (r.o + align - 1) // align * align
    out = {}
    for name, dims, typ, off in infos:
        dt = NPTYPE[typ].newbyteorder("<")
        cnt = int(np.prod(dims))
        out[name] = np.frombuffer(b, dtype=dt, count=cnt, offset=base + off).reshape(list(reversed(dims)))
    return meta, out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Bundle(dict):
    """Dict-like view used by the model loaders: z[name] arrays, z[name + '.E'] exponents, z['cfg'] config JSON."""

    def __init__(self, path):
        meta, t = read(path, tensors_only=True)
        super().__init__(t)
        cfg = {k: meta["intcodec." + k] for k in CONFIG_KEYS}
        for k, v in meta.items():
            if k.startswith("intcodec.exp."):
                self[k[len("intcodec.exp."):] + ".E"] = np.array(v, dtype=np.int32)
        self["cfg"] = np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)


def load(path):
    """Open a converted model (.gguf), or a legacy .npz from before spec v1."""
    return Bundle(path) if str(path).endswith(".gguf") else np.load(path)
