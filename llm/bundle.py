"""v1 model bundle file (.i13k): one self-describing binary file, easy to read from any language.

    magic    8 bytes  b"I13KMDL1"
    u32      length of the config JSON, then the JSON (UTF-8, sorted keys, no spaces)
    u32      number of tensors, then for each tensor, in file order:
               u16 name length, name (ASCII)
               u8  dtype: 1 = int8, 2 = int16, 3 = int32, 4 = int64
               u8  ndim, then ndim x u32 dims (row-major)
               data, little-endian, row-major

All integers little-endian. The file's SHA-256 identifies the model (archives name it through the manifest).
"""
import hashlib, json, struct
from pathlib import Path
import numpy as np

MAGIC = b"I13KMDL1"
DTYPES = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int64}
CODES = {np.dtype(v): k for k, v in DTYPES.items()}

# The config fields the spec uses; nothing else goes into the bundle, so the same weights give the same file.
CONFIG_KEYS = ("bos_token_id", "hidden_size", "intermediate_size", "max_position_embeddings", "num_attention_heads",
               "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "rope_theta", "tie_word_embeddings",
               "vocab_size")


def canonical_config(cfg):
    return json.dumps({k: cfg[k] for k in CONFIG_KEYS}, sort_keys=True, separators=(",", ":")).encode()


def write(path, cfg_bytes, tensors):
    """tensors: list of (name, int array) in file order."""
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(cfg_bytes)) + cfg_bytes + struct.pack("<I", len(tensors)))
        for name, a in tensors:
            a = np.ascontiguousarray(a) if np.ndim(a) else np.asarray(a)  # ascontiguousarray makes 0-d arrays 1-d
            nb = name.encode("ascii")
            f.write(struct.pack("<H", len(nb)) + nb + struct.pack("<BB", CODES[a.dtype], a.ndim))
            f.write(struct.pack(f"<{a.ndim}I", *a.shape))
            f.write(a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes())


def read(path):
    """-> (config dict, {name: array}) with arrays viewing the file's bytes."""
    b = Path(path).read_bytes()
    assert b[:8] == MAGIC, "not a v1 bundle"
    (n,) = struct.unpack_from("<I", b, 8)
    cfg = json.loads(b[12 : 12 + n])
    o = 12 + n
    (nt,) = struct.unpack_from("<I", b, o)
    o += 4
    out = {}
    for _ in range(nt):
        (ln,) = struct.unpack_from("<H", b, o)
        name = b[o + 2 : o + 2 + ln].decode("ascii")
        o += 2 + ln
        code, nd = struct.unpack_from("<BB", b, o)
        o += 2
        dims = struct.unpack_from(f"<{nd}I", b, o)
        o += 4 * nd
        dt = np.dtype(DTYPES[code]).newbyteorder("<")
        cnt = int(np.prod(dims)) if nd else 1
        out[name] = np.frombuffer(b, dtype=dt, count=cnt, offset=o).reshape(dims)
        o += cnt * dt.itemsize
    assert o == len(b), "trailing bytes"
    return cfg, out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class Bundle(dict):
    """Dict-like view used by the model loaders: z[name] arrays, z['cfg'] as bytes of the config JSON."""

    def __init__(self, path):
        cfg, t = read(path)
        super().__init__(t)
        self["cfg"] = np.frombuffer(json.dumps(cfg).encode(), dtype=np.uint8)


def load(path):
    """Open a v1 bundle (.i13k), or a legacy .npz bundle."""
    return Bundle(path) if str(path).endswith(".i13k") else np.load(path)
