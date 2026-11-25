"""The archive container, the levels and the long-input rule.

    byte 0   kind            0xA1 a stream, 0xA2 a file set
    byte 1   version, level  header version 1 (high nibble) and compression level 0-9 (low nibble); the level's
                             model is spec/models-v1.json "levels"[level]; level 0 stores the bytes uncoded
    varint   token count n   unsigned LEB128 (levels 1-9)
    rest     the arithmetic-coded body (Witten-Neal-Cleary, 32-bit, frequency total 2^28)

A level names a model ("levels") and an attention window ("windows", 0 = full attention; with a window w, each
position attends to the last w positions of its segment and the first "sink"). The reference codes streams.

Long inputs: the tokens are coded in segments of at most SEG = 32,768 tokens. Each segment starts a fresh context
(the start token at position 0, an empty KV cache); the arithmetic coder runs on, unbroken, across segments.
"""
import json
from pathlib import Path

def _spec_dir():
    # spec/ beside this file (as Modal images lay it out), else the repository's top-level spec/
    here = Path(__file__).resolve().parent
    for d in (here / "spec", here.parent.parent / "spec"):
        if d.is_dir():
            return d
    return here / "spec"


SPEC = _spec_dir()

KIND_STREAM = 0xA1
KIND_FILES = 0xA2
VERSION = 1
SEG = 32768
MANIFEST = SPEC / "models-v1.json"


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def pack(level, n, body):
    return bytes([KIND_STREAM, VERSION << 4 | level]) + varint(n) + body


def first_level(model_id):
    return manifest()["levels"].index(model_id)


def model_of(level):
    return manifest()["levels"][level]


def window(level):
    """-> (window, sink) for a level; (0, 0) is full attention."""
    m = manifest()
    w = m["windows"][level]
    return (w, m["sink"]) if w else (0, 0)


def unpack(data):
    """-> (level, token count, body)."""
    if data[0] not in (KIND_STREAM, KIND_FILES):
        raise ValueError(f"not a .13k archive (first byte {data[0]:#x})")
    version, level = data[1] >> 4, data[1] & 15
    if version != VERSION:
        raise ValueError(f"header version {version}; this reference reads version {VERSION}")
    if not 1 <= level <= 9:
        raise ValueError(f"level {level}: only levels 1-9 are model-coded")
    n, shift, i = 0, 0, 2
    while True:
        b = data[i]
        n |= (b & 0x7F) << shift
        shift += 7
        i += 1
        if not b & 0x80:
            break
    return level, n, data[i:]


def segments(n):
    """[(start, end)) token ranges, each a fresh context."""
    return [(a, min(a + SEG, n)) for a in range(0, n, SEG)] or [(0, 0)]


def manifest():
    return json.loads(MANIFEST.read_text())


def model_id_for(bundle_path):
    """The manifest id of a bundle, by file name."""
    name = Path(bundle_path).name
    for m in manifest()["models"]:
        if m["bundle"]["file"] == name:
            return m["id"]
    raise KeyError(f"{name} is not in {MANIFEST.name}")
