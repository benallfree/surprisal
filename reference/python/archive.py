"""v1 archive container and the long-input rule.

    byte 0   0xA1            magic (high nibble 0xA) and format version 1 (low nibble)
    byte 1   model id        an entry in spec/models-v1.json (0 is reserved)
    varint   token count n   unsigned LEB128
    rest     the arithmetic-coded body (Witten-Neal-Cleary, 32-bit, frequency total 2^28)

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

MAGIC_V1 = 0xA1
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


def pack(model_id, n, body):
    assert 1 <= model_id <= 255
    return bytes([MAGIC_V1, model_id]) + varint(n) + body


def unpack(data):
    """-> (model id, token count, body)."""
    if data[0] != MAGIC_V1:
        raise ValueError(f"not a v1 archive (first byte {data[0]:#x})")
    n, shift, i = 0, 0, 2
    while True:
        b = data[i]
        n |= (b & 0x7F) << shift
        shift += 7
        i += 1
        if not b & 0x80:
            break
    return data[1], n, data[i:]


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
