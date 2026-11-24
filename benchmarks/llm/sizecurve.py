"""How compressed size grows with input size: prefixes of one file under each method.

    .venv/bin/python llm/sizecurve.py <file>

Prints, for prefixes of 2, 4, 8, ... KB, the bytes used by Qwen (pretrained, 2,048-token window),
the untrained context mixer, brotli -q11, xz -9e and zip PPMd. A widening or shrinking ratio between
methods as the prefix grows is the inflection point in question.
"""
import json, math, subprocess, sys, tempfile, os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1])
data = path.read_bytes()
sys.argv = ["x", "mlx-community/Qwen2.5-Coder-0.5B-bf16", "probe"]
__file__ = str(root / "llm" / "score.py")
exec((root / "llm" / "score.py").read_text().split("slugs = json.loads")[0])

sizes = [s for s in (2048, 4096, 8192, 16384, 32768, 65536, 131072) if s < len(data)] + [len(data)]
tmp = Path(tempfile.mkdtemp())
print(f"{'prefix':>8} {'qwen':>7} {'cm':>7} {'brotli':>7} {'xz':>7} {'ppmd':>7}   qwen/brotli  qwen/cm  qwen bits/byte")
for n in sizes:
    # cut on a UTF-8 boundary
    while n < len(data) and (data[n] & 0xC0) == 0x80:
        n += 1
    chunk = data[:n]
    f = tmp / "p.bin"
    f.write_bytes(chunk)
    q = math.ceil(bits_of(chunk.decode("utf-8"))[0] / 8) + 6
    cm = int(subprocess.check_output(["node", "-e", f"""
      import('{root}/codec.mjs').then(({{Model,cost}})=>{{const b=require('fs').readFileSync('{f}');console.log(Math.ceil(cost(new Model({{B:20}}),b)/8)+6)}})"""]).decode())
    br = len(subprocess.check_output(["brotli", "-c", "-q", "11", "-w", "24", str(f)]))
    xz = len(subprocess.check_output(["xz", "-c", "--format=raw", "--lzma2=preset=9e", str(f)]))
    best = None
    for o in (4, 6, 8, 12, 16):
        z = tmp / f"p{o}.zip"
        z.unlink(missing_ok=True)
        subprocess.run(["7z", "a", "-tzip", "-bd", "-mx=9", "-mm=PPMd", f"-mo={o}", "-mmem=256m", str(z), str(f)], stdout=subprocess.DEVNULL, check=True)
        best = min(best or 1 << 30, z.stat().st_size)
    print(f"{n:>8} {q:>7} {cm:>7} {br:>7} {xz:>7} {best:>7}   {q / br:>10.3f} {q / cm:>8.3f} {q * 8 / n:>10.3f}", flush=True)
