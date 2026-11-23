"""Build v1 model bundles on Modal from the official Qwen weights (cached in Volume compress13k-hf) and report each
bundle's SHA-256 and the source revision, so they can be checked against builds made elsewhere.

    .venv/bin/modal run llm/buildv1.py --sizes 0.5B,1.5B,7B
"""
import json, os, sys, time
from pathlib import Path
import modal

here = Path(__file__).parent
image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy")
         .add_local_file(str(here / "intref.py"), "/root/intref.py")
         .add_local_file(str(here / "bundle.py"), "/root/bundle.py")
         .add_local_file(str(here / "spec" / "tables-v1.i13k"), "/root/spec/tables-v1.i13k"))
app = modal.App("compress13k-buildv1", image=image)
hf = modal.Volume.from_name("compress13k-hf")
iv = modal.Volume.from_name("compress13k-int")


@app.function(volumes={"/cache": hf, "/int": iv}, memory=98304, cpu=8, timeout=3600)
def build(size):
    sys.path.insert(0, "/root")
    import intref, bundle
    repo = f"Qwen/Qwen2.5-Coder-{size}"
    d = Path("/cache/hub") / ("models--" + repo.replace("/", "--"))
    rev = (d / "refs" / "main").read_text().strip()
    src = d / "snapshots" / rev
    dst = f"/int/coder{size.lower()}-v1.i13k"
    t0 = time.time()
    intref.build(str(src), dst)
    iv.commit()
    return dict(size=size, repo=repo, revision=rev, sha256=bundle.sha256(dst), bytes=os.path.getsize(dst),
                seconds=round(time.time() - t0), files=sorted(p.name for p in src.iterdir()))


@app.local_entrypoint()
def main(sizes: str = "0.5B"):
    for r in build.map(sizes.split(",")):
        print(json.dumps(r), flush=True)
