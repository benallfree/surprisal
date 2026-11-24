"""Run nollama / surprisal Go test binaries on a Modal GPU (cross-compiled on the Mac; no Go needed remotely).

    GORUN_STAGE=<dir> modal run scripts/gorun.py --gpu H100 --cmd "cd engine && ./engine.test -test.run TestIntTables -test.v"

<dir> (staged by scripts/gostage.sh, which wipes it: write run.sh after) holds engine/engine.test, engine/testdata/..., surprisal/surprisal.test, spec/vectors/... (see scripts/gostage.sh).
The converted models are in the Volume compress13k-int at /int (NOLLAMA_INTMODEL_DIR and SURPRISAL_HOME point there).
"""
import os
import subprocess
import sys
import time

import modal

app = modal.App("compress13k-go")
intvol = modal.Volume.from_name("compress13k-int", create_if_missing=True)
STAGE = os.environ.get("GORUN_STAGE", "")
image = modal.Image.debian_slim(python_version="3.12")
if STAGE:
    image = image.add_local_dir(STAGE, "/work")


def body(cmd):
    env = dict(os.environ, NOLLAMA_INTMODEL_DIR="/int", SURPRISAL_HOME="/int")
    print(f"{time.strftime('%T')} $ {cmd}", flush=True)
    subprocess.run(["bash", "-c", "nvidia-smi -L; chmod +x /work/*/*.test; " + cmd], cwd="/work", env=env, check=False,
                   stdout=sys.stdout, stderr=subprocess.STDOUT)
    print(f"{time.strftime('%T')} done", flush=True)


kw = dict(image=image, volumes={"/int": intvol}, timeout=3600)
# the CUDA toolkit image, for Nsight Compute (ncu)
ncu_image = modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
if STAGE:
    ncu_image = ncu_image.add_local_dir(STAGE, "/work")


@app.function(gpu="H100!", **kw)  # "!": no upgrade to an H200
def run_h100(cmd: str):
    body(cmd)


@app.function(gpu="H100!", image=ncu_image, volumes={"/int": intvol}, timeout=3600)
def run_h100_ncu(cmd: str):
    body(cmd)


@app.local_entrypoint()
def main(cmd: str, gpu: str = "H100"):
    {"H100": run_h100, "H100-ncu": run_h100_ncu}[gpu].remote(cmd)
