# Reference implementations (Python)

These define spec v1 ([../../SPEC.md](../../SPEC.md)) and produced its test vectors ([../../spec/vectors/](../../spec/vectors/)).
The Go implementation in this repository must give the same tables and the same archives, byte for byte; the vector
tests check that.

| file | what it is |
|---|---|
| [intref.py](intref.py) | **the spec in code**: the integer forward pass in numpy. Slow (a few tokens per second for the 0.5B model) and authoritative. Also builds the frozen tables (`spec/tables-v1.gguf`) and converts models. |
| [intgpu.py](intgpu.py) | the same forward pass in PyTorch (any device), and with `--fast` a Triton/CUDA path with CUDA graphs and int8 tensor-core attention: the fast reference the Go CUDA kernels were ported from. Runs on Modal. |
| [bundle.py](bundle.py) | reads and writes model bundles (GGUF v3, SPEC.md §1.3). |
| [archive.py](archive.py) | the `.13k` archive header and segments (SPEC.md §3). |
| [buildv1.py](buildv1.py) | builds the v1 bundles on Modal from the official Qwen weights and reports their SHA-256. |
| [logs/](logs/) | the reference runs on H100 and H200 the speed comparison in the README cites. |

They need Python 3.12 and numpy (`intref.py`), plus PyTorch and Triton (`intgpu.py`) and [Modal](https://modal.com)
for the cloud runs. Run them from the repository root; each script's docstring has its commands. They find `spec/`
beside themselves (as the Modal images lay it out) or at the repository root.
