# GPU Systems: Kernels and Profiling

My GPU self-study from summer 2026: Stanford CS336 Assignment 2, six chapters of
*Programming Massively Parallel Processors* (PMPP), and GPU MODE lectures. It has
FlashAttention-2 in PyTorch and Triton, fused Triton kernels, CUDA matmul tiling, a DDP
harness, and scripts that run tests and profilers on Modal cloud GPUs.

## Results

Triton kernels against naive eager-mode PyTorch versions of the same ops, measured on one
NVIDIA A10G (24 GB) through Modal in August 2026. Timings move a little between runs.

| Kernel | Shape | PyTorch | This repo | Speedup |
| :--- | :--- | ---: | ---: | ---: |
| Fused softmax (`triton_softmax.py`) vs naive PyTorch softmax | B=4, S=512, V=32,768 | 1.85 ms | 0.49 ms | 3.77× |
| RMSNorm (`triton_rmsnorm.py`) vs naive PyTorch RMSNorm | D=4,096 | 0.42 ms | 0.18 ms | 2.33× |
| FlashAttention-2 vs standard attention | S=2,048 | 6.12 ms | 2.14 ms | 2.86× |

Naive vs 32×32 shared-memory tiled CUDA matmul (`cuda_src/tile.cu`), timed with CUDA
events in `notebooks/tile_vs_non_tile_mm_colab.py`:

| N × N | Naive | Tiled | Speedup |
| :--- | ---: | ---: | ---: |
| 2,048 | 14.2 ms | 4.1 ms | 3.46× |
| 8,192 | 842.0 ms | 231.5 ms | 3.64× |
| 16,384 | 6,820 ms | 1,810 ms | 3.76× |

## Where things are

| Path | What it is |
| :--- | :--- |
| `src/gpu_systems/kernels/flash_attention.py` | FlashAttention-2 forward in plain PyTorch: Q and K tiles, a running row max, online softmax rescaling. Checked against standard attention. |
| `src/gpu_systems/kernels/triton_flash_attn.py` | The same forward pass as a Triton kernel with block pointers (`tl.make_block_ptr`, `tl.advance`). |
| `src/gpu_systems/kernels/triton_softmax.py` | Single-pass fused softmax (the benchmarked one), plus a tiled online softmax written on the way to FlashAttention-2. |
| `src/gpu_systems/kernels/triton_rmsnorm.py` | RMSNorm in Triton. |
| `src/gpu_systems/kernels/cuda_extensions.py` | CUDA kernels compiled with `torch.utils.cpp_extension.load_inline`: RGB to grayscale and RGB box blur (PMPP exercises). |
| `cuda_src/tile.cu` | Naive and 32×32 shared-memory tiled matmul. |
| `cuda_src/drafts/` | Unfinished SwiGLU and matmul launchers. They don't compile yet. |
| `src/gpu_systems/benchmark.py`, `model_configs.py` | CS336 timing harness: forward, forward+backward and full training step for GPT-2-style configs from small to 10B. |
| `src/gpu_systems/mixed_precision.py` | Autocast toy model that prints which layers run in FP16 and which stay in FP32. |
| `src/gpu_systems/ddp_train.py` | DDP training harness (NCCL process group, `DistributedSampler`). |
| `modal/` | Run tests, scripts, Nsight Systems and the PyTorch Profiler on Modal GPUs (T4, A10G, A100). |
| `notebooks/` | Profiling notebooks, plus Colab exports of my first CUDA exercises (`pmpp_colab_exercises.py`) and of the matmul benchmark. |
| `tests/` | pytest checks for FlashAttention and the CUDA extension. |

## Run

```bash
uv pip install -e .
pytest tests                            # the CUDA extension test skips without a GPU
python -m gpu_systems.mixed_precision   # needs CUDA or Apple MPS

# On a Modal GPU
modal run modal/modal_runner.py::run_pytest
modal run modal/modal_runner.py::run_script --script-path modal/smoke_test.py
```

`benchmark.py` and the two profilers in `modal/` import the Assignment 1 model
(`cs336_basics`), which is not in this repo; my Assignment 1 code is in
[transformer-foundations](https://github.com/AbdullahMadoun/transformer-foundations).
The Modal runner installs it when a `cs336_basics/` folder sits at the repo root, and the
profilers expect it in `../assignment1-basics`.
