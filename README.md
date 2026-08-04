# GPU Systems: GPU Engineering, FlashAttention-2, CUDA & Triton Kernels, & Profiling

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance GPU systems repository containing custom FlashAttention-2 implementations, C++/CUDA kernels, inline PyTorch extensions, CUDA shared-memory tiling benchmarks, and automated cloud GPU profiling runners (Nsight Systems & PyTorch Profiler).

---

## 📊 Benchmark Results & Reproducibility

Benchmarks were executed on remote **NVIDIA A10G Tensor Core GPUs (24GB VRAM)** orchestrated via serverless Modal containers:

### 1. Triton Softmax & RMSNorm Kernel Speedup (32K Vocab Size)
Row-wise tiled online softmax (`triton_softmax.py`) and RMS Layer Normalization (`triton_rmsnorm.py`) benchmarked against native PyTorch `torch.softmax` and `torch.nn.RMSNorm` at $V = 32,768$ vocabulary size ($B=4, S=512$):

| Kernel / Operator | PyTorch Native Latency | Triton Custom Kernel | Peak Acceleration |
| :--- | :--- | :--- | :--- |
| **Row-wise Tiled Online Softmax ($V=32,768$)** | `1.85 ms` | `0.49 ms` | **3.77× (~3.8×)** |
| **RMS Normalization ($D=4,096$)** | `0.42 ms` | `0.18 ms` | **2.33×** |
| **FlashAttention-2 vs. Standard Attention ($S=2,048$)** | `6.12 ms` | `2.14 ms` | **2.86×** |

> **Reproducibility**: Run `python3 -m gpu_systems.benchmark` or execute remotely on Modal via `modal run modal/modal_runner.py`.

### 2. CUDA Shared-Memory Matmul Tiling ($32 \times 32$ Blocks)
Evaluated up to $16,384 \times 16,384$ matrix dimensions comparing global VRAM reads vs. $32 \times 32$ `__shared__` memory tiles (`cuda_src/tile.cu`):

| Matrix Size ($N \times N$) | Naive CUDA Matmul | Shared-Memory Tiled Matmul | Speedup |
| :--- | :--- | :--- | :--- |
| **$2,048 \times 2,048$** | `14.2 ms` | `4.1 ms` | **3.46×** |
| **$8,192 \times 8,192$** | `842.0 ms` | `231.5 ms` | **3.64×** |
| **$16,384 \times 16,384$** | `6,820.0 ms` | `1,810.0 ms` | **3.76×** |

---

## 🛠️ Codebase Structure & Documentation

### 1. FlashAttention-2 Implementations (`src/gpu_systems/kernels/`)
- **Triton FlashAttention-2 Forward Kernel (`triton_flash_attn.py`)**:
  - Hardware-blocked attention GPU kernel using Triton block pointers (`tl.make_block_ptr`, `tl.advance`), online row-max tracking ($m_{\text{new}} = \max(m, m_{\text{block}})$), and online exponential rescaling ($\alpha = e^{m - m_{\text{new}}}$).
- **PyTorch Tiled FlashAttention-2 Operator (`flash_attention.py`)**:
  - Tiled sequence attention autograd operator (`FlashAttention_2.apply`) tracking query/key block tiles ($B_q, B_k$), row max values, and online output accumulator rescaling. Verified against native PyTorch attention (`torch.allclose`).

### 2. Custom C++/CUDA Kernels & Inline Extensions (`cuda_src/` & `src/gpu_systems/kernels/`)
- **Shared-Memory Matmul Tiling (`cuda_src/tile.cu` & `tile_vs_non_tile_mm.py`)**:
  - 2D CUDA shared-memory matrix multiplication (`tiled_matmul`) using $32 \times 32$ `__shared__` memory tiles (`A_tile`, `B_tile`) and thread barrier synchronization (`__syncthreads()`).
  - Benchmarked against naive global memory matmul up to $16,384 \times 16,384$ scale on NVIDIA GPUs using CUDA events (`cudaEventElapsedTime`), plotted via matplotlib.
- **Triton High-Performance Kernels (`src/gpu_systems/kernels/`)**:
  - `triton_rmsnorm.py`: Vectorized Triton RMS Layer Normalization.
  - `triton_softmax.py`: Vectorized Triton Tiled Online Softmax and 3D vocabulary softmax.
- **Inline C++/CUDA Image Processing Extensions (`src/gpu_systems/kernels/untitled29.py`)**:
  - Bound via `torch.utils.cpp_extension.load_inline`:
    * `MMkernel`: 2D matrix multiplication CUDA kernel.
    * `to_grayscale_kernel` & `to_grayscale`: RGB to Grayscale conversion kernel ($\text{Gray} = 0.21R + 0.71G + 0.07B$).
    * `box_blur_kernel` & `box_blur`: 1-channel spatial moving-average box blur filter.
    * `box_blur_rgb_kernel` & `rgb_blur`: 3-channel RGB image box blur filter with CUDA strided indexing `(row * width + col) * 3`.
- **Host Launchers (`cuda_src/stuff.cu` & `MM.cu`)**:
  - Host wrapper launchers and CUDA grid/block dimension setups.

### 3. Benchmarking & Precision Inspection (`src/gpu_systems/`)
- **`benchmark.py`**: Timing harness benchmarking `forward`, `forward_backward`, and `full` training step modes across model configs with NVTX range markers (`nvtx.range_push`/`pop`) and LaTeX summary table formatting.
- **`model_configs.py`**: Model architecture specifications (GPT-2 `small` to `10B`) and JSON config file generation.
- **`benchmarking_mixed_precision.py`**: PyTorch Autocast precision type inspector evaluating Float16 linear layers and Float32 LayerNorm stability.

### 4. Serverless Modal Cloud Profiling (`modal/`)
- **`modal_runner.py`**: Serverless Modal CLI app running unit tests and benchmarks on remote GPU containers (T4, A10G, A100-80GB).
- **`modal_tb_profiler.py`**: Runs PyTorch Profiler on remote A10G GPUs, writing traces to Modal persistent volume `gpu-systems-traces` and serving an embedded WSGI TensorBoard UI.
- **`modal_nsys_profiler.py`**: Generates Nsight Systems `.nsys-rep` trace files remotely on A10G GPUs and downloads them to your Desktop.
- **`smoke_test.py`**: Modal container diagnostic check verifying CUDA availability, dependencies, and imports.

---

## 📐 Repository Tree

```
gpu-systems/
├── pyproject.toml              # Project dependencies managed via uv
├── README.md                   # Technical documentation & benchmark results
├── LICENSE                     # MIT License
├── src/gpu_systems/            # Core Python Package
│   ├── __init__.py
│   ├── model_configs.py        # Model size configurations
│   ├── benchmark.py            # Forward/Backward/Full step timing harness
│   ├── benchmarking_mixed_precision.py # Autocast precision inspector
│   └── kernels/                # FlashAttention, Triton & CUDA extensions
│       ├── flash_attention.py  # PyTorch FlashAttention-2 operator
│       ├── triton_flash_attn.py# Triton FlashAttention-2 forward kernel
│       ├── triton_rmsnorm.py   # Triton RMS Layer Normalization GPU kernel
│       ├── triton_softmax.py   # Triton Tiled Online Softmax
│       ├── untitled29.py       # Inline C++/CUDA image filters & matmul
│       └── tile_vs_non_tile_mm.py # 16K scale CUDA matmul tiling benchmark
├── cuda_src/                   # Raw C++/CUDA Source Files
│   ├── tile.cu                 # 2D Tiled Shared Memory Matmul kernel
│   ├── stuff.cu                # SwiGLU host launcher
│   └── MM.cu                   # Matrix multiplication host wrapper
└── modal/                      # Serverless Modal Cloud Utilities
    ├── modal_runner.py         # Remote GPU execution harness
    ├── modal_tb_profiler.py    # Remote PyTorch Profiler + WSGI TensorBoard app
    ├── modal_nsys_profiler.py  # Remote Nsight Systems profiler
    └── smoke_test.py           # Container diagnostic test suite
```

---

## ⚡ Quick Start

```bash
cd gpu-systems
uv pip install -e .

# Run Benchmarks
python3 -m gpu_systems.benchmark
```
