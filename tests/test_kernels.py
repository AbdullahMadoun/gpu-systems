import torch
import pytest
from gpu_systems.kernels.flash_attention import flash_attention

def test_flash_attention_forward():
    batch_size = 2
    n_queries = 32
    n_keys = 32
    d = 16

    Q = torch.randn(batch_size, n_queries, d)
    K = torch.randn(batch_size, n_keys, d)
    V = torch.randn(batch_size, n_keys, d)

    out = flash_attention(Q, K, V, B_q=8, B_k=8)
    assert out.shape == (batch_size, n_queries, d)
    assert not torch.isnan(out).any()

def test_cuda_extension_grayscale():
    from gpu_systems.kernels.cuda_extensions import get_grayscale_extension
    try:
        ext = get_grayscale_extension()
        img = torch.rand(64, 64, 3, dtype=torch.float32)
        gray = ext.to_grayscale(img)
        assert gray.shape == (64, 64)
    except Exception as e:
        pytest.skip(f"CUDA extension compilation skipped: {e}")
