import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    rmsnorm_kernel = None

if HAS_TRITON:
    @triton.jit
    def rmsnorm_kernel(
        x_ptr,
        g_ptr,
        output_ptr,
        B_stride, S_stride, H_stride,
        H,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton RMS Layer Normalization GPU Kernel."""
        B_pid = tl.program_id(axis=0)
        S_pid = tl.program_id(axis=1)
        local_offsets = tl.arange(0, BLOCK_SIZE)
        global_offsets = (B_pid * B_stride) + (S_pid * S_stride) + (local_offsets * H_stride)
        g_offsets = local_offsets * H_stride
        
        mask = local_offsets < H
        x = tl.load(x_ptr + global_offsets, mask=mask, other=0.0)
        g = tl.load(g_ptr + g_offsets, mask=mask, other=0.0)
        
        x_fp32 = x.to(tl.float32)
        rms = tl.rsqrt(tl.sum(x_fp32 * x_fp32) / H + eps)
        x_scaled = x_fp32 * rms
        output = g * (x_scaled).to(x.dtype)
        tl.store(output_ptr + global_offsets, output, mask=mask)

def triton_rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Launcher wrapper for Triton RMSNorm kernel."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed on this system.")
    
    assert x.is_cuda, "Input tensor x must be on CUDA."
    output = torch.empty_like(x)
    B, S, H = x.shape
    BLOCK_SIZE = triton.next_power_of_2(H)
    grid = (B, S)
    
    rmsnorm_kernel[grid](
        x, g, output,
        x.stride(0), x.stride(1), x.stride(2),
        H,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
