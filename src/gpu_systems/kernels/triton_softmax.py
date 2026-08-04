import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    tiled_online_softmax_kernel = None
    softmax_kernel_last_layer = None

if HAS_TRITON:
    @triton.jit
    def tiled_online_softmax_kernel(
        X_ptr, Y_ptr,
        stride_x_row, stride_y_row,
        N_COLS,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton Tiled Online Softmax Kernel for long row vectors."""
        row_idx = tl.program_id(axis=0)
        row_X_ptr = X_ptr + row_idx * stride_x_row
        row_Y_ptr = Y_ptr + row_idx * stride_y_row

        row_max = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
        row_sum = tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float32)

        for offset in range(0, N_COLS, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N_COLS
            x_chunk = tl.load(row_X_ptr + cols, mask=mask, other=float("-inf"))
            new_max = tl.maximum(row_max, x_chunk)
            row_sum = row_sum * tl.exp(row_max - new_max) + tl.exp(x_chunk - new_max)
            row_max = new_max

        global_max = tl.max(row_max, axis=0)
        global_sum = tl.sum(row_sum * tl.exp(row_max - global_max), axis=0)

        for offset in range(0, N_COLS, BLOCK_SIZE):
            cols = offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < N_COLS
            x_chunk = tl.load(row_X_ptr + cols, mask=mask, other=float("-inf"))
            y_chunk = tl.exp(x_chunk - global_max) / global_sum
            tl.store(row_Y_ptr + cols, y_chunk, mask=mask)

    @triton.jit
    def softmax_kernel_last_layer(
        x_ptr, output_ptr,
        B_stride, S_stride, V_stride,
        V,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton Fast Softmax Kernel for 3D tensor (B, S, V) on vocabulary dimension V."""
        B_pid = tl.program_id(axis=0)
        S_pid = tl.program_id(axis=1)
        local_v_offsets = tl.arange(0, BLOCK_SIZE)
        global_offsets = (B_pid * B_stride) + (S_pid * S_stride) + (local_v_offsets * V_stride)
        
        x_ptrs = x_ptr + global_offsets
        out_ptrs = output_ptr + global_offsets
        mask = local_v_offsets < V
        
        v_data = tl.load(x_ptrs, mask=mask, other=float("-inf"))
        v_max = tl.max(v_data, axis=0)
        v_normed = v_data - v_max
        v_exp = tl.exp(v_normed)
        v_exp_masked = tl.where(mask, v_exp, 0.0)
        v_sum = tl.sum(v_exp_masked, axis=0)
        v_softmax = v_exp / v_sum
        tl.store(out_ptrs, v_softmax, mask=mask)

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Launcher wrapper for Triton 3D Softmax kernel."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed on this system.")
        
    assert x.is_cuda, "Input tensor x must be on CUDA."
    output = torch.empty_like(x)
    B, S, V = x.shape
    BLOCK_SIZE = triton.next_power_of_2(V)
    grid = (B, S)
    
    softmax_kernel_last_layer[grid](
        x, output,
        x.stride(0), x.stride(1), x.stride(2),
        V,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
