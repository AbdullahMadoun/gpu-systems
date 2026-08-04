import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    flash_fwd_kernel = None

if HAS_TRITON:
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):
        """Triton FlashAttention-2 Forward Pass Kernel using Triton Block Pointers."""
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(0, 1),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(0, 1),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(0, 1),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES, 1),
            strides=(stride_lq, 1),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, 1),
            order=(0, 1),
        )

        Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        m = tl.full((Q_TILE_SIZE, 1), float("-inf"), dtype=tl.float32)
        l = tl.full((Q_TILE_SIZE, 1), 0.0, dtype=tl.float32)
        O_i = tl.full((Q_TILE_SIZE, D), 0.0, dtype=tl.float32)

        for j in range(N_KEYS // K_TILE_SIZE):
            K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

            S = tl.dot(Q_i, tl.trans(K_j)) * scale
            m_block = tl.max(S, axis=1, keep_dims=True)
            m_new = tl.maximum(m, m_block)

            alpha = tl.exp(m - m_new)
            alpha = tl.where(m == float("-inf"), 0.0, alpha)

            P = tl.exp(S - m_new)
            O_i = O_i * alpha
            O_i = tl.dot(P.to(Q_i.dtype), V_j, acc=O_i)
            l = l * alpha + tl.sum(P, axis=1, keep_dims=True)
            m = m_new

        O = O_i / l
        tl.store(O_block_ptr, O, boundary_check=(0, 1))
        L = m + tl.log(l)
        tl.store(L_block_ptr, L, boundary_check=(0, 1))
