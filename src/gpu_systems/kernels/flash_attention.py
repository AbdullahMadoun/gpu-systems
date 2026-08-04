import math
import torch

class FlashAttention_2(torch.autograd.Function):
    """
    Pure PyTorch autograd implementation of FlashAttention-2.
    Tracks sequence tiling (B_q, B_k), row-max tracking (m_new), online output rescaling,
    and row-sum tracking (l_i).
    """
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, B_q: int = 16, B_k: int = 16) -> torch.Tensor:
        """
        Forward pass for FlashAttention-2.
        Q: (batch, n_queries, d)
        K: (batch, n_keys, d)
        V: (batch, n_keys, d)
        """
        batch_size, n_queries, d = Q.shape
        _, n_keys, _ = K.shape
        scale = 1.0 / math.sqrt(d)

        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, n_queries), device=Q.device, dtype=Q.dtype)

        num_q_blocks = math.ceil(n_queries / B_q)
        num_k_blocks = math.ceil(n_keys / B_k)

        for b in range(batch_size):
            Q_b = Q[b]
            K_b = K[b]
            V_b = V[b]

            for i in range(num_q_blocks):
                q_start = i * B_q
                q_end = min((i + 1) * B_q, n_queries)
                Q_i = Q_b[q_start:q_end] # (q_block_len, d)

                m_i = torch.full((Q_i.shape[0], 1), float("-inf"), device=Q.device, dtype=Q.dtype)
                l_i = torch.zeros((Q_i.shape[0], 1), device=Q.device, dtype=Q.dtype)
                O_i = torch.zeros((Q_i.shape[0], d), device=Q.device, dtype=Q.dtype)

                for j in range(num_k_blocks):
                    k_start = j * B_k
                    k_end = min((j + 1) * B_k, n_keys)
                    K_j = K_b[k_start:k_end] # (k_block_len, d)
                    V_j = V_b[k_start:k_end] # (k_block_len, d)

                    # Compute score tile
                    S_ij = (Q_i @ K_j.T) * scale # (q_block_len, k_block_len)
                    m_ij = torch.max(S_ij, dim=1, keepdim=True).values
                    m_new = torch.maximum(m_i, m_ij)

                    # Online exponential scaling factors
                    alpha = torch.exp(m_i - m_new)
                    alpha[m_i == float("-inf")] = 0.0

                    P_ij = torch.exp(S_ij - m_new)
                    O_i = O_i * alpha + P_ij @ V_j
                    l_i = l_i * alpha + P_ij.sum(dim=1, keepdim=True)
                    m_i = m_new

                O[b, q_start:q_end] = O_i / l_i
                L[b, q_start:q_end] = (m_i + torch.log(l_i)).squeeze(-1)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.B_q = B_q
        ctx.B_k = B_k
        return O

def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, B_q: int = 16, B_k: int = 16) -> torch.Tensor:
    """Wrapper function invoking FlashAttention_2 autograd operator."""
    return FlashAttention_2.apply(Q, K, V, B_q, B_k)
