import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Mamba2Mixer(nn.Module):
    def __init__(self, d_model, expand=2, d_head=64, chunk_size=64, bias=False):
        """
        Mamba2-Style Mixer (Structured State Space Duality).

        Args:
            d_model: Input dimension.
            expand: Expansion factor for internal dimensions.
            d_head: Dimension per head (keep small, e.g., 64 or 128, for speed).
            chunk_size: Size of the chunks for block decomposition (64-128 is optimal).
        """
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_head = d_head
        self.chunk_size = chunk_size

        d_inner = int(d_model * expand)
        assert d_inner % d_head == 0, "Expanded dim must be divisible by head dim"
        self.n_heads = d_inner // d_head
        self.in_proj = nn.Linear(d_model, d_inner * 2 + d_inner + d_inner, bias=bias)

        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

        self.norm = nn.LayerNorm(d_inner)

    def forward(self, u, a=0.95):
        """
        Args:
            u: Input tensor (Batch, Seq_Len, Dim)
            a: Decay factor (scalar for simplicity in this implementation)
        """
        b, l, d = u.shape
        chunk_size = self.chunk_size

        z, q, k, v = self.in_proj(u).split([
            self.d_model * self.expand,
            self.d_model * self.expand,
            self.d_model * self.expand,
            self.d_model * self.expand
        ], dim=-1)

        q = F.silu(q)
        k = F.silu(k)
        # v = v
        pad_len = (chunk_size - (l % chunk_size)) % chunk_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        n_padded = l + pad_len
        n_chunks = n_padded // chunk_size
        q = q.view(b, n_chunks, chunk_size, self.n_heads, self.d_head)
        k = k.view(b, n_chunks, chunk_size, self.n_heads, self.d_head)
        v = v.view(b, n_chunks, chunk_size, self.n_heads, self.d_head)
        scores = torch.einsum('bclhd, bcmhd -> bchlm', q, k)
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=u.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, 0.0)
        y_local = torch.einsum('bchlm, bcmhd -> bclhd', scores, v)
        chunk_state = torch.einsum('bclhd, bclhe -> bchde', k, v)
        if a == 1.0:
            history_states = torch.cumsum(chunk_state, dim=1)
        else:
            decay_chunk = a ** chunk_size
            powers = torch.arange(n_chunks, device=u.device, dtype=torch.float32)
            powers = (decay_chunk ** powers).view(1, n_chunks, 1, 1, 1)
            history_states = torch.cumsum(chunk_state * (1.0 / (powers + 1e-6)), dim=1) * powers

        history_states = F.pad(history_states, (0, 0, 0, 0, 0, 0, 1, -1))

        y_global = torch.einsum('bclhd, bchde -> bclhd', q, history_states)

        y = y_local + y_global
        y = y.permute(0, 1, 2, 4, 3).reshape(b, n_padded, -1)

        y = y[:, :l, :]
        y = y * F.silu(z)

        return self.out_proj(y)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(2, 1000, 256).to(device)
    model = Mamba2Mixer(d_model=256, expand=2, d_head=64, chunk_size=128).to(device)

    with torch.no_grad():
        out = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print("Mamba2 (SSD) Mixer forward pass successful.")