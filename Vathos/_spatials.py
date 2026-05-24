from Vathos._basics import *


class SinusoidalPositionalEncoding(Layer):
    __name__ = "SinusoidalPositionalEncoding"
    __complexity__ = "O(L^2 d^2)"

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        return x + self.pe[:L]


class RoPE(Layer):
    __name__ = "RoPE"

    def __init__(self, dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_len = max_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))

            self._cos_cached = freqs.cos()
            self._sin_cached = freqs.sin()

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          start_pos: int = 0) -> torch.Tensor:
        """Apply rotary embeddings starting from start_pos"""
        seq_len = x.shape[-2]

        cos = cos[start_pos:start_pos + seq_len]
        sin = sin[start_pos:start_pos + seq_len]

        shape = [1] * x.ndim
        shape[-2] = seq_len
        shape[-1] = self.dim // 2

        cos = cos.view(*shape)
        sin = sin.view(*shape)

        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, start_pos: int = 0):
        """
        Args:
            q: Query tensor
            k: Key tensor (optional)
            start_pos: Starting position for RoPE (used during generation)
        """
        assert q.shape[-1] == self.dim, f"Last dim of q must be {self.dim}, got {q.shape[-1]}"
        if k is not None:
            assert k.shape[-2:] == q.shape[-2:], "k must have same seq_len and head_dim as q"

        seq_len = q.shape[-2]
        self._update_cache(start_pos + seq_len, q.dtype, q.device)

        cos = self._cos_cached
        sin = self._sin_cached

        q_rope = self._apply_rotary_emb(q, cos, sin, start_pos)
        k_rope = self._apply_rotary_emb(k, cos, sin, start_pos) if k is not None else None

        return (q_rope, k_rope) if k_rope is not None else q_rope


class ALiBi(Layer):
    __name__ = "ALiBi"

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

        # Calculate standard ALiBi geometric slopes
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
        slopes = [math.pow(base, i) for i in range(1, closest_power_of_2 + 1)]

        # Handle non-power-of-2 head counts (e.g., 12 heads)
        if closest_power_of_2 != n_heads:
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            slopes.extend([math.pow(extra_base, i) for i in range(1, 2 * (n_heads - closest_power_of_2) + 1, 2)])

        self.register_buffer(
            "slopes",
            torch.tensor(slopes, dtype=torch.float32).view(n_heads, 1, 1),
            persistent=False
        )

    def forward(self, q_len: int, k_len: int, start_pos: int, device: torch.device, dtype: torch.dtype,
                causal: bool = True) -> torch.Tensor:
        # Create position indices
        q_idx = torch.arange(start_pos, start_pos + q_len, device=device)[:, None]
        k_idx = torch.arange(0, k_len, device=device)[None, :]

        # Calculate relative distances (j - i).
        # For causal attention, we only attend to the past, so k_idx <= q_idx, meaning distances <= 0.
        distances = k_idx - q_idx

        # Scale distances by head-specific slopes
        alibi_bias = distances * self.slopes
        alibi_bias = alibi_bias.to(dtype)

        # Embed the causal mask directly into the ALiBi bias
        if causal:
            causal_mask = distances > 0
            alibi_bias.masked_fill_(causal_mask, float('-inf'))

        # Add batch dimension for SDPA broadcasting -> (1, n_heads, q_len, k_len)
        return alibi_bias.unsqueeze(0)


class YaRN(Layer):
    __name__ = "YaRN"

    def __init__(self, dim: int, original_max_len: int = 4096, scale: float = 1.0,
                 base: float = 10000.0, beta_fast: int = 32, beta_slow: int = 1):
        super().__init__()
        self.dim = dim
        self.base = base
        self.original_max_len = original_max_len
        self.scale = scale

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        if scale > 1.0:
            wavelengths = 2 * math.pi / inv_freq

            low = original_max_len / beta_slow
            high = original_max_len / beta_fast

            w = torch.clamp((wavelengths - high) / (low - high), 0.0, 1.0)

            inv_freq_interpolated = inv_freq / scale
            inv_freq_extrapolated = inv_freq

            inv_freq = (1 - w) * inv_freq_extrapolated + w * inv_freq_interpolated

            self.mscale = math.sqrt(0.1 * math.log(scale) + 1.0)
        else:
            self.mscale = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cache(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))

            # Apply the YaRN mscale directly to the cache
            self._cos_cached = (freqs.cos() * self.mscale)
            self._sin_cached = (freqs.sin() * self.mscale)

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          start_pos: int = 0) -> torch.Tensor:
        """Apply rotary embeddings starting from start_pos"""
        seq_len = x.shape[-2]

        cos = cos[start_pos:start_pos + seq_len]
        sin = sin[start_pos:start_pos + seq_len]

        shape = [1] * x.ndim
        shape[-2] = seq_len
        shape[-1] = self.dim // 2

        cos = cos.view(*shape).to(x.dtype)
        sin = sin.view(*shape).to(x.dtype)

        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, start_pos: int = 0):
        assert q.shape[-1] == self.dim, f"Last dim of q must be {self.dim}, got {q.shape[-1]}"
        if k is not None:
            assert k.shape[-2:] == q.shape[-2:], "k must have same seq_len and head_dim as q"

        seq_len = q.shape[-2]
        self._update_cache(start_pos + seq_len, q.dtype, q.device)

        cos = self._cos_cached
        sin = self._sin_cached

        q_rope = self._apply_rotary_emb(q, cos, sin, start_pos)
        k_rope = self._apply_rotary_emb(k, cos, sin, start_pos) if k is not None else None

        return (q_rope, k_rope) if k_rope is not None else q_rope


class NanoYaRN(Layer):
    """Modded Nano-Gpt implmenetation of YaRN"""

    def __init__(self, head_dim, max_seq_len, paired=False):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim // 4, dtype=torch.float32)
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim // 2)])
        t = torch.arange(2 * self.max_seq_len, dtype=torch.float32)
        if not self.paired:
            theta = torch.outer(t, angular_freq)
            self.factor1 = nn.Buffer(
                theta.cos().to(torch.bfloat16), persistent=False
            )
            self.factor2 = nn.Buffer(
                theta.sin().to(torch.bfloat16), persistent=False
            )
        else:
            t_even = 2 * t
            t_odd = 2 * t + 1
            theta1 = torch.outer(t_even, angular_freq)
            theta2 = torch.outer(t_odd, angular_freq)
            self.factor1 = nn.Buffer(
                torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
            self.factor2 = nn.Buffer(
                torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int = 1, beta: int = 32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2 * self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        if not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1.copy_(theta.cos())
            self.factor2.copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = 2 * t + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1.copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2.copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


class MultiheadAttentionMixer(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 pos_emb: nn.Module = None, dropout: float = 0.0, qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.pos_emb = pos_emb
        self.kv_cache = None
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor, ve=None) -> torch.Tensor:
        """
        ve: Stands for value embeddings, which should be already weighted byt the caller, of course the dimension must match x
        """
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)
        if ve is not None:
            v = v + ve.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def show_weights_forward(self, x):
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.causal:
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(attn_weights, v)

        out = self.out(attn.transpose(1, 2).contiguous().view(B, L, D))
        return out, attn_weights

    def generate(self, x: torch.Tensor, ve=None) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        pos_offset = self.kv_cache[0].shape[2] if self.kv_cache is not None else 0

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=pos_offset)

        if ve is not None:
            v = v + ve.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal and (L > 1),
            dropout_p=0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qkv.weight.requires_grad = False
        self.out.weight.requires_grad = False


class MultiheadGatedAttentionMixer(Layer):
    """MHA + sparse per-head output gate (Modded-NanoGPT PR #117 / parameter-golf 1667).

    Gate input: prime `gate_input_dim` (default 12) dimensioni di x — sparse, ottimizzato
    per Muon. Output: un sigmoid scalare per head, broadcast su head_dim. Init zero ⇒
    gate parte a 0.5 (no-op).
    """
    __name__ = "MultiheadGatedAttentionMixer"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 pos_emb: nn.Module = None, dropout: float = 0.0, qk_norm: bool = False,
                 gate_input_dim: int = 12):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.gate_input_dim = gate_input_dim

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(gate_input_dim, n_heads, bias=False)

        self.pos_emb = pos_emb
        self.kv_cache = None
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.gate_proj.weight)

    def _apply_gate(self, attn: torch.Tensor, x_gate_in: torch.Tensor) -> torch.Tensor:
        # attn: [B, H, L, D_h] | x_gate_in: [B, L, gate_input_dim]
        gate = torch.sigmoid(self.gate_proj(x_gate_in))                 # [B, L, H]
        return attn * gate.transpose(1, 2).unsqueeze(-1)                # broadcast su D_h

    def forward(self, x: torch.Tensor, ve=None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)
        if ve is not None:
            v = v + ve.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn = self._apply_gate(attn, x[..., :self.gate_input_dim])
        return self.out(attn.transpose(1, 2).reshape(B, L, D))

    def generate(self, x: torch.Tensor, ve=None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        pos_offset = self.kv_cache[0].shape[2] if self.kv_cache is not None else 0
        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=pos_offset)
        if ve is not None:
            v = v + ve.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        self.kv_cache = (k, v)

        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal and (L > 1), dropout_p=0.0,
        )
        attn = self._apply_gate(attn, x[..., :self.gate_input_dim])
        return self.out(attn.transpose(1, 2).reshape(B, L, D))

    def clear_cache(self):
        self.kv_cache = None


class MultiheadAttentionMixerALIBI(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 pos_emb: nn.Module = None, alibi: nn.Module = None, dropout: float = 0.0, qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.pos_emb = pos_emb
        self.alibi = alibi  # ADDED: ALiBi module injection
        self.kv_cache = None
        self.qk_norm = qk_norm

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor, ve=None) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)

        if ve is not None:
            v = v + ve.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # ADDED: ALiBi Mask Logic
        attn_mask = None
        is_causal = self.causal
        if self.alibi is not None:
            attn_mask = self.alibi(q_len=L, k_len=L, start_pos=0, device=x.device, dtype=q.dtype, causal=self.causal)
            is_causal = False  # PyTorch requires is_causal=False if a custom attn_mask is provided

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def show_weights_forward(self, x):
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # ADDED: ALiBi logic replaces strict causal mask if active
        if self.alibi is not None:
            alibi_bias = self.alibi(q_len=L, k_len=L, start_pos=0, device=x.device, dtype=q.dtype, causal=self.causal)
            scores = scores + alibi_bias
        elif self.causal:
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(attn_weights, v)

        out = self.out(attn.transpose(1, 2).contiguous().view(B, L, D))
        return out, attn_weights

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        pos_offset = self.kv_cache[0].shape[2] if self.kv_cache is not None else 0

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=pos_offset)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)
        k_len = k.shape[2]

        # ADDED: ALiBi Mask Logic for Generation
        attn_mask = None
        is_causal = self.causal and (L > 1)
        if self.alibi is not None:
            attn_mask = self.alibi(q_len=L, k_len=k_len, start_pos=pos_offset, device=x.device, dtype=q.dtype,
                                   causal=self.causal)
            is_causal = False

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    # [clear_cache and finetune remain unchanged]


class MultiheadAttentionMixerXSA(Layer):
    __name__ = "MultiheadAttentionMixerXSA"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 pos_emb: nn.Module = None, dropout: float = 0.0, qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.pos_emb = pos_emb
        self.kv_cache = None
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor, ve=None) -> torch.Tensor:
        """
        ve: Stands for value embeddings, which should be already weighted by the caller, of course the dimension must match x
        """
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)
        if ve is not None:
            v = v + ve

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        v_norm = F.normalize(v, dim=-1)
        attn = attn - (attn * v_norm).sum(dim=-1, keepdim=True) * v_norm

        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def show_weights_forward(self, x):
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.causal:
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(attn_weights, v)

        v_norm = F.normalize(v, dim=-1)
        attn = attn - (attn * v_norm).sum(dim=-1, keepdim=True) * v_norm

        out = self.out(attn.transpose(1, 2).contiguous().view(B, L, D))
        return out, attn_weights

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        pos_offset = self.kv_cache[0].shape[2] if self.kv_cache is not None else 0

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=pos_offset)

        v_self = v

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal and (L > 1),
            dropout_p=0.0,
        )

        v_norm = F.normalize(v_self, dim=-1)
        attn = attn - (attn * v_norm).sum(dim=-1, keepdim=True) * v_norm

        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qkv.weight.requires_grad = False
        self.out.weight.requires_grad = False


class MultiheadAttentionMixerNOV(Layer):
    __name__ = "MultiheadAttentionMixerNOV"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 pos_emb: nn.Module = None, dropout: float = 0.0, qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qk = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.pos_emb = pos_emb
        self.kv_cache = None

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qk.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qk = self.qk(x).view(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        v = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qk = self.qk(x).view(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        v = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        pos_offset = self.kv_cache[0].shape[2] if self.kv_cache is not None else 0

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=pos_offset)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal and (L > 1),
            dropout_p=0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qk.weight.requires_grad = False
        self.out.weight.requires_grad = False


class MultiheadAttentionMixerNOVLa2(Layer):
    __name__ = "MultiheadAttentionMixerNOV"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 pos_emb: nn.Module = None, dropout: float = 0.0, qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qk = nn.Linear(d_model, 2 * d_model, bias=False)
        self.sec = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.pos_emb = pos_emb
        self.kv_cache = None

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qk.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.xavier_uniform_(self.sec.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qk = self.qk(self.sec(x)).view(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        v = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qk = self.qk(x).view(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        v = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        pos_offset = self.kv_cache[0].shape[2] if self.kv_cache is not None else 0

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=pos_offset)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal and (L > 1),
            dropout_p=0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qk.weight.requires_grad = False
        self.out.weight.requires_grad = False


class MultiheadAttentionMixerNOVLa3(Layer):
    __name__ = "MultiheadAttentionMixerNOV"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 pos_emb: nn.Module = None, dropout: float = 0.0, qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qk = DoubleLinear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.pos_emb = pos_emb
        self.kv_cache = None

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qk = self.qk(x).view(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        v = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qk = self.qk(x).view(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        v = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        pos_offset = self.kv_cache[0].shape[2] if self.kv_cache is not None else 0

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=pos_offset)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal and (L > 1),
            dropout_p=0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qk.weight.requires_grad = False
        self.out.weight.requires_grad = False


class MultiheadAttentionMixerNOVLa4(Layer):
    __name__ = "MultiheadAttentionMixerNOV"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 pos_emb: nn.Module = None, dropout: float = 0.0, qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qk = DoubleLinear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.pos_emb = pos_emb
        self.kv_cache = None

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qk = self.qk(x).view(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        v = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qk = self.qk(x).view(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.unbind(dim=2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        v = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        pos_offset = self.kv_cache[0].shape[2] if self.kv_cache is not None else 0

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=pos_offset)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal and (L > 1),
            dropout_p=0.0,
        )
        return self.out(attn.transpose(1, 2).contiguous().view(B, L, D))

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qk.weight.requires_grad = False
        self.out.weight.requires_grad = False


class GroupedQueryAttention(Layer):
    __name__ = "GroupedQueryAttention"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 causal: bool = True, pos_emb: nn.Module = None,
                 dropout: float = 0.0, bias: bool = False, qk_norm=False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.causal = causal
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, n_kv_heads * self.head_dim * 2, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.pos_emb = pos_emb
        self.kv_cache = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.kv_proj.bias)
            nn.init.zeros_(self.o_proj.bias)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, H, L, D = x.shape
        return (x.unsqueeze(2)
                .expand(B, H, self.n_rep, L, D)
                .reshape(B, H * self.n_rep, L, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2, self.head_dim)
        k, v = kv.unbind(dim=3)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=0)

        k, v = self._expand_kv(k), self._expand_kv(v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.o_proj(attn.transpose(1, 2).contiguous().view(B, T, C))

    def generate(self, x: torch.Tensor, ve=None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2, self.head_dim)
        k, v = kv.unbind(dim=3)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        pos_offset = self.kv_cache[0].shape[2] if self.kv_cache is not None else 0

        if self.pos_emb is not None:
            q, k = self.pos_emb(q, k, start_pos=pos_offset)

        if ve is not None:
            raise NotImplementedError(
                "GroupedQueryAttention.forward does not support value embeddings; "
                "use MultiheadAttentionMixer if you need ve."
            )

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)

        k, v = self._expand_kv(k), self._expand_kv(v)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal and (T > 1),
            dropout_p=0.0,
        )
        return self.o_proj(attn.transpose(1, 2).contiguous().view(B, T, C))

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.q_proj.weight.requires_grad = False
        self.kv_proj.weight.requires_grad = False
        self.o_proj.weight.requires_grad = False


class GroupedQueryAttentionNOV(Layer):

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int,
            dropout: float = 0.0,
            bias: bool = False,
            causal: bool = True,
            rope: bool = False,
            qk_norm: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = n_heads
        self.num_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout_prob = dropout
        self.causal = causal

        self.n_rep = self.num_heads // self.num_kv_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(f"embed_dim ({d_model}) must be divisible by num_heads ({n_heads})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({n_heads}) must be divisible by num_kv_heads ({n_kv_heads})")

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RoPE(self.head_dim) if rope else None

        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.o_proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q, k)

        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.n_rep, T, self.head_dim)
            k = k.reshape(B, self.num_heads, T, self.head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=self.causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)


class GroupedQueryAttentionNOV2(Layer):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int,
            dropout: float = 0.0,
            bias: bool = False,
            causal: bool = True,
            rope: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = n_heads
        self.num_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout_prob = dropout
        self.causal = causal

        self.n_rep = self.num_heads // self.num_kv_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(f"embed_dim ({d_model}) must be divisible by num_heads ({n_heads})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({n_heads}) must be divisible by num_kv_heads ({n_kv_heads})")

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.trainv = False
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RoPE(self.head_dim) if rope else None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.eye_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.k_proj.bias, 0)
            nn.init.constant_(self.v_proj.bias, 0)
            nn.init.constant_(self.o_proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if not self.trainv:
            v = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE BEFORE expansion
        if self.rope is not None:
            q, k = self.rope(q, k)

        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.n_rep, T, self.head_dim)
            k = k.reshape(B, self.num_heads, T, self.head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=self.causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_output)


class GroupedQueryAttentionNOO(Layer):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_kv_heads: int,
            dropout: float = 0.0,
            bias: bool = False,
            causal: bool = True,
            rope: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = n_heads
        self.num_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout_prob = dropout
        self.causal = causal

        self.n_rep = self.num_heads // self.num_kv_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError(f"embed_dim ({d_model}) must be divisible by num_heads ({n_heads})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({n_heads}) must be divisible by num_kv_heads ({n_kv_heads})")

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, n_kv_heads * self.head_dim * 2, bias=bias)

        self.rope = RoPE(self.head_dim) if rope else None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.kv_proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        kv = self.kv_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        kv = kv.view(B, T, self.num_kv_heads, 2, self.head_dim)
        k, v = kv.unbind(dim=3)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE BEFORE expansion
        if self.rope is not None:
            q, k = self.rope(q, k)

        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.n_rep, T, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, self.n_rep, T, self.head_dim)
            k = k.reshape(B, self.num_heads, T, self.head_dim)
            v = v.reshape(B, self.num_heads, T, self.head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=self.causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return attn_output


class MultiheadLatentAttentionMixer(Layer):  # Changed Layer to nn.Module for standard torch
    __name__ = "MultiheadLatentAttentionMixer"
    # Adjusted complexity notation
    __complexity__ = "O(L^2 d + L d * d_kv_lora)"

    def __init__(self, d_model: int, n_heads: int, d_kv_lora: int, causal: bool, rope=False, dropout=0.1):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_kv_lora = d_kv_lora
        self.rope_enabled = rope

        self.q = nn.Linear(d_model, d_model, bias=False)

        self.kv_compress = nn.Linear(d_model, d_kv_lora, bias=False)
        self.k_content_decompress = nn.Linear(d_kv_lora, d_model, bias=False)
        self.v_decompress = nn.Linear(d_kv_lora, d_model, bias=False)

        self.out = nn.Linear(d_model, d_model, bias=False)

        if self.rope_enabled:
            self.rope = RoPE(self.head_dim)

        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        H = self.n_heads
        HD = self.head_dim

        q = self.q(x).view(B, L, H, HD).transpose(1, 2)

        c_kv = self.kv_compress(x)
        k = self.k_content_decompress(c_kv).view(B, L, H, HD).transpose(1, 2)
        v = self.v_decompress(c_kv).view(B, L, H, HD).transpose(1, 2)
        if self.rope_enabled:
            q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.causal
        )

        attn = attn.transpose(1, 2).contiguous().view(B, L, D)

        return self.out(attn)


class CausalMultiheadAttentionMixer2NOK(nn.Module):
    __name__ = "CausalMultiheadAttentionMixer"

    def __init__(self, d_model: int, n_heads: int, causal=True, rope=False, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(self.head_dim) if rope else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qv = self.qv(x)

        qv = qv.view(B, L, 2, self.n_heads, self.head_dim)

        q, v = qv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0
        )

        attn = attn.transpose(1, 2).contiguous().view(B, L, D)

        return self.out(attn)


class MultiheadDecoupledAttention(Layer):
    __name__ = "DecoupledSelfAttention"
    __complexity__ = "O(L^2 * d_qk + L * d_model * (d_qk + d_v))"

    def __init__(self, d_model: int, n_heads: int, qk_dim: int, causal: bool, rope=False, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qk_dim = qk_dim
        self.v_dim = d_model // n_heads
        self.causal = causal
        self.rope_enabled = rope

        self.qk_proj = nn.Linear(d_model, n_heads * 2 * qk_dim, bias=False)

        self.v_proj = nn.Linear(d_model, n_heads * self.v_dim, bias=False)

        self.out_proj = nn.Linear(n_heads * self.v_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        if self.rope_enabled:
            self.rope = RoPE(qk_dim)

        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        H = self.n_heads

        q, k = self.qk_proj(x).view(B, L, H, 2 * self.qk_dim).transpose(1, 2).chunk(2, -1)

        v = self.v_proj(x).view(B, L, H, self.v_dim).transpose(1, 2)

        if self.rope_enabled:
            q, k = self.rope(q, k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.causal
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H * self.v_dim)

        return self.out_proj(attn_out)


class MultiheadDecoupledAttentionNOV(Layer):
    __name__ = "DecoupledSelfAttention"
    __complexity__ = "O(L^2 * d_qk + L * d_model * (d_qk + d_v))"

    def __init__(self, d_model: int, n_heads: int, qk_dim: int, causal: bool, rope=False, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qk_dim = qk_dim
        self.v_dim = d_model // n_heads
        self.causal = causal
        self.rope_enabled = rope

        self.qk_proj = nn.Linear(d_model, n_heads * 2 * qk_dim, bias=False)

        self.out_proj = nn.Linear(n_heads * self.v_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        if self.rope_enabled:
            self.rope = RoPE(qk_dim)

        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        H = self.n_heads

        q, k = self.qk_proj(x).view(B, L, H, 2 * self.qk_dim).transpose(1, 2).chunk(2, -1)

        v = x.view(B, L, H, self.v_dim).transpose(1, 2)

        if self.rope_enabled:
            q, k = self.rope(q, k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=self.causal
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, H * self.v_dim)

        return self.out_proj(attn_out)


class FFMultiheadAttentionMixerNOV(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False, dropout=0.05):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self.dropout = nn.Dropout(dropout)

        self.kv_cache = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        q = self.q(x).reshape(B, L, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def generate(self, x: torch.Tensor):
        B, L, D = x.shape

        qk = self.qk(x).reshape(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)
        v = x.view(B, L, 1, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)[0]

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            pos_offset = k_cache.shape[2]
        else:
            pos_offset = 0
            k_cache, v_cache = None, None

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=pos_offset)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)
        use_causal = self.causal and (L > 1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)

        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qk.weight.data.requires_grad = False
        self.out.weight.data.requires_grad = False


class FFFMultiheadAttentionMixer(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False, dropout=0.05):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = LowRankLinear(d_model, d_model, rank=math.isqrt(d_model), bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self.dropout = nn.Dropout(dropout)

        self.kv_cache = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        q = self.q(x).reshape(B, L, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def generate(self, x: torch.Tensor):
        B, L, D = x.shape

        qk = self.qk(x).reshape(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)
        v = x.view(B, L, 1, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)[0]

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            pos_offset = k_cache.shape[2]
        else:
            pos_offset = 0
            k_cache, v_cache = None, None

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=pos_offset)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        self.kv_cache = (k, v)
        use_causal = self.causal and (L > 1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)

        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qk.weight.data.requires_grad = False
        self.out.weight.data.requires_grad = False


class NSAttention(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d + L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 pos_emb: nn.Module = None, dropout: float = 0.0, qk_norm=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        self.pos_emb = pos_emb
        self.kv_cache = None
        self.qk_norm = qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)
        kv = k.transpose(-1, -2) @ v
        return q @ kv.transpose(-1, -2)


# MODDED NANO GPT

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # FUSED QKV: Output dimension is 3 * dim
        self.c_qkv = CastedLinear(dim, 3 * dim, bias=False)

        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        # 1. Single fused linear projection
        # Shape: (bsz, seqlen, 3 * dim)
        qkv = self.c_qkv(x)

        # 2. Reshape and slice into Q, K, V
        # Reshape to: (bsz, seqlen, 3, num_heads, head_dim)
        qkv = qkv.reshape(bsz, seqlen, 3, self.num_heads, self.head_dim)
        # Permute to: (3, bsz, num_heads, seqlen, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Unbind separates the first dimension (the '3') into a tuple
        q, k, v = qkv.unbind(0)

        # 3. Apply norms
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # 4. RoPE
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 5. Q-gain scaling
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        # 6. Attention
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=False,  # Standard MHA
        )

        # 7. Final projection
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)
