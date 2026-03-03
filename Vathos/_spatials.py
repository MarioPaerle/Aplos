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


class RoPE(nn.Module):
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

            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                          start_pos: int = 0) -> torch.Tensor:
        """Apply rotary embeddings starting from start_pos"""
        seq_len = x.shape[-2]

        cos = cos[start_pos:start_pos + seq_len]
        sin = sin[start_pos:start_pos + seq_len]

        shape = [1] * x.ndim
        shape[-2] = seq_len
        shape[-1] = self.dim

        cos = cos.view(*shape)
        sin = sin.view(*shape)

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


class MultiheadAttentionMixer(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False, dropout=0.00):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self.dropout = nn.Dropout(dropout)

        self.kv_cache = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=0)
        attn = attn.transpose(1, 2).reshape(B, L, D)
        return self.out(attn)

    def generate(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # 1. Handle Cache Offset
        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache
            pos_offset = k_cache.shape[2]
        else:
            pos_offset = 0
            k_cache, v_cache = None, None

        # 2. Apply RoPE (Rotates the *new* q and k relative to history)
        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=pos_offset)

        # 3. Update Cache
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Save updated cache
        self.kv_cache = (k, v)

        # 4. Attention with Dynamic Masking
        # If L > 1, we are processing a prompt (Prefill), so we MUST be causal.
        # If L == 1, we are generating a token step-by-step, attending to the whole history.
        use_causal = self.causal and (L > 1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)

        attn = attn.transpose(1, 2).reshape(B, L, D)
        attn = self.dropout(attn)
        return self.out(attn)

    def clear_cache(self):
        self.kv_cache = None

    def finetune(self):
        self.qkv.weight.data.requires_grad = False
        self.out.weight.data.requires_grad = False


class MultiheadAttentionMixerNOV(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal: bool, rope=False, dropout=0.00):
        super().__init__()
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qk = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self.dropout = nn.Dropout(dropout)

        self.kv_cache = None

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qk = self.qk(x).reshape(B, L, 2, self.n_heads, self.head_dim)
        q, k = qk.permute(2, 0, 3, 1, 4)
        v = x.view(B, L, 1, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)[0]

        if self.rope is not None:
            q, k = self.rope(q, k, start_pos=0)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        attn = attn.transpose(1, 2).reshape(B, L, D)
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



class GroupedQueryAttention(Layer):
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
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RoPE(self.head_dim) if rope else None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
            nn.init.constant_(self.kv_proj.bias, 0)
            nn.init.constant_(self.o_proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x)
        kv = self.kv_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        kv = kv.view(B, T, self.num_kv_heads, 2, self.head_dim)
        k, v = kv.unbind(dim=3)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE BEFORE expansion to save compute
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
        return self.o_proj(attn_output)


class GroupedQueryAttentionNOV(Layer):
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

        self.rope = RoPE(self.head_dim) if rope else None

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


class CausalMultiheadAttentionMixer(Layer):
    __name__ = "CausalMultiheadAttentionMixer"
    __complexity__ = "O(L^2 d +  L d^2)"

    def __init__(self, d_model: int, n_heads: int, causal=True, rope=False, dropout=0.1):
        super().__init__()
        assert causal, \
            ("CausalMultiheadAttentionMixLayer only supports causal=True, "
             "if you meant to create a non Causal Attention use the MultiheadAttentionMixer")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0
        )

        attn = attn.transpose(1, 2).contiguous().reshape(B, L, D)

        return self.out(attn)


class CausalMultiheadAttentionMixer2(nn.Module):
    __name__ = "CausalMultiheadAttentionMixer"

    def __init__(self, d_model: int, n_heads: int, causal=True, rope=False, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.rope = RoPE(self.head_dim) if rope else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x)

        qkv = qkv.view(B, L, 3, self.n_heads, self.head_dim)

        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        """q = self.q_norm(q)
        k = self.k_norm(k)"""

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
