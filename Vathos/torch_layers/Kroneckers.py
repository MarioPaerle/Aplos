import torch
from Vathos.blocks import *
from Vathos.functions import toeplitz_init


def _kronecker_batch_matmul_einsum(A: torch.Tensor, B: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Compute (A kron B) @ X efficiently using reshapes (O(n^3) with n = sqrt(L) per feature,
    much cheaper than materializing the full kron matrix).
    Inputs:
      - A: [n, n]
      - B: [n, n]
      - X: [B, L, d] with L == n*n
    Returns:
      - Y: [B, L, d] equal to (A ⊗ B) @ X (matrix multiply on the L dimension for each batch/feature)
    Note:
      This implementation assumes X is laid out in row-major order (PyTorch default).
      We exploit the identity (with this ordering):
        vec_row(Y_mat) == (A ⊗ B) vec_row(X_mat)
      which is implemented by reshaping X -> [B, n, n, d] and computing Y_mat = A @ X_mat @ B.T,
      then flattening back to [B, L, d].
    """

    n = A.shape[0]

    Bbatch, L, d = X.shape

    Xm = X.view(Bbatch, n, n, d)  # b, i, j, f

    Y = torch.einsum('ij,bjkf,lk->bilf', A, Xm, B)

    return Y.reshape(Bbatch, L, d)


def _kronecker_batch_matmul_diagonal_einsum(A: torch.Tensor, B: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Compute (A kron B) @ X efficiently using reshapes (O(n^3) with n = sqrt(L) per feature,
    much cheaper than materializing the full kron matrix).
    Inputs:
      - A: [n, n]
      - B: [n, n]
      - X: [B, L, d] with L == n*n
    Returns:
      - Y: [B, L, d] equal to (A ⊗ B) @ X (matrix multiply on the L dimension for each batch/feature)
    Note:
      This implementation assumes X is laid out in row-major order (PyTorch default).
      We exploit the identity (with this ordering):
        vec_row(Y_mat) == (A ⊗ B) vec_row(X_mat)
      which is implemented by reshaping X -> [B, n, n, d] and computing Y_mat = A @ X_mat @ B.T,
      then flattening back to [B, L, d].
    """

    n = A.shape[0]

    Bbatch, L, d = X.shape

    Xm = X.view(Bbatch, n, n, d)  # b, i, j, f

    Y = torch.einsum('i,bikf,lk->bilf', A, Xm, B)

    return Y.reshape(Bbatch, L, d)


class LSqrtPadder(Layer):
    __name__ = "LPadder"
    __complexity__ = "O(k d)"
    __piped__ = True

    def __init__(self, element=0):
        super(LSqrtPadder, self).__init__()
        self.element = element

    def forward(self, x, pipe):
        n = int((x.shape[1] ** 0.5) + 0.999999)
        pipe['original_L'] = x.shape[1]
        return F.pad(x, (0, 0, 0, n ** 2 - x.shape[1]), mode="constant", value=self.element)


class LSqrtUnPadder(Layer):
    __name__ = "LUnPadder"
    __complexity__ = "O(k d)"
    __piped__ = True

    def __init__(self, right=0, left=0):
        super().__init__()
        self.right = right
        self.left = left

    def forward(self, x, pipe):
        L = pipe['original_L']
        return x[:, :L, :]


class SqPadEmbedder(Layer):
    __name__ = "SqPadEmbedder"

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padder = LSqrtPadder()
        self.embedder = Embedder(vocab_size, d_model)

    def forward(self, x):
        x = self.embedder(x)
        x = self.padder(x)
        return x


class SqPadUnembedder(Layer):
    __name__ = "SqPadEmbedder"

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.unpadder = LSqrtUnPadder()
        self.unembedder = Embedder(vocab_size, d_model)

    def forward(self, x):
        x = self.unpadder(x)
        x = self.unembedder(x)
        return x


kronecker_batch_matmul = _kronecker_batch_matmul_einsum
kronecker_batch_matmul_diagonal = _kronecker_batch_matmul_diagonal_einsum


class KroneckerMixer1(Layer):
    __name__ = "KroneckerMixer1"
    __complexity__ = 'O(sqrt(L) L)'

    def __init__(self, d_model, max_len):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        nmax = int((max_len ** 0.5) + 1)
        self.nmax = nmax

        self.As = nn.Parameter(torch.randn(nmax, nmax) * 0.1)
        self.Bs = nn.Parameter(torch.randn(nmax, nmax) * 0.1)
        self.Aps = nn.Parameter(torch.randn(nmax) * 0.1)
        self.Bps = nn.Parameter(torch.randn(nmax, nmax) * 0.1)

    def _mask_params(self, A, Bp):
        A = torch.tril(A, diagonal=-1)
        Bp = torch.tril(Bp)
        return A, Bp

    def single_level(self, X, A, B, Ap, Bp):
        Y = _kronecker_batch_matmul_einsum(A, B, X) + _kronecker_batch_matmul_diagonal_einsum(Ap, Bp, X)
        return Y

    def forward(self, X):
        B, L, d = X.shape
        Y = X
        n = int((L ** 0.5) + 0.999999)
        A = self.As[:n, :n]
        B = self.Bs[:n, :n]
        Ap = self.Aps[:n]
        Bp = self.Bps[:n, :n]
        A, Bp = self._mask_params(A, Bp)

        Y = self.single_level(Y, A, B, Ap, Bp) + Y

        return Y  # [:, :L, :]


class KroneckerMixer2(Layer):
    __name__ = "KroneckerMixer2"
    __complexity__ = 'O(sqrt(L) L d)'

    def __init__(self, d_model, max_len, k=3):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        nmax = int((max_len ** 0.5) + 1)
        self.nmax = nmax

        self.As = nn.Parameter(torch.randn(nmax, nmax) * 0.02)
        self.Bs = nn.Parameter(torch.randn(nmax, nmax) * 0.02)
        self.Aps = nn.Parameter(torch.randn(nmax) * 0.02)
        self.Bps = nn.Parameter(torch.randn(nmax, nmax) * 0.02)
        self.conv = CausalConv1d(k=k, d_model=self.d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def _mask_params(self, A, Bp):
        A = torch.tril(A, diagonal=-1)
        Bp = torch.tril(Bp)
        return A, Bp

    def single_level(self, X, A, B, Ap, Bp):
        Y = _kronecker_batch_matmul_einsum(A, B, X) + _kronecker_batch_matmul_diagonal_einsum(Ap, Bp, X)
        return Y

    def forward(self, X):
        B, L, d = X.shape
        n = int((L ** 0.5) + 0.999999)
        B = self.Bs[:n, :n]
        Ap = self.Aps[:n]
        A, Bp = self._mask_params(self.As[:n, :n], self.Bps[:n, :n])
        X = F.gelu(self.conv(X))
        g = torch.sigmoid(self.gate(X))
        X = self.single_level(X, A, B, Ap, Bp) * g + (1 - g) * X
        X = self.dropout(X)

        return X  # [:, :L, :]


import torch
import torch.nn as nn
import torch.nn.functional as F


# Assuming your helper functions are defined elsewhere:
# _kronecker_batch_matmul_einsum, _kronecker_batch_matmul_diagonal_einsum

class KroneckerMixer3(nn.Module):
    __name__ = "KroneckerMixer3"
    __complexity__ = 'O(sqrt(L) L d)'

    def __init__(self, d_model, max_len, k=3):
        super().__init__()
        self.d_model = d_model

        self.nmax = int((max_len ** 0.5) + 1)

        self.As = nn.Parameter(torch.randn(self.nmax, self.nmax) * 0.02)
        self.Bs = nn.Parameter(torch.randn(self.nmax, self.nmax) * 0.02)

        self.Aps = nn.Parameter(torch.randn(self.nmax) * 0.02)
        self.Bps = nn.Parameter(torch.randn(self.nmax, self.nmax) * 0.02)

        self.coarse_q = nn.Linear(d_model, d_model // 4, bias=False)
        self.coarse_k = nn.Linear(d_model, d_model // 4, bias=False)
        self.coarse_gate_scale = nn.Parameter(torch.tensor(1.0))

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model)
        self.out_gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def _mask_params(self, A, Bp):
        A = torch.tril(A, diagonal=-1)
        Bp = torch.tril(Bp)
        return A, Bp

    def single_level(self, X, A, B, Ap, Bp):
        Y = _kronecker_batch_matmul_einsum(A, B, X) + \
            _kronecker_batch_matmul_diagonal_einsum(Ap, Bp, X)
        return Y

    def compute_dynamic_A(self, X, n, A_static):
        """
        Computes a dynamic modulation for the coarse matrix A.
        X: [B, L, d]
        """
        B_size, L, d = X.shape

        X_pad = X

        X_coarse = X_pad.view(B_size, n, n, d).mean(dim=2)
        q = self.coarse_q(X_coarse)
        k = self.coarse_k(X_coarse)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.coarse_gate_scale

        dynamic_gate = torch.sigmoid(attn_scores)

        A_dynamic = A_static.unsqueeze(0) * dynamic_gate

        return A_dynamic

    def forward(self, X):
        B_size, L, d = X.shape
        n = int((L ** 0.5) + 0.999999)

        Bs = self.Bs[:n, :n]
        Ap = self.Aps[:n]
        A_static, Bp = self._mask_params(self.As[:n, :n], self.Bps[:n, :n])

        X_conv = self.conv(X.transpose(1, 2)).transpose(1, 2)
        X = F.gelu(X_conv)

        A_dynamic = self.compute_dynamic_A(X, n, A_static)

        Y = self.single_level(X, A_dynamic, Bs, Ap, Bp)

        g = torch.sigmoid(self.out_gate(X))
        out = Y * g + (1 - g) * X
        out = self.dropout(out)

        return out


class KOBRA1(Layer):
    __name__ = "KroneckerMixer2"
    __complexity__ = 'O(sqrt(L) L d)'

    def __init__(self, d_model, max_len, k=3):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        nmax = int((max_len ** 0.5) + 1)
        self.nmax = nmax

        self.As = nn.Parameter(torch.randn(nmax, nmax) * 0.02)
        self.Bs = nn.Parameter(torch.randn(nmax, nmax) * 0.02)
        self.Aps = nn.Parameter(torch.randn(nmax) * 0.02)
        self.Bps = nn.Parameter(torch.randn(nmax, nmax) * 0.02)
        self.conv = CausalConv1d(k=k, d_model=self.d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.qk = nn.Linear(d_model, 2 * d_model)
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(0.1)

    def _mask_params(self, A, Bp):
        A = torch.tril(A, diagonal=-1)
        Bp = torch.tril(Bp)
        return A, Bp

    def single_level(self, X, A, B, Ap, Bp):
        Y = _kronecker_batch_matmul_einsum(A, B, X) + _kronecker_batch_matmul_diagonal_einsum(Ap, Bp, X)
        return Y

    def forward(self, X):
        b, L, d = X.shape
        q, k = self.qk(X).chunk(2, dim=-1)
        n = int((L ** 0.5) + 0.999999)
        B = self.Bs[:n, :n]
        Ap = self.Aps[:n]
        A, Bp = self._mask_params(self.As[:n, :n], self.Bps[:n, :n])

        X = F.gelu(self.conv(X))
        g = torch.sigmoid(self.gate(X))
        X = self.single_level(X, A, B, Ap, Bp) * g + (1 - g) * X

        q = q.view(b * L, d).unsqueeze(1)
        k = k.view(b * L, d).unsqueeze(-1)

        qk = (q @ k) * self.scale
        qk = qk.squeeze(-1).view(b, L, 1)

        X = X * qk
        X = self.dropout(X)

        return X  # [:, :L, :]

class QKVKroneckerMixer(Layer):
    __name__ = "MultiheadAttentionMixer"
    __complexity__ = "O(L sqrt(L) d^2)"

    def __init__(self, d_model: int, causal: bool, rope=False, max_len=2116):
        super().__init__()
        self.causal = causal
        self.max_len = max_len
        self.d_model = d_model
        nmax = int((max_len ** 0.5) + 1)
        self.nmax = nmax

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        if rope:
            self.rope = RoPE(self.head_dim)
        else:
            self.rope = None

        self.As = nn.Parameter(torch.empty(nmax, nmax), requires_grad=False)
        self.Bs = nn.Parameter(torch.randn(nmax, nmax) * 0.02)
        self.Aps = nn.Parameter(torch.randn(nmax) * 0.02)
        self.Bps = nn.Parameter(torch.randn(nmax, nmax) * 0.02)

        toeplitz_init(self.As, 0.95)

    def _mask_params(self, A, Bp):
        A = torch.tril(A, diagonal=-1)
        Bp = torch.tril(Bp)
        return A, Bp

    def single_level(self, X, A, B, Ap, Bp):
        Y = _kronecker_batch_matmul_einsum(A, B, X) + _kronecker_batch_matmul_diagonal_einsum(Ap, Bp, X)
        return Y

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, -1)
        B, L, d = x.shape
        n = int((L ** 0.5) + 0.999999)
        B = self.Bs[:n, :n]
        Ap = self.Aps[:n]
        A, Bp = self._mask_params(self.As[:n, :n], self.Bps[:n, :n])
        X = self.single_level((F.elu(k) + 1) * v, A, B, Ap, Bp) * (F.elu(q) + 1)

        return self.out(X)


if __name__ == '__main__':
    X = torch.randint(0, 10, (2, 1024))

    hybrid = HybridAttentionBlock1d
    attn_params = {'n_heads': 8, 'causal': True}
    kronecker = KroneckerMixer1
    kronecker_params = {'max_len': 2116}
    hybrid_params = {
        'sec_mixer': kronecker,
        'sec_params': kronecker_params,
        'attn_params': attn_params,
        'channel_mixer': DLPGelu,
        'channel_params': {'expand': 4},
        'n_attn': 1,
        'n_sec': 3

    }

    model = SequenceModel(
        vocab_size=10,
        d_model=128,
        n_layers=5,
        max_len=2116,
        pos_encoder=True,
        embedder=EasyEmbedder,
        unembedder=UnbiasedLinear,
        channel_mixer=MLP,
        channel_args={'expand': 2, 'activation': SwiGLU, 'depth': 2},
        rope=False,
        # pad='sqrt'
        spatial_mixer=KOBRA1,
        spatial_args={'max_len': 2116, 'k': 8}
    )
    model.summary()
    out = model(X)
    model.profile(plot=True, plot_level=5, avg=True)
