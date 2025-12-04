from Vathos.blocks import *


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
    def __init__(self, d_model, max_len):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        nmax = int((max_len ** 0.5) + 1)
        self.nmax = nmax

        self.As = nn.Parameter(torch.randn(nmax, nmax) * 0.02)
        self.Bs = nn.Parameter(torch.randn(nmax, nmax) * 0.02)
        self.Aps = nn.Parameter(torch.randn(nmax) * 0.02)
        self.Bps = nn.Parameter(torch.randn(nmax, nmax) * 0.02)

    def _mask_params(self, A, Bp):
        A = torch.tril(A, diagonal=-1)
        Bp = torch.tril(Bp)
        return A, Bp

    def single_level(self, X, A, B, Ap, Bp):
        Y1 = _kronecker_batch_matmul_einsum(A, B, X)
        Y2 = _kronecker_batch_matmul_diagonal_einsum(Ap, Bp, X)

        Y = Y1 + Y2
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

        Y = self.single_level(Y, A, B, Ap, Bp)  # + Y

        return Y[:, :L, :]


if __name__ == '__main__':
    X = torch.randint(0, 10, (2, 1024))

    model = SequenceModel(
        vocab_size=10,
        d_model=512,
        n_layers=4,
        max_len=1024,
        pos_encoder=True,
        embedder=Embedder,
        unembedder=UnbiasedLinear,
        channel_mixer=MLP,
        channel_args={"expand": 2, "depth": 2, "activation": nn.GELU},
        spatial_mixer=MultiheadAttentionMixer,
        spatial_args={'n_heads': 8, 'causal': True}
    )
    model.summary()
    out = model(X)
    model.profile(plot=True, plot_level=1)
