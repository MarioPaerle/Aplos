from Vathos import *


d_models = [64 for i in range(4)]
m_dims = [128, 64, 32, 16]
attn = Builder(GQA, n_heads=4, n_kv_heads=2)
model = ModdedFormer(
    vocab_size=100,
    embed_dim=64,
    d_models=d_models,
    spatials=[attn for _ in range(len(d_models))],
    M_dims=m_dims
)
model.summary()

out = model(
    torch.randint(0, 99, (2, 128))
)
print(out.shape)

model.profile()

from ModelLens import run_inspector

run_inspector(model)


