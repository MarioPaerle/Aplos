# A100 PiCOFormer Training

`train_A100.py` is the Guido-style pretraining entrypoint backed by the newer
Aplos PiCOFormer library. It is meant for fixed-shape A100 runs: bf16 autocast,
Torch compile, Flash-SDPA, Muon for hidden matrices, Adam for embeddings/scalars,
and direct mmap reads from the existing Guido shard format.

## Fast Single-GPU Run

From the Aplos repo:

```bash
export PYTHONPATH=/leonardo_work/IscrC_YENDRI/gcirillo/qwen35_packages:$PWD
python train_A100.py \
  --corpus-root /leonardo_scratch/fast/IscrC_YENDRI/mprignan/corpus_v2 \
  --mixture-preset default \
  --preset guido200m \
  --batch-size 16 \
  --seq-len 2048 \
  --steps 1000 \
  --attention gqa_gated \
  --channel torch_lrelu2 \
  --compile-model
```

The qwen package path is just the newer Torch/Triton stack we benchmarked on
Leonardo, not a Qwen model dependency.

## Main Architecture Knobs

- `--attention gqa_gated`: default for next Guido. Uses compact GQA KV cache plus
  sparse per-head gates. This now supports static-cache GRPO and graph decode.
- `--attention gated_mha`: full MHA gated attention. More KV memory, good baseline.
- `--attention xsa_mha`: Exclusive Self Attention in every layer.
- `--xsa-last-n N`: keep the chosen attention in early layers and use MHA-XSA in
  the last `N` layers.
- `--channel torch_lrelu2`: LeakyReLU squared UDLP, fastest current default.
- `--channel torch_leaky_reglu2`: LeakyReGLU squared GLU-style FFN. More params and
  compute, useful A/B against Guido's experimental GLU branch.
- `--channel triton_lrelu2_a100`: available but currently slower than Torch on the
  measured quick/200M shapes; keep benchmarking before making it default.

## Loss Mode

- `--loss logits`: materializes full model logits and uses PyTorch CE.
- `--loss cce`: requires fused `cut_cross_entropy` and fails loudly if the active
  Torch environment does not provide it.
- `--loss cce_auto`: uses fused `cut_cross_entropy` when installed and otherwise
  falls back to hidden-state torch CE.
- `--loss hidden_logits`: explicit hidden-state torch fallback. Useful for
  debugging portability, not the preferred production path.

For the 98.68M `v3_noloops` batch-32 benchmark shape, fused CCE was both faster
and much lower memory than full logits CE (`144k tok/s`, `34.4 GB` vs `134k tok/s`,
`58.9 GB`). Use `cce` when the environment has the package installed; rebuild it
for the newer torch stack before large production runs.

## FineWeb-Edu Only

```bash
python train_A100.py \
  --mixture-preset fineweb_only \
  --corpus-root /leonardo_scratch/fast/IscrC_YENDRI/mprignan/corpus_v2 \
  --steps 200 \
  --batch-size 16
```

For the v4 FineWeb-Edu shards:

```bash
python train_A100.py \
  --mixture-preset fineweb_v4_only \
  --corpus-root /leonardo_scratch/large/userexternal/mprignan/corpus_v4 \
  --steps 200 \
  --batch-size 16
```

## SLURM Single A100

```bash
#!/bin/bash
#SBATCH --job-name=aplos_train_a100
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=bench_outputs/train_A100_%j.out

cd /leonardo_work/IscrC_YENDRI/paerle/Aplos
export PYTHONPATH=/leonardo_work/IscrC_YENDRI/gcirillo/qwen35_packages:$PWD
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python train_A100.py \
  --preset guido200m \
  --batch-size 16 \
  --steps 1000 \
  --attention gqa_gated \
  --channel torch_lrelu2 \
  --generate-smoke-tokens 64 \
  --compile-decode-smoke
```

For 4 GPU training later, use:

```bash
torchrun --standalone --nproc-per-node=4 train_A100.py --preset guido200m
```

The default request here stays single GPU because it skips the long Leonardo
multi-GPU queue and matches the benchmark target.

## Outputs

- Checkpoints: `ckpts/train_A100/latest.pt` and periodic `step_*.pt`.
- Loss trace: `logs/train_A100_loss_trace.csv`.
- Each checkpoint stores the Aplos `PiCOFormerConfig`, CLI args, model state, and
  optimizer states.

## Minimal Diff For Guido Agents

Use Aplos as an importable library and move architecture selection into flags:

```python
from Vathos.picoformer import PiCOFormerConfig, build_picoformer

cfg = PiCOFormerConfig(
    vocab_size=32768,
    d_model=768,
    n_layers=24,
    n_heads=12,
    n_kv_heads=3,
    attention="gqa_gated",
    channel="torch_lrelu2",
    max_seq_len=2048,
    smear_gate=True,
)
model = build_picoformer(cfg)
```

The training loop can keep the same data mixture, Muon split, bf16 autocast, and
Torch compile strategy while replacing the old inline model definition.
