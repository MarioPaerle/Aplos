#!/bin/bash
#SBATCH --job-name=pico_a100
#SBATCH --account=IscrC_YENDRI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_YENDRI/paerle/Aplos/bench_outputs/picoformer_a100_%j.out

set -euo pipefail

if [[ "${PICO_BENCH_USE_GUIDO_ENV:-0}" == "1" ]]; then
  module purge 2>/dev/null || true
  module load profile/deeplrn python/3.11.7 cuda/12.2
  source /leonardo_work/IscrC_YENDRI/paerle/PiCO/envs/pico2/bin/activate
else
  module load python/3.11 2>/dev/null || true
fi

REPO=/leonardo_work/IscrC_YENDRI/paerle/Aplos
TORCH_PKGS=/leonardo_work/IscrC_YENDRI/gcirillo/qwen35_packages

USE_QWEN_PKGS="${PICO_BENCH_USE_QWEN_PKGS:-1}"
if [[ "${PICO_BENCH_USE_GUIDO_ENV:-0}" == "1" ]]; then
  USE_QWEN_PKGS="${PICO_BENCH_USE_QWEN_PKGS:-0}"
fi
if [[ "$USE_QWEN_PKGS" == "1" ]]; then
  export PYTHONPATH="$TORCH_PKGS:$REPO${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

CACHE_ROOT="${PICO_CACHE_ROOT:-$REPO/.runtime_cache/pico_${SLURM_JOB_ID:-manual}}"
export HF_HOME="$CACHE_ROOT/.cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/.cache/torchinductor_pico_${SLURM_JOB_ID:-manual}"
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR"

if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
else
  PY=python3
fi

PRESET="${PICO_BENCH_PRESET:-quick}"
CHANNELS="${PICO_BENCH_CHANNELS:-torch_lrelu2,triton_lrelu2_a100}"
TRAIN_ITERS="${PICO_BENCH_TRAIN_ITERS:-4}"
GEN_ITERS="${PICO_BENCH_GEN_ITERS:-3}"
STABILITY_STEPS="${PICO_BENCH_STABILITY_STEPS:-8}"
COMPILE_MODE="${PICO_BENCH_COMPILE_MODE:-default}"
COMPILE_TARGET="${PICO_BENCH_COMPILE_TARGET:-model}"
LOSS_IMPL="${PICO_BENCH_LOSS_IMPL:-logits}"
EXTRA_ARGS=()
for spec in \
  "PICO_BENCH_D_MODEL:--d-model" \
  "PICO_BENCH_N_LAYERS:--n-layers" \
  "PICO_BENCH_N_HEADS:--n-heads" \
  "PICO_BENCH_N_KV_HEADS:--n-kv-heads" \
  "PICO_BENCH_VOCAB_SIZE:--vocab-size" \
  "PICO_BENCH_SEQ_LEN:--seq-len" \
  "PICO_BENCH_TRAIN_BATCH:--train-batch" \
  "PICO_BENCH_PROMPT_LEN:--prompt-len" \
  "PICO_BENCH_MAX_NEW_TOKENS:--max-new-tokens" \
  "PICO_BENCH_GROUP_SIZE:--group-size" \
  "PICO_BENCH_FFN_MULTIPLIER:--ffn-multiplier"; do
  env_name="${spec%%:*}"
  arg_name="${spec#*:}"
  if [[ -n "${!env_name:-}" ]]; then
    EXTRA_ARGS+=("$arg_name" "${!env_name}")
  fi
done
if [[ "${PICO_BENCH_SKIP_COMPILE_TRAIN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-compile-train)
fi
if [[ "${PICO_BENCH_SKIP_GENERATION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-generation)
fi
if [[ "${PICO_BENCH_SKIP_STABILITY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-stability)
fi

echo "=== node $(hostname) ==="
echo "=== repo $REPO ==="
echo "=== python $($PY --version 2>&1) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

echo "=== import check ==="
$PY - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    import triton
    print("triton", triton.__version__)
except Exception as exc:
    print("triton unavailable", repr(exc))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "=== benchmark ==="
$PY "$REPO/benchmarks/bench_picoformer_a100.py" \
  --preset "$PRESET" \
  --channels "$CHANNELS" \
  --dtype bf16 \
  --warmup 1 \
  --train-iters "$TRAIN_ITERS" \
  --gen-iters "$GEN_ITERS" \
  --stability-steps "$STABILITY_STEPS" \
  --compile-mode "$COMPILE_MODE" \
  --compile-target "$COMPILE_TARGET" \
  --loss-impl "$LOSS_IMPL" \
  "${EXTRA_ARGS[@]}"

echo "=== done ==="
