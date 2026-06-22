#!/bin/bash
#SBATCH --job-name=aplos_train
#SBATCH --account=IscrC_YENDRI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_YENDRI/paerle/Aplos/bench_outputs/train_A100_full_%j.out

set -euo pipefail

if [[ "${PICO_TRAIN_USE_GUIDO_ENV:-0}" == "1" ]]; then
  module purge 2>/dev/null || true
  module load profile/deeplrn python/3.11.7 cuda/12.2
  source /leonardo_work/IscrC_YENDRI/paerle/PiCO/envs/pico2/bin/activate
else
  module load python/3.11 2>/dev/null || true
fi

REPO=/leonardo_work/IscrC_YENDRI/paerle/Aplos
TORCH_PKGS=/leonardo_work/IscrC_YENDRI/gcirillo/qwen35_packages

USE_QWEN_PKGS="${PICO_TRAIN_USE_QWEN_PKGS:-1}"
if [[ "${PICO_TRAIN_USE_GUIDO_ENV:-0}" == "1" ]]; then
  USE_QWEN_PKGS="${PICO_TRAIN_USE_QWEN_PKGS:-0}"
fi
if [[ "$USE_QWEN_PKGS" == "1" ]]; then
  export PYTHONPATH="$TORCH_PKGS:$REPO${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

CACHE_ROOT="${PICO_TRAIN_CACHE_ROOT:-$REPO/.runtime_cache/train_A100_${SLURM_JOB_ID:-manual}}"
export HF_HOME="$CACHE_ROOT/.cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/.cache/torchinductor"
mkdir -p "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$REPO/bench_outputs"

if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
else
  PY=python3
fi

CORPUS_ROOT="${PICO_TRAIN_CORPUS_ROOT:-/leonardo_scratch/fast/IscrC_YENDRI/mprignan/corpus_v2}"
MIXTURE_PRESET="${PICO_TRAIN_MIXTURE_PRESET:-default}"
PRESET="${PICO_TRAIN_PRESET:-guido200m}"
ATTENTION="${PICO_TRAIN_ATTENTION:-gqa_gated}"
CHANNEL="${PICO_TRAIN_CHANNEL:-torch_lrelu2}"
LOSS="${PICO_TRAIN_LOSS:-logits}"
OPTIMIZER="${PICO_TRAIN_OPTIMIZER:-muon}"
N_KV_HEADS="${PICO_TRAIN_N_KV_HEADS:-4}"
SEQ_LEN="${PICO_TRAIN_SEQ_LEN:-2048}"
BATCH_SIZE="${PICO_TRAIN_BATCH_SIZE:-16}"
GRAD_ACCUM="${PICO_TRAIN_GRAD_ACCUM:-1}"
TRAIN_STEPS="${PICO_TRAIN_STEPS:-160}"
LOG_EVERY="${PICO_TRAIN_LOG_EVERY:-10}"
SAVE_EVERY="${PICO_TRAIN_SAVE_EVERY:-0}"
COMPILE_MODE="${PICO_TRAIN_COMPILE_MODE:-max-autotune-no-cudagraphs}"
COMPILE_MODEL="${PICO_TRAIN_COMPILE_MODEL:-1}"
XSA_LAST_N="${PICO_TRAIN_XSA_LAST_N:-0}"
GENERATE_SMOKE_TOKENS="${PICO_TRAIN_GENERATE_SMOKE_TOKENS:-0}"
FFN_MULTIPLIER="${PICO_TRAIN_FFN_MULTIPLIER:-4}"
GATE_INPUT_DIM="${PICO_TRAIN_GATE_INPUT_DIM:-12}"

COMPILE_ARGS=()
if [[ "$COMPILE_MODEL" == "1" ]]; then
  COMPILE_ARGS+=(--compile-model)
else
  COMPILE_ARGS+=(--no-compile-model)
fi

echo "=== node $(hostname) ==="
echo "=== repo $REPO ==="
echo "=== python $($PY --version 2>&1) ==="
echo "=== job $SLURM_JOB_ID ==="
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

echo "=== full train benchmark ==="
echo "corpus_root=$CORPUS_ROOT mixture=$MIXTURE_PRESET preset=$PRESET"
echo "attention=$ATTENTION xsa_last_n=$XSA_LAST_N channel=$CHANNEL loss=$LOSS optimizer=$OPTIMIZER"
echo "seq_len=$SEQ_LEN batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM steps=$TRAIN_STEPS n_kv_heads=$N_KV_HEADS ffn_multiplier=$FFN_MULTIPLIER gate_input_dim=$GATE_INPUT_DIM compile=$COMPILE_MODEL mode=$COMPILE_MODE"

$PY "$REPO/train_A100.py" \
  --corpus-root "$CORPUS_ROOT" \
  --mixture-preset "$MIXTURE_PRESET" \
  --preset "$PRESET" \
  --n-kv-heads "$N_KV_HEADS" \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --steps "$TRAIN_STEPS" \
  --log-every "$LOG_EVERY" \
  --save-every "$SAVE_EVERY" \
  --attention "$ATTENTION" \
  --xsa-last-n "$XSA_LAST_N" \
  --channel "$CHANNEL" \
  --ffn-multiplier "$FFN_MULTIPLIER" \
  --gate-input-dim "$GATE_INPUT_DIM" \
  --loss "$LOSS" \
  --optimizer "$OPTIMIZER" \
  --compile-mode "$COMPILE_MODE" \
  --generate-smoke-tokens "$GENERATE_SMOKE_TOKENS" \
  --ckpt-dir "$REPO/bench_outputs/train_A100_full_${SLURM_JOB_ID}_ckpts" \
  --loss-trace-csv "$REPO/bench_outputs/train_A100_full_${SLURM_JOB_ID}.csv" \
  "${COMPILE_ARGS[@]}"

echo "=== done ==="
