#!/bin/bash
#SBATCH --job-name=grpo_bench
#SBATCH --account=IscrC_YENDRI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/leonardo_work/IscrC_YENDRI/gcirillo/degree/Aplos_study/bench_grpo_%j.out

set -euo pipefail
module load python/3.11 2>/dev/null || true

REPO=/leonardo_work/IscrC_YENDRI/gcirillo/degree/Aplos
TORCH=/leonardo_work/IscrC_YENDRI/gcirillo/qwen35_packages
export PYTHONPATH="$TORCH:$REPO"
export HF_HOME=$WORK/.cache/huggingface

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=== correctness check (must be 6/6) ==="
python3 -W ignore "$REPO/tests/test_grpo_generate.py" 2>&1 | grep -v "FLAG LV" | tail -10
echo "=== throughput benchmark ==="
python3 "$REPO/tests/bench_grpo_generate.py"
echo "=== done ==="
