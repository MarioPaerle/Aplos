#!/bin/bash
#SBATCH --job-name=grpo_compile
#SBATCH --account=IscrC_YENDRI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/leonardo_work/IscrC_YENDRI/gcirillo/degree/Aplos_study/bench_compile_%j.out

set -uo pipefail
module load python/3.11 2>/dev/null || true
REPO=/leonardo_work/IscrC_YENDRI/gcirillo/degree/Aplos
export PYTHONPATH="/leonardo_work/IscrC_YENDRI/gcirillo/qwen35_packages:$REPO"
export HF_HOME=$WORK/.cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=$WORK/.cache/torchinductor
echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
python3 "$REPO/tests/bench_compile.py"
echo "=== done ==="
