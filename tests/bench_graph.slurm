#!/bin/bash
#SBATCH --job-name=grpo_graph
#SBATCH --account=IscrC_YENDRI
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:28:00
#SBATCH --output=/leonardo_work/IscrC_YENDRI/gcirillo/degree/Aplos_study/bench_graph_%j.out

set -uo pipefail
module load python/3.11 2>/dev/null || true
REPO=/leonardo_work/IscrC_YENDRI/gcirillo/degree/Aplos
export PYTHONPATH="/leonardo_work/IscrC_YENDRI/gcirillo/qwen35_packages:$REPO"
export HF_HOME=$WORK/.cache/huggingface
export TORCHINDUCTOR_CACHE_DIR=$WORK/.cache/torchinductor
echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "=== correctness (expect 9/9) ==="
python3 -W ignore "$REPO/tests/test_grpo_generate.py" 2>&1 | grep -v "FLAG LV" | tail -3
echo "=== graph-decode benchmark ==="
python3 -W ignore "$REPO/tests/bench_graph_decode.py" 2>&1 | grep -vE "Dynamo|recompile|UserWarning|warn_once"
echo "=== done ==="
