#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --partition=debug

# =============================================================================
# One single-GPU T2I-ReID run (train.py). Usage:
#
#   sbatch --job-name=t2i-exp1 --output=logs/slurm/%x_%j.log slurm/train.sh configs/exp1_baseline.yaml [key=value ...]
#
# Training resumes from <output_dir>/checkpoint_latest.pth when it exists (resume=true in the config), so the
# same command can be resubmitted after a crash or timeout.
#
# The code that runs is the checkout sbatch is called from (each experiment branch has its own git worktree), not
# the editable install of t2i_reid, which points at /mnt/data/ishant/T2I-ReID.
# =============================================================================

set -euo pipefail

CONFIG="$1"; shift
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/mnt/data/ishant/T2I-ReID}}"

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# --- GPU pinning (same scheme as TransReID/slurm/kd/train_kd.sh) ---
# gres.conf on this node doesn't restrict which cards a job sees; everyone runs on the physical GPU whose index
# equals their Slurm slot, so do the same. Slots in AVOID_GPUS are used only if the job has no other slot.
GPU_UUIDS=(GPU-eb46644a8c5d4e1f GPU-d15214a684825ef4 GPU-671a6b2f23cd8ce6 GPU-6796b15664f9f25f
           GPU-17b7be1aa9004ca8 GPU-a5ff767c22fb3096 GPU-428ca7e1eef4565b GPU-9a4d4be11adce22a)

AVOID_GPUS="${AVOID_GPUS:-6}"
JOB_GPUS="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}"

if [ -z "${PIN_GPU:-}" ]; then
    preferred=(); avoided=()
    for part in ${JOB_GPUS//,/ }; do
        for i in $(seq ${part%-*} ${part#*-}); do
            if [[ ",$AVOID_GPUS," == *",$i,"* ]]; then avoided+=("$i"); else preferred+=("$i"); fi
        done
    done
    slots=("${preferred[@]}" "${avoided[@]}")
    PIN_GPU="${slots[0]:?no Slurm GPU slot found (SLURM_JOB_GPUS / SLURM_STEP_GPUS unset)}"
fi

# --- Environment ---
module purge
module load python/3.11
module load rocm/6.2.4
source "/apps/software/anaconda3/etc/profile.d/conda.sh"
conda activate TransReID
cd "$PROJECT_DIR"

# Import t2i_reid from PROJECT_DIR: a directory on PYTHONPATH wins over the editable-install finder.
SHIM="$(mktemp -d)"; trap 'rm -rf "$SHIM"' EXIT
ln -s "$PROJECT_DIR" "$SHIM/t2i_reid"
export PYTHONPATH="$SHIM${PYTHONPATH:+:$PYTHONPATH}"

export ROCR_VISIBLE_DEVICES="${GPU_UUIDS[$PIN_GPU]}"
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 GPU_DEVICE_ORDINAL=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "============================================"
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "Config:  $CONFIG $*"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Code:    $PROJECT_DIR ($(git rev-parse --abbrev-ref HEAD))"
echo "Commit:  $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo ' (dirty)')"
echo "GPU:     physical $PIN_GPU ($ROCR_VISIBLE_DEVICES); Slurm slots ${JOB_GPUS:-?}"
python -c "import t2i_reid, os; assert os.path.realpath(t2i_reid.__path__[0]) == os.path.realpath('$PROJECT_DIR'), t2i_reid.__path__"
python -c "import torch; assert torch.cuda.device_count() == 1, torch.cuda.device_count(); print('GPU:', torch.cuda.get_device_name(0))"
echo "============================================"

python train.py --config "$CONFIG" "$@"

echo "Done at $(date)"
