#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=debug

# Smoke test of train.py: 2 short epochs (with evals and a checkpoint), then a resume for a 3rd epoch, W&B offline.
# Exits non-zero if either run fails, so a real run submitted with --dependency=afterok:<this job> never starts.
#
#   sbatch --job-name=t2i-smoke --output=logs/slurm/%x_%j.log slurm/smoke.sh configs/exp1_baseline.yaml

set -uo pipefail
CONFIG="$1"
OUT=/mnt/data/ishant/T2I-ReID/logs/smoke
rm -rf "$OUT"
Q="log.max_iters_per_epoch=30 log.every=10 log.eval_every=1 log.eval_max_queries=200 wandb.mode=offline output_dir=$OUT"

bash /mnt/data/ishant/T2I-ReID/slurm/train.sh "$CONFIG" $Q optim.epochs=2 resume=false || exit 1
bash /mnt/data/ishant/T2I-ReID/slurm/train.sh "$CONFIG" $Q optim.epochs=3 resume=true || exit 1
grep -q "Resumed from" "$OUT/train_log.txt" || { echo "resume did not happen"; exit 1; }
rm -f "$OUT"/*.pth
echo "SMOKE OK"
