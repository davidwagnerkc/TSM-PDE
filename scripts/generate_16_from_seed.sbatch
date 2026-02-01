#!/usr/bin/env bash
#SBATCH --job-name=tsm-pde-generation
#SBATCH --nodelist=kc-sse-ml-rn03,kc-sse-ml-rn04,kc-sse-ml-rn05
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/home/dwcgt/logs/%j.log
#SBATCH --error=/home/dwcgt/logs/%j.log
#SBATCH --open-mode=append
#SBATCH --export=NONE

set -euo pipefail

SEED=${1:-909}
export TRAINING_DATA_SEED=$SEED

echo "Generating trajectories with TRAINING_DATA_SEED=$TRAINING_DATA_SEED"

OUTPUT_DIR=/scratch/dwcgt/tsm-pde/cfd/models/dns_2048x2048_train_${SEED}
if [ -d "$OUTPUT_DIR" ]; then
  echo "Output directory already exists: $OUTPUT_DIR — skipping."
  exit 0
fi

mkdir -p /scratch/dwcgt/tsm-pde

srun --cpu-bind=none --export=ALL bash -lc "
  set -euo pipefail
  export TRAINING_DATA_SEED=$SEED
  NP_RUNTIME=bwrap nix develop \"\$HOME/git/dotfiles/\" --command bash -lic 'deactivate && conda activate cfd && cd /home/dwcgt/git/TSM-PDE/ && ./scripts/train_warmup.sh'
  NP_RUNTIME=bwrap nix develop \"\$HOME/git/dotfiles/\" --command bash -lic 'deactivate && conda activate cfd && cd /home/dwcgt/git/TSM-PDE/ && ./scripts/train_generation.sh'
"
