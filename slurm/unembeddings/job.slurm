#!/bin/bash -xl
#
#SBATCH --gres=gpu::1
#SBATCH --time=24:00:00
#SBATCH --output=./slurm/logs/unembeddings_%x.%j.log

unset SLURM_EXPORT_ENV

export https_proxy=http://proxy:80
export http_proxy=http://proxy:80

module load python/3.12-conda
module load cuda/12.6

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

python3 "$SLURM_SUBMIT_DIR/unembeddings.py"
