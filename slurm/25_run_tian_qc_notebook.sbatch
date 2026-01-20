#!/bin/bash
#SBATCH --job-name=tian_qc_nb
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Lab server compliance: proper environment setup
set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

export PYTHONUNBUFFERED=1

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Execute notebook in a non-interactive, Slurm-safe way
srun python -m jupyter nbconvert --to notebook --execute \
  --output notebooks/tian_qc_diagnosis.executed.ipynb \
  notebooks/tian_qc_diagnosis.ipynb
