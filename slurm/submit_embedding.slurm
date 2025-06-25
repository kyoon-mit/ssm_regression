#!/bin/bash
#SBATCH --job-name=embedding_batch
#SBATCH --output=logs/embedding_%A_%a.out
#SBATCH --error=logs/embedding_%A_%a.err
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-1

# List of your scripts
SCRIPTS=(
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sho_run_embedding.sh
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sg_run_embedding.sh
)

bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}