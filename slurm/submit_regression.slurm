#!/bin/bash
#SBATCH --job-name=regression_batch
#SBATCH --output=/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/logs/regression_%A_%a.out
#SBATCH --error=/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/logs/regression_%A_%a.err
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-1

# List of your scripts
SCRIPTS=(
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sho_run_regression.sh
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sg_run_regression.sh
)

bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}