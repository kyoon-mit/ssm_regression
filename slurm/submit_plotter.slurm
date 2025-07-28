#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --output=/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/logs/plot_%A_%a.out
#SBATCH --error=/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/logs/plot_%A_%a.err
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --array=0

# List of your scripts
SCRIPTS=(
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/plot_toy.sh
)

bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}