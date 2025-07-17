#!/bin/bash
#SBATCH --job-name=regression_batch
#SBATCH --output=/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/logs/bns_regression_%A_%a.out
#SBATCH --error=/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/logs/bns_regression_%A_%a.err
#SBATCH --partition=submit-gpu
#SBATCH --nodelist=submit22-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=20:00:00
#SBATCH --array=0

# List of your scripts
SCRIPTS=(
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/bns_run_regression_d16_n10_o2.sh
)

bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}