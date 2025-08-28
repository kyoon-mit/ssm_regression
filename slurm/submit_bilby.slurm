#!/bin/bash
#SBATCH --job-name=bilby_batch
#SBATCH --output=/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/logs/bilby_%A_%a.out
#SBATCH --error=/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/logs/bilby_%A_%a.err
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7G
#SBATCH --time=24:00:00

# List of your scripts
SCRIPTS=(
$(ls /ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sg_run_bilby_id*.sh | sort)
)

bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}