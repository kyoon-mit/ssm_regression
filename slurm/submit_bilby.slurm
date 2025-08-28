#!/bin/bash
#SBATCH --job-name=bilby_batch
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/logs/bilby_%A_%a.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/logs/bilby_%A_%a.err
#SBATCH --array=0-199
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=24:00:00

# List of your scripts
SCRIPTS=(
$(ls /ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sg_run_bilby_id*.sh | sort)
)

bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}