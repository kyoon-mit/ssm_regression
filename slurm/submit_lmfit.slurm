#!/bin/bash
#SBATCH --job-name=lmfit_batch
#SBATCH --output=logs/lmfit_%A_%a.out
#SBATCH --error=logs/lmfit_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00

# List of your scripts
SCRIPTS=(
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sg_run_lmfit_id00000-09999.sh
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sho_run_lmfit_id00000-09999.sh
)

bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}