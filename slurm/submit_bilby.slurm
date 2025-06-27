#!/bin/bash
#SBATCH --job-name=bilby_batch
#SBATCH --output=logs/bilby_%A_%a.out
#SBATCH --error=logs/bilby_%A_%a.err
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

# List of your scripts
SCRIPTS=(
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sg_run_bilby_id{00000..09999..1000}.sh
/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/slurm/bash_scripts/sho_run_bilby_id{00000..09999..1000}.sh
)

bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}