#!/bin/bash
#SBATCH --job-name=bilby_batch
#SBATCH --output=logs/bilby_%A_%a.out
#SBATCH --error=logs/bilby_%A_%a.err
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

# List of your scripts
SCRIPTS=(
run_bilby_id00000-00999.sh
run_bilby_id01000-01999.sh
run_bilby_id02000-02999.sh
run_bilby_id03000-03999.sh
run_bilby_id04000-04999.sh
run_bilby_id05000-05999.sh
run_bilby_id06000-06999.sh
run_bilby_id07000-07999.sh
run_bilby_id08000-08999.sh
run_bilby_id09000-09999.sh
)

cd /ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression
bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}