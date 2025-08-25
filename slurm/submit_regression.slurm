#!/bin/bash
#SBATCH --job-name=regression_batch
#SBATCH --output=<your directory>/ssm_regression/slurm/logs/regression_%A_%a.out
#SBATCH --error=<your directory>
#SBATCH --partition=submit-gpu
#SBATCH --nodelist=submit77-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --time=5:00:00
#SBATCH --array=0-1

# List of your scripts
SCRIPTS=(
<your directory>/ssm_regression/slurm/bash_scripts/sho_run_regression.sh
<your directory>/ssm_regression/slurm/bash_scripts/sg_run_regression.sh
)

bash ${SCRIPTS[$SLURM_ARRAY_TASK_ID]}