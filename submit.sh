#!/bin/bash
#SBATCH --job-name=SSMEmbedder
#SBATCH --partition=gpu_test
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --chdir=/n/home03/creissel/SSM_Embedding/
#SBATCH --output=slurm_monitoring/%x-%j.out

### init virtual environment if needed
source ~/.bashrc
source /n/home03/creissel/miniforge3/etc/profile.d/conda.sh
conda activate ssm

srun python embedding.py 
