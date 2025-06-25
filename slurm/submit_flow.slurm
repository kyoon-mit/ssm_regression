#!/bin/bash
#SBATCH --job-name=flow
#SBATCH --output=logs/flow_%j.out
#SBATCH --error=logs/flow_%j.err
#SBATCH --partition=submit-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# Load conda and activate your environment
source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm

# Go to your working directory
cd /ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression

# Run your script
python parameter_estimation.py