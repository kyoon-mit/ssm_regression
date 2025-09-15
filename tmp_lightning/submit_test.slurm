#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --time=48:00:00
#SBATCH --job-name=bns
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/slurm_logs/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/slurm_logs/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm

echo "Running on host: $(hostname)"
echo "Using config file: $1"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ${TOP_DIR}/tmp_lightning
python cli.py fit --config "$1"