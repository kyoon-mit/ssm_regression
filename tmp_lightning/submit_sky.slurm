#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --job-name=sky_large
#SBATCH --nodes=8
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/slurm_logs/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/slurm_logs/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm

config_path=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/config_sky_localization.yaml

echo "Running on host: $(hostname)"
echo "Using config file: $config_path"
echo "d_model: $1"
echo "n_layers: $2"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_NAME=$(basename "$config_path" .yaml)
MY_LOG_NAME="${CONFIG_NAME}_d${1}_l${2}_${SLURM_JOB_ID}"

cd ${TOP_DIR}/tmp_lightning
srun --cpu-bind=none python cli.py fit --config "$config_path" \
    --trainer.logger.init_args.id=${MY_LOG_NAME} \
    --model.init_args.d_model=$1 \
    --model.init_args.n_layers=$2 \
    --trainer.accelerator="gpu" \
    --trainer.strategy="ddp" \
    --trainer.num_nodes=8 \
    --trainer.devices=1
