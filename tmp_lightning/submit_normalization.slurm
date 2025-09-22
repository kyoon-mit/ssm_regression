#!/bin/bash
#SBATCH --partition=serial_requeue
#SBATCH --time=12:00:00
#SBATCH --job-name=normalization
#SBATCH --cpus-per-task=4
#SBATCH --mem=140G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/slurm_logs/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/slurm_logs/error-%j.err

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm

cd ${TOP_DIR}/tmp_lightning
python normalize_h5.py /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/bns/bns_waveforms_100000.h5