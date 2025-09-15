#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/slurm_logs/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/slurm_logs/error-%j.err
RANDOM_SEED=$SLURM_ARRAY_TASK_ID

export NCCL_DEBUG=INFO
# Set NCCL_SOCKET_IFNAME to the correct interface or allow override via environment variable
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
# Use a CUDA allocator setting supported on this cluster to reduce fragmentation.
# 'expandable_segments' may be unsupported; use max_split_size_mb instead.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

TOP_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm

echo "Running on host: $(hostname)"
echo "Using config file: $1"

# ===== GPU pre-check and optional cleanup =====
# If KILL_STALE_GPUS is set to 'yes' the script will kill GPU compute processes
# owned by the submitting user on each allocated node. Default: do not kill.
KILL_STALE_GPUS="${KILL_STALE_GPUS:-yes}"

echo "Listing GPU compute processes on allocated nodes..."
# Run one task per node to list GPU compute processes (fallback to plain nvidia-smi if query fails)
srun -N${SLURM_NNODES} -n${SLURM_NNODES} --ntasks-per-node=1 bash -lc '
	echo "---- $(hostname) ----";
	nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null || nvidia-smi;
'

if [ "${KILL_STALE_GPUS}" = "yes" ]; then
	echo "KILL_STALE_GPUS=yes — attempting to kill compute processes owned by $(whoami) on allocated nodes"
	srun -N${SLURM_NNODES} -n${SLURM_NNODES} --ntasks-per-node=1 bash -lc '
		for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do
			if [ -n "$pid" ]; then
				owner=$(ps -o user= -p $pid 2>/dev/null || true)
				if [ "$owner" = "'$(whoami)'" ] || [ "$owner" = "$(whoami)" ]; then
					echo "Killing pid $pid on $(hostname) (owner=$owner)";
					kill -9 $pid 2>/dev/null || true;
				else
					echo "Skipping pid $pid on $(hostname) (owner=$owner)";
				fi;
			fi;
		done
	'
	echo "Waiting a moment for kernel cleanup..."
	sleep 3
	echo "GPU usage after cleanup:"
	srun -N${SLURM_NNODES} -n${SLURM_NNODES} --ntasks-per-node=1 bash -lc 'nvidia-smi || true'
else
	echo "KILL_STALE_GPUS is not 'yes' (currently='${KILL_STALE_GPUS}'). Skipping automatic kills."
fi

cd ${TOP_DIR}/tmp_lightning
srun python cli.py fit --config "$1" \
# --trainer.accelerator=gpu \
# --trainer.devices=1 \
# --trainer.num_nodes=1 \
# --trainer.strategy=ddp \
# Use mixed precision to reduce per-GPU memory. If this causes issues, remove the flag.