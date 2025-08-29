source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/toy/call_regression.py -t SHO -s _dho_gaussian_pink_sigma0.4 \
--device cuda \
--epochs 120 \
--d_model 6 \
--n_layers 4 \
--d_output 6 \
--lr 0.001 \
--loss Quantile \
--logfile="${BASE_DIR}/slurm/logs/sho_regression.log" \
--comment="Gaussian pink noise with sigma 0.4; loss=Quantile; d_model=6, n_layers=4"