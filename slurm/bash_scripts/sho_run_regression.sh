source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/toy/call_regression.py -t SHO -s _dho_gaussian_smear_sigma0.4 \
--device cuda \
--epochs 120 \
--d_model 16 \
--n_layers 4 \
--d_output 4 \
--lr 0.001 \
--loss NLLGaussian \
--logfile="${BASE_DIR}/slurm/logs/sho_regression.log" \
--comment="Gaussian noise with sigma 0.4; loss=Quantile; d_model=16, n_layers=4"