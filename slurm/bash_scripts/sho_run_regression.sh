source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/call_regression.py -t SHO -s _sigma0.4_gaussian \
--device cuda \
--epochs 100 \
--d_model 2 \
--n_layers 1 \
--loss NLLGaussian \
--logfile="${BASE_DIR}/slurm/logs/sho_regression.log" \
--comment="Gaussian noise with sigma 0.4; loss=NLLGaussian; minimalistic model w/ d_model=2, n_layers=1"