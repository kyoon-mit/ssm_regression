source <your conda installation here>/conda.sh
conda activate ssm
BASE_DIR="<your directory here>/ssm_regression"
python ${BASE_DIR}/toy/call_regression.py -t SHO -s _gaussian_smear_sigma0.4 \
--device cuda \
--epochs 120 \
--d_model 6 \
--n_layers 4 \
--d_output 4 \
--lr 0.001 \
--loss NLLGaussian \
--logfile="${BASE_DIR}/slurm/logs/sho_regression.log" \
--comment="Gaussian noise with sigma 0.4; loss=NLLGaussian; d_model=6, n_layers=4"