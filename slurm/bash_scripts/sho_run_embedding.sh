source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/call_embedding.py -t SHO -s _sigma0.4_gaussian \
--logfile="${BASE_DIR}/slurm/logs/sho_embedding.log"
