source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/toy/call_embedding.py -t SineGaussian -s _sigma0.4_gaussian \
-e 220 \
-s _sigma0.4_gaussian \
-d cuda \
-l 5 \
--logfile="${BASE_DIR}/slurm/logs/sg_embedding.out" \
--comment="Gaussian noise with sigma 0.4; num_hidden_layers=5"