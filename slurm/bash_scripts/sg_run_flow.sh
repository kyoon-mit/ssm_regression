source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/toy/call_flow.py -t SineGaussian \
--embed_model /ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/embedding.CNN.SineGaussian.250717075146.pt \
-s _sigma0.4_gaussian \
-d cuda \
-e 60 \
--logfile="${BASE_DIR}/slurm/logs/sg_flow.out" \
--comment="Gaussian noise with sigma 0.4; default setting"
