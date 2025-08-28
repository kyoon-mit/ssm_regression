source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/toy/call_flow.py -t SineGaussian \
--embed_model /ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/saved_models/SG/embedding.CNN.SineGaussian.250828030459.pt \
-l 2 \
--hidden_features 20 \
--num_points 500 \
--embed_hidden_channels 10 \
--embed_kernel_size 11 \
--embed_d_output 12 \
-s _sg_gaussian_pink_sigma0.4 \
-d cuda \
-e 120 \
--logfile="${BASE_DIR}/slurm/logs/sg_flow.out" \
--comment="Gaussian pink noise with sigma 0.4; num_hidden_layers=3; hidden_features=100"