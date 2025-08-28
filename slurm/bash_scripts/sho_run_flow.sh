source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/toy/call_flow.py -t SHO \
--embed_model /ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/saved_models/DHO/embedding.CNN.SHO.250828032239.pt \
-l 2 \
--hidden_features 30 \
--num_points 200 \
--embed_hidden_channels 20 \
--embed_kernel_size 21 \
--embed_d_output 6 \
-s _dho_gaussian_pink_sigma0.4 \
-d cuda \
-e 120 \
--logfile="${BASE_DIR}/slurm/logs/sho_flow.out" \
--comment="Gaussian pink noise with sigma 0.4; num_hidden_layers=3; hidden_features=100"
