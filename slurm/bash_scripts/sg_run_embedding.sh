source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/toy/call_embedding.py -t SineGaussian -s _sg_gaussian_pink_sigma0.4 \
-e 200 \
-l 2 \
--num_points 500 \
--hidden_channels 10 \
--kernel_size 11 \
--d_output 12 \
--milestones 20 40 \
--activation 'tanh' \
-d cuda \
--logfile="${BASE_DIR}/slurm/logs/sg_embedding.out" \
--comment="Gaussian pink noise with sigma 0.4; num_hidden_layers=2"