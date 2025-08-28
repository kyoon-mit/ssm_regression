source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/toy/call_embedding.py -t SHO -s _dho_gaussian_pink_sigma0.4 \
-e 200 \
-l 2 \
--num_points 200 \
--hidden_channels 20 \
--kernel_size 21 \
--d_output 6 \
--milestones 50 100 \
--activation 'relu' \
-d cuda \
--logfile="${BASE_DIR}/slurm/logs/sho_embedding.out" \
--comment="Gaussian pink noise with sigma 0.4; num_hidden_layers=2"