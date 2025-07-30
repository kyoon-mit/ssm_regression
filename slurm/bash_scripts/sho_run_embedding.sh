source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
python ${BASE_DIR}/toy/call_embedding.py -t SHO -s _sigma0.4_gaussian \
-e 220 \
-l 2 \
--num_points 200 \
--hidden_channels 20 \
--kernel_size 21 \
--d_output 6 \
--milestones 50 100 \
--activation 'relu' \
-d cpu \
--logfile="${BASE_DIR}/slurm/logs/sho_embedding.out" \
--comment="Gaussian noise with sigma 0.4; num_hidden_layers=2"