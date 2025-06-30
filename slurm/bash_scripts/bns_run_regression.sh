source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression"
MODEL_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy/models"
python ${BASE_DIR}/ligo/call_regression_bns.py -t BNS \
--hdf5-path ${MODEL_DIR}/BNS/bns_waveforms.hdf5 \
--split-indices-file ${MODEL_DIR}/BNS/bns_data_indices.npz \
--batch_size 100 \
--device cuda \
--epochs 200 \
--d_model 2 \
--n_layers 1 \
--loss NLLGaussian \
--logfile="${BASE_DIR}/slurm/logs/bns_regression.log" \
--comment="First trial with BNS injection (20000 samples); loss=NLLGaussian; minimalistic model w/ d_model=2, n_layers=1"