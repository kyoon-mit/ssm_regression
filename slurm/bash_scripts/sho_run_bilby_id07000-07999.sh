source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy"
python /ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/call_fitting.py -t SHO -j bilby -b 7000 8000 \
--logfile="${BASE_DIR}/fitresults/bilby_sho/id07000-07999.log"
