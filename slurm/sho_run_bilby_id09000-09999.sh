source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy"
python /ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/call_fitting.py -t SHO -j bilby -b 9000 10000 \
--logfile="${BASE_DIR}/fitresults/bilby_sho/id09000-09999.log"
