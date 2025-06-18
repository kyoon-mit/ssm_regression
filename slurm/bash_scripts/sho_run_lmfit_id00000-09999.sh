source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy"
python ${BASE_DIR}/ssm_regression/call_fitting.py -t SHO -j lmfit -b 0 10000 \
--logfile="${BASE_DIR}/fitresults/lmfit_sho/id00000-09999.log"