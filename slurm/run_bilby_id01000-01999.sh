source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy"
python call_fitting.py -t SineGaussian -j bilby -b 1000 2000 \
--logfile="${BASE_DIR}/fitresults/bilby_sg/id01000-01999.log"
