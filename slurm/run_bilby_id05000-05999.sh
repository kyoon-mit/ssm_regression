source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy"
python call_fitting.py -t SineGaussian -j bilby -b 5000 6000 \
--logfile="${BASE_DIR}/fitresults/bilby_sg/id05000-05999.log"
