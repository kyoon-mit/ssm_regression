source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy"
python call_fitting.py -t SineGaussian -j bilby -b 7000 8000 \
--logfile="${BASE_DIR}/fitresults/bilby_sg/id07000-07999.log"
