source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy"
python call_fitting.py -t SineGaussian -j bilby -b 0 1000 \
--logfile="${BASE_DIR}/fitresults/bilby_sg/id00000-00999.log"
