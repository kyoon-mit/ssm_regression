source /work/submit/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/ceph/submit/data/user/k/kyoon/KYoonStudy"
python call_fitting.py -t SineGaussian -j bilby -b 3000 4000 \
--logfile="${BASE_DIR}/fitresults/bilby_sg/id03000-03999.log"
