# Instructions

## Dataset generation
There are two toy datasets (described in the paper):
1. Damped Harmonic Oscillator
2. Sine Gaussian pulse

For data generation, simply do
```
python data_sho.py
python data_sinegaussin.py
```
Make sure to open the files first and modify ```savepath```.

## Training
Modify the files in ```slurm/bash_scripts``` so that your conda installation and ```BASE_DIR``` are correctly pointed. Submit your training via e.g. ```slurm/submit_regression.slurm```.

! Make sure to modify ```toy/regression.py``` so that ```self.datadir``` points to the right place.