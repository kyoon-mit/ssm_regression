# Instructions

## Dataset generation
There are two toy datasets (described in the paper):
1. Damped Harmonic Oscillator
2. Sine Gaussian pulse

For data generation, simply do
```
python data_sho.py
python data_sinegaussian.py
```
Make sure to open the files first and modify ```savepath```.

This will produce toy datasets in your ```savepath```, 100,000 samples each, split into 80:10:10 for training, validation, and testing. Each sample contains the true parameters of the models, the amplitude of the timeseries (i.e. values of y(t) for given t values), and the unique event identifier. In the ```.py``` files, these are identified as ```theta_u```, ```data_u```, and ```event_id```.

Unnecessary for training the SSM models, each sample also consists of parameters and amplitudes that represent values that are randomly shifted in time (i.e. t -> t + shift), and the arrays are repeated 10 times per sample. You may feel free to delete these parts. The shifted values and repeats were introduced for training with a baseline model that uses embedding + normalizing flow.

## Training
Modify the files in ```slurm/bash_scripts``` so that your conda installation and ```BASE_DIR``` are correctly pointed. Submit your training to slurm via e.g. ```slurm/submit_regression.slurm```.

! Make sure to modify ```toy/regression.py``` so that ```self.datadir``` points to the right place.
