import os
import numpy as np
import scipy.stats
import torch
from torch.utils.data import Dataset
import random

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

num_simulations = 100000 # number of time series to be generated
num_repeats = 10 # number of augmentations
num_points = 200 # length of time series
sigma = 0.4 # std of Gaussian to be added as noise

savepath = './models/SHO' # MODIFY YOUR DIRECTORY HERE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

priors = dict()
priors['omega_0'] = scipy.stats.uniform(loc=0.1, scale=1.9)
priors['beta'] = scipy.stats.uniform(loc=0, scale=0.5)
priors['shift'] = scipy.stats.uniform(loc=-4, scale=8)

# generate time series data

def damped_sho(t, omega_0, beta, shift=0):
    # beta less than 1 for underdamped
    envel = beta * omega_0
    osc = torch.sqrt(1 - beta**2) * omega_0
    tau = t - shift
    data = torch.exp(-envel * tau) * torch.cos(osc * tau)
    data[tau < 0] = 0  # assume oscillator starts at tau = 0
    return data

def damped_sho_np(t, omega_0, beta, shift=0):
    # beta less than 1 for underdamped
    envel = beta * omega_0
    osc = np.sqrt(1 - beta**2) * omega_0
    tau = t - shift
    data = np.exp(-envel * tau) * np.cos(osc * tau)
    data[tau < 0] = 0  # assume oscillator starts at tau = 0
    return data

def get_sho_data(omega_0=None, beta=None, shift=None, num_points=1):
    """Sample omega, beta, shift and return a batch of data with noise"""
    omega_0 = priors['omega_0'].rvs() if omega_0 is None else omega_0
    omega_0 = torch.as_tensor(omega_0, dtype=torch.float32).clone().detach()
    beta = priors['beta'].rvs() if beta is None else beta
    beta = torch.as_tensor(beta, dtype=torch.float32).clone().detach()
    shift = priors['shift'].rvs() if shift is None else shift
    shift = torch.as_tensor(shift, dtype=torch.float32).clone().detach()

    t_start, t_end = -1, 10
    t_vals = torch.linspace(t_start, t_end, num_points).to(dtype=torch.float32)

    y_clean = damped_sho(t_vals, omega_0=omega_0, beta=beta, shift=shift)

    y_noise = sigma * torch.randn(size=y_clean.size()).to(dtype=torch.float32)
    y = y_clean + y_noise

    return t_vals, y, y_clean, y_noise, omega_0, beta, shift

def get_sho_data_np(omega_0=None, beta=None, shift=None, num_points=1):
    """Sample omega, beta, shift and return a batch of data with noise (numpy version)"""
    from scipy.fft import rfft, irfft
    omega_0 = priors['omega_0'].rvs() if omega_0 is None else omega_0
    beta = priors['beta'].rvs() if beta is None else beta
    shift = priors['shift'].rvs() if shift is None else shift

    t_start, t_end = -1, 10
    t_vals = np.linspace(t_start, t_end, num_points).astype(np.float32)

    y_clean = damped_sho_np(t_vals, omega_0=omega_0, beta=beta, shift=shift)
    y_noise = sigma * np.random.randn(*y_clean.shape).astype(np.float32)
    y = y_clean + y_noise

    return t_vals, y, y_clean, y_noise, omega_0, beta, shift

# add augmentation
def generate_dataset():
    theta_unshifted_vals = []
    theta_shifted_vals = []
    data_unshifted_vals = []
    data_shifted_vals = []
    event_id = []
    t_vals_array = []

    for ii in range(num_simulations):
        # generated data with a fixed shift
        t_vals, y_unshifted, y_clean_u, y_noise_u, omega_u, beta_u, shift_u =\
            get_sho_data(num_points=num_points, shift=1)
        # create repeats
        theta_unshifted = torch.tensor([omega_u, beta_u, shift_u]).repeat(num_repeats, 1).to(device=device)
        theta_unshifted_vals.append(theta_unshifted)
        data_unshifted_vals.append(y_unshifted.repeat(num_repeats, 1).to(device=device))
        # generate shifted data
        theta_shifted = []
        data_shifted = []
        for _ in range(num_repeats):
            _, y_shifted, y_clean_s, y_noise_s, omega_s, beta_s, shift_s =\
            get_sho_data(
                omega_0=omega_u, beta=beta_u,  # omega and beta same as above
                shift=None, # generates random shift
                num_points=num_points
            )
            theta_shifted.append(torch.tensor([omega_s, beta_s, shift_s]))
            data_shifted.append(y_shifted)
        theta_shifted_vals.append(torch.stack(theta_shifted).to(device=device))
        data_shifted_vals.append(torch.stack(data_shifted).to(device=device))
        event_id.append(ii)

    # Return dictionary of tensors
    return_dict = {
        'theta_u': torch.stack(theta_unshifted_vals),
        'theta_s': torch.stack(theta_shifted_vals),
        'data_u':  torch.stack(data_unshifted_vals),
        'data_s':  torch.stack(data_shifted_vals),
        'event_id': torch.tensor(event_id, dtype=torch.int32),
    }

    return return_dict

class DataGenerator(Dataset):
    def __init__(self, data):
        self.theta_unshifted_vals = data['theta_u']
        self.theta_shifted_vals   = data['theta_s']
        self.data_unshifted_vals  = data['data_u']
        self.data_shifted_vals    = data['data_s']
        self.event_id = data['event_id']

    def __len__(self):
        return self.theta_unshifted_vals.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # return shifted and unshifted theta and data
        return (
            self.theta_unshifted_vals[idx].to(dtype=torch.float32),
            self.theta_shifted_vals[idx].to(dtype=torch.float32),
            self.data_unshifted_vals[idx].to(dtype=torch.float32),
            self.data_shifted_vals[idx].to(dtype=torch.float32),
            self.event_id[idx].to(dtype=torch.int32),
        )

def save_raw_tensors(theta_u, theta_s, data_u, data_s,
                     event_id):
    # Split indices
    num_total  = theta_u.shape[0]
    train_size = int(0.8 * num_total)
    val_size   = int(0.1 * num_total)

    indices   = torch.randperm(num_total)
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]
    test_idx  = indices[train_size + val_size:]

    # Helper to slice and save each split
    def save_split(name, idx):
        torch.save({
            'theta_u': theta_u[idx],
            'theta_s': theta_s[idx],
            'data_u':  data_u[idx],
            'data_s':  data_s[idx],
            'event_id': event_id[idx],
        },
        os.path.join(savepath, f'{name}.pt'),
        _use_new_zipfile_serialization=True)

    save_split(f'train{sfx}', train_idx)
    save_split(f'val{sfx}', val_idx)
    save_split(f'test{sfx}', test_idx)

if __name__=='__main__':
    sigma = 0.4
    sfx = f'_gaussian_smear_sigma{sigma:.1f}'
    dataset = generate_dataset()
    save_raw_tensors(**dataset)
    print(f'Files are saved to \'{savepath}\'.')