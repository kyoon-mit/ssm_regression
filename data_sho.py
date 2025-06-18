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

# path = '/n/holystore01/LABS/iaifi_lab/Users/creissel/SHO/'
savepath = '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

priors = dict()
priors['omega_0'] = scipy.stats.uniform(loc=0.1, scale=1.9)
priors['beta'] = scipy.stats.uniform(loc=0, scale=0.5)
priors['shift'] = scipy.stats.uniform(loc=-4, scale=8)

num_simulations = 100000 # number of time series to be generated
num_repeats = 10 # number of augmentations
num_points = 200 # length of time series
sigma = 0.4 # std of Gaussian to be added as noise

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
    omega_0 = torch.tensor(omega_0).to(dtype=torch.float32)
    beta = priors['beta'].rvs() if beta is None else beta
    beta = torch.tensor(beta).to(dtype=torch.float32)
    shift = priors['shift'].rvs() if shift is None else shift
    shift = torch.tensor(shift).to(dtype=torch.float32)

    t_vals = torch.linspace(-1, 10, num_points).to(dtype=torch.float32) #

    y = damped_sho(t_vals, omega_0=omega_0, beta=beta, shift=shift)
    y += sigma * torch.randn(size=y.size()).to(dtype=torch.float32)

    return t_vals, y, omega_0, beta, shift

# add augmentation
def generate_dataset():
    theta_unshifted_vals = []
    theta_shifted_vals = []
    data_unshifted_vals = []
    data_shifted_vals = []

    for ii in range(num_simulations):
        # generated data with a fixed shift
        t_vals, y_unshifted, omega, beta, shift = get_sho_data(num_points=num_points, shift=1)
        # create repeats
        theta_unshifted = torch.tensor([omega, beta, shift]).repeat(num_repeats, 1).to(device=device)
        theta_unshifted_vals.append(theta_unshifted)
        data_unshifted_vals.append(y_unshifted.repeat(num_repeats, 1).to(device=device))
        # generate shifted data
        theta_shifted = []
        data_shifted = []
        for _ in range(num_repeats):
            t_val, y_shifted, _omega, _beta, shift = get_sho_data(
                omega_0=omega, beta=beta,  # omega and beta same as above
                shift=None,
                num_points=num_points
            )
            theta_shifted.append(torch.tensor([omega, beta, shift]))
            data_shifted.append(y_shifted)
        theta_shifted_vals.append(torch.stack(theta_shifted).to(device=device))
        data_shifted_vals.append(torch.stack(data_shifted).to(device=device))

    return theta_unshifted_vals, theta_shifted_vals, data_unshifted_vals, data_shifted_vals

class DataGenerator(Dataset):
    def __init__(self, data):
        self.theta_unshifted_vals = data['theta_unshifted']
        self.theta_shifted_vals   = data['theta_shifted']
        self.data_unshifted_vals  = data['data_unshifted']
        self.data_shifted_vals    = data['data_shifted']

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
        )

def save_raw_tensors(theta_u, theta_s, data_u, data_s):
    # Convert lists to stacked tensors (size: [num_simulations, num_repeats, ...])
    theta_unshifted = torch.stack(theta_unshifted_vals)  # [100000, 10, 3]
    theta_shifted   = torch.stack(theta_shifted_vals)    # same shape
    data_unshifted  = torch.stack(data_unshifted_vals)   # [100000, 10, 200]
    data_shifted    = torch.stack(data_shifted_vals)

    # Split indices
    num_total  = theta_unshifted.shape[0]
    train_size = int(0.8 * num_total)
    val_size   = int(0.1 * num_total)
    test_size  = num_total - train_size - val_size

    indices   = torch.randperm(num_total)
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]
    test_idx  = indices[train_size + val_size:]

    # Helper to slice and save each split
    def save_split(name, idx):
        torch.save({
            'theta_unshifted': theta_unshifted[idx],
            'theta_shifted':   theta_shifted[idx],
            'data_unshifted':  data_unshifted[idx],
            'data_shifted':    data_shifted[idx],
        }, os.path.join(savepath, f'{name}.pt'))

    save_split('train', train_idx)
    save_split('val', val_idx)
    save_split('test', test_idx)

if __name__=='__main__':

    theta_unshifted_vals, theta_shifted_vals, data_unshifted_vals, data_shifted_vals = generate_dataset()

    save_raw_tensors(theta_unshifted_vals, theta_shifted_vals, data_unshifted_vals, data_shifted_vals)
    # dataset = DataGenerator(theta_unshifted_vals, theta_shifted_vals, data_unshifted_vals, data_shifted_vals)

    # train_set_size = int(0.8 * num_simulations)
    # val_set_size = int(0.1 * num_simulations)
    # test_set_size = int(0.1 * num_simulations)

    # train_data, val_data, test_data = torch.utils.data.random_split(
    #     dataset, [train_set_size, val_set_size, test_set_size])

    # torch.save(train_data, os.path.join(savepath, 'train.pt'))
    # torch.save(val_data, os.path.join(savepath, 'val.pt'))
    # torch.save(test_data, os.path.join(savepath, 'test.pt'))
    print(f'Files are saved to \'{savepath}\'.')