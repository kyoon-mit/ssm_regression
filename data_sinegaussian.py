import os
import numpy as np
import scipy.stats
import torch
from torch.utils.data import Dataset

# path = '/n/holystore01/LABS/iaifi_lab/Users/creissel/SHO/'
savepath = '/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SineGaussian'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

priors = dict()
priors['f_0'] = scipy.stats.uniform(0.1, 1)
priors['tau'] = scipy.stats.uniform(1, 4)
priors['shift'] = scipy.stats.uniform(-2, 2)

num_simulations = 100000 # number of time series to be generated
num_repeats = 10 # number of augmentations
num_points = 200 # length of time series
sigma = 0.4 # std of Gaussian to be added as noise

# generate time series data

def sine_gaussian(t, f_0, tau, shift=0):
    t_shift = t - shift
    data = torch.exp(-1*t_shift*t_shift / tau / tau) * torch.sin(2*torch.pi*f_0 * t_shift)
    return data

def get_sg_data(f_0=None, tau=None, shift=None, _num_points=num_points, _sigma=sigma):
    """Sample f_0, tau, shift and return a batch of data with noise"""
    f_0 = priors['f_0'].rvs() if f_0 is None else f_0
    f_0 = torch.tensor(f_0).to(dtype=torch.float32)
    tau = priors['tau'].rvs() if tau is None else tau
    tau = torch.tensor(tau).to(dtype=torch.float32)
    shift = priors['shift'].rvs() if shift is None else shift
    shift = torch.tensor(shift).to(dtype=torch.float32)
    
    t_vals = torch.linspace(-1, 10, _num_points).to(dtype=torch.float32) #
    
    y = sine_gaussian(t_vals, f_0=f_0, tau=tau, shift=shift)
    y += _sigma * torch.randn(size=y.size()).to(dtype=torch.float32)
    
    return t_vals, y, f_0, tau, shift

# add augmentation
def generate_dataset():
    theta_unshifted_vals = []
    theta_shifted_vals = []
    data_unshifted_vals = []
    data_shifted_vals = []

    for ii in range(num_simulations):
        # generated data with a fixed shift
        t_vals, y_unshifted, f_0, tau, shift = get_sg_data(num_points=num_points, shift=1)
        # create repeats
        theta_unshifted = torch.tensor([f_0, tau, shift]).repeat(num_repeats, 1).to(device=device)
        theta_unshifted_vals.append(theta_unshifted)
        data_unshifted_vals.append(y_unshifted.repeat(num_repeats, 1).to(device=device))
        # generate shifted data
        theta_shifted = []
        data_shifted = []
        for _ in range(num_repeats):
            t_val, y_shifted, _f_0, _tau, shift = get_sg_data(
                f_0=f_0, tau=tau,  # f_0 and tau same as above
                shift=None,
                num_points=num_points
            )
            theta_shifted.append(torch.tensor([f_0, tau, shift]))
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