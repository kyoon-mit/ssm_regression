import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import lightning as L
import h5py

class BNSDataset(Dataset):
    def __init__(self,
        hdf5_path,
        variables, # set of variables
        downsample_factor=1,
        start_time=0,
        end_time=64,
        scale_factor=1.,
        normalize=False
    ):
        super().__init__()
        self.valid_keys = {
            'chi1', 'chi2', 'chirp_mass', 'dec', 'distance', 'inclination',
            'mass_1', 'mass_2', 'mass_ratio', 'phi', 'phic', 'psi', 's1z', 's2z', 'snr'
        }
        self.derived_keys = {'total_mass'}
        self.h5file = h5py.File(hdf5_path, 'r')
        # if True:
        # with h5py.File(hdf5_path, 'r') as h5file:
            # self.coalescence_time = self.h5file.attrs['coalescence_time'] # Time of coalescence
            # self.duration = h5file.attrs['duration'] # Duration of the waveform in seconds
            # self.ifos = h5file.attrs['ifos'] # List of interferometers
            # self.length = h5file.attrs['length'] # Number of samples.
            # self.num_injections = h5file.attrs['num_injections'] # Number of waveform injections.
            # self.sample_rate = self.h5file.attrs['sample_rate'] # Sample rate in Hz
        self.sample_rate = 2048 # Sample rate in Hz
            # self.waveforms_h1 = self.h5file['waveforms/h1']
            # self.waveforms_l1 = self.h5file['waveforms/l1']
        self.data = self.h5file['data']
            # self.param_data_from_file = self.h5file['parameters']
            # self.length = self.waveforms_h1.shape[0]
        self.length = self.data.shape[0]
        self.keys = set(variables) & (self.valid_keys | self.derived_keys)
        if not self.keys:
            raise ValueError(f'Valid variables are: {self.valid_keys}.')
        self.downsample_factor = int(downsample_factor)
        self.start_time, self.end_time = start_time, end_time
        self.scale_factor = scale_factor
        self.normalize = normalize

    def _prepare_params_(self, idx: int):
        """
        Compute derived parameters and update the params dict.
        Derived parameters include chirp_mass, mass_ratio, and total_mass.
        """
        params = {}


        # compute derived only if requested
        # if 'chirp_mass' in self.keys:
        #     params['chirp_mass'] = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        # if 'mass_ratio' in self.keys:
        #     params['mass_ratio'] = m2 / m1
        if {'total_mass', 'chirp_mass', 'mass_ratio'} & self.keys:
            # load masses from file if available (safe access)
            m1 = torch.tensor(self.h5file['mass_1'][idx], dtype=torch.float32)
            m2 = torch.tensor(self.h5file['mass_2'][idx], dtype=torch.float32)
            # params['total_mass'] = m1 + m2 # <-- ?

        # add any other requested scalar parameters that exist
        for k in (self.keys - self.derived_keys):
            if k == 'redshift' and 'distance' in self.h5file:
                # if user asked for redshift but file has distance, map if appropriate
                params['redshift'] = torch.tensor(self.h5file['distance'][idx], dtype=torch.float32)
            else:
                params[k] = torch.tensor(self.h5file[k][idx], dtype=torch.float32)
        return params

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load waveforms
        # h1 = self.scale_factor * torch.tensor(self.waveforms_h1[idx][::self.downsample_factor], dtype=torch.float32)
        # l1 = self.scale_factor * torch.tensor(self.waveforms_l1[idx][::self.downsample_factor], dtype=torch.float32)
        start_idx = int(self.start_time * self.sample_rate)
        end_idx = int(self.end_time * self.sample_rate)

        h1 = self.scale_factor * torch.tensor(self.data[idx][0][start_idx:end_idx:self.downsample_factor], dtype=torch.float32)
        l1 = self.scale_factor * torch.tensor(self.data[idx][1][start_idx:end_idx:self.downsample_factor], dtype=torch.float32)

        if self.normalize:
            h1 = (h1 - h1.mean()) / (h1.std())
            l1 = (l1 - l1.mean()) / (l1.std())
            
        # Load parameters as a dictionary
        params = self._prepare_params_(idx)
        return (h1, l1, params, idx)

class LitBNSDataModule(L.LightningDataModule):
    def __init__(self,
        hdf5_path,
        variables, # set of variables to include in the dataset
        downsample_factor=1,
        start_time=0,
        end_time=64,
        scale_factor=1.,
        normalize=False,
        train_batch_size=1000, val_batch_size=1000, test_batch_size=1000,
        train_split=0.8, test_split=0.1,
        split_indices_file='',
        random_seed=42
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hdf5_path = hdf5_path
        self.variables = variables
        self.downsample_factor = downsample_factor
        self.start_time = start_time
        self.end_time = end_time
        self.scale_factor = scale_factor
        self.normalize = normalize
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_split = train_split
        self.test_split = test_split
        self.split_indices_file = split_indices_file
        self.random_seed = random_seed

    def prepare_data(self):
        if hasattr(self, 'dataset'):
            # already prepared
            return
        self.dataset = BNSDataset(self.hdf5_path, variables=self.variables,
                                  downsample_factor=self.downsample_factor,
                                  start_time=self.start_time, end_time=self.end_time,
                                  normalize=self.normalize, scale_factor=self.scale_factor)
        if self.split_indices_file:
            if not self.split_indices_file.endswith('.npz'):
                raise ValueError("split_indices_file must be a .npz file containing precomputed indices.")
            # Check if the file exists
            import os
            if not os.path.exists(self.split_indices_file):
                raise FileNotFoundError(f"The file {self.split_indices_file} does not exist.")
            # Load precomputed indices
            self.indices = np.load(self.split_indices_file)
        else:
            # Get indices
            num_samples = len(self.dataset)
            indices = np.arange(num_samples)
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
            
            # Split indices
            train_idx = int(self.train_split * num_samples)
            val_idx = int((1-self.test_split) * num_samples)

            self.indices = {
                'train_indices': indices[:train_idx],
                'val_indices': indices[train_idx:val_idx],
                'test_indices': indices[val_idx:]
            }
            np.savez(self.split_indices_file, **self.indices)

    def setup(self, stage: str):
        if not hasattr(self, 'dataset'):
            self.prepare_data()
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset, self.val_dataset = \
                Subset(self.dataset, self.indices['train_indices']), \
                Subset(self.dataset, self.indices['val_indices'])

        # Assign test dataset for use in dataloader
        elif stage in ("test", "predict"):
            self.test_dataset = Subset(self.dataset, self.indices['test_indices'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)