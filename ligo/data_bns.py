import torch
from torch.utils.data import Dataset
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

class DataGenerator(Dataset):
    def __init__(self, hdf5_path, downsample_factor=1, duration=4, scale_factor=1.,
                 normalize=False):
        self.h5file = h5py.File(hdf5_path, 'r')
        self.coalescence_time = self.h5file.attrs['coalescence_time'] # Time of coalescence
        # self.duration = self.h5file.attrs['duration'] # Duration of the waveform in seconds
        # self.ifos = self.h5file.attrs['ifos'] # List of interferometers
        # self.length = self.h5file.attrs['length'] # Number of samples.
        # self.num_injections = self.h5file.attrs['num_injections'] # Number of waveform injections.
        self.sample_rate = self.h5file.attrs['sample_rate'] # Sample rate in Hz
        self.waveforms_h1 = self.h5file['waveforms/h1']
        self.waveforms_l1 = self.h5file['waveforms/l1']
        self.param_group = self.h5file['parameters']
        self.keys = list(self.param_group.keys())
        self.length = self.waveforms_h1.shape[0]
        self.downsample_factor, self.duration = int(downsample_factor), duration
        self.scale_factor = scale_factor
        self.normalize = normalize

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load waveforms
        h1 = self.scale_factor * torch.tensor(self.waveforms_h1[idx][::self.downsample_factor], dtype=torch.float32)
        l1 = self.scale_factor * torch.tensor(self.waveforms_l1[idx][::self.downsample_factor], dtype=torch.float32)

        start_idx = int(-self.duration * (self.sample_rate // self.downsample_factor))
        h1 = h1[start_idx:]
        l1 = l1[start_idx:]

        # Load parameters as a dictionary
        params = {k: torch.tensor(self.param_group[k][idx], dtype=torch.float32) for k in self.keys}
        return (h1, l1, params, idx)

def get_dataloaders(hdf5_path, downsample_factor=1, duration=64, scale_factor=1.,
                    train_batch_size=1000, val_batch_size=1000, test_batch_size=1000,
                    train_split=0.8, test_split=0.1,
                    split_indices_file='', random_seed=42):
    import os
    import numpy as np
    from torch.utils.data import DataLoader, Subset
    dataset = DataGenerator(hdf5_path, downsample_factor=downsample_factor,
                            duration=duration, scale_factor=scale_factor)
    if split_indices_file:
        if not split_indices_file.endswith('.npz'):
            raise ValueError("split_indices_file must be a .npz file containing precomputed indices.")
        # Check if the file exists
        if not os.path.exists(split_indices_file):
            raise FileNotFoundError(f"The file {split_indices_file} does not exist.")
        # Load precomputed indices
        indices = np.load(split_indices_file)
        train_indices = indices['train_indices']
        val_indices = indices['val_indices']
        test_indices = indices['test_indices']
    else:
        # Get indices
        num_samples = len(dataset)
        indices = np.arange(num_samples)
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
        # Split indices
        train_idx = int(train_split * num_samples)
        val_idx = int((1-test_split) * num_samples)

        train_indices = indices[:train_idx]
        val_indices = indices[train_idx:val_idx]
        test_indices = indices[val_idx:]

        np.savez('bns_data_indices.npz', train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)

    # Ensure indices are unique and sorted
    train_indices = np.unique(train_indices)
    val_indices = np.unique(val_indices)
    test_indices = np.unique(test_indices)
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    # Ensure indices are within the dataset length
    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        raise ValueError("One of the splits has no samples. Check your split ratios and dataset size.")
    if len(train_indices) + len(val_indices) + len(test_indices) != len(np.unique(np.concatenate((train_indices, val_indices, test_indices)))):
        raise ValueError("There are duplicate indices across the splits. Ensure that the indices are unique.")
    if len(train_indices) + len(val_indices) + len(test_indices) != len(dataset):
        raise ValueError("The total number of indices does not match the dataset size. Check your split ratios and dataset size.")
    # Check that the indices don't overlap
    if (set(train_indices) & set(val_indices)) or (set(train_indices) & set(test_indices)) or (set(val_indices) & set(test_indices)):
        raise ValueError("Indices overlap between train, validation, and test sets. Ensure that the splits are disjoint.")

    # Create Subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader