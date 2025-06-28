import torch
from torch.utils.data import Dataset
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

class DataGenerator(Dataset):
    def __init__(self, hdf5_path):
        self.h5file = h5py.File(hdf5_path, 'r')
        self.coalescence_time = self.h5file.attrs['coalescence_time'] # GPS time of coalescence
        self.duration = self.h5file.attrs['duration'] # Duration of the waveform in seconds
        self.ifos = self.h5file.attrs['ifos'] # List of interferometers
        self.length = self.h5file.attrs['length'] # Number of samples.
        self.num_injections = self.h5file.attrs['num_injections'] # Number of waveform injections.
        self.sample_rate = self.h5file.attrs['sample_rate'] # Sample rate in Hz
        self.waveforms_h1 = self.h5file['waveforms_h1']
        self.waveforms_l1 = self.h5file['waveforms_l1']
        self.param_group = self.h5file['parameters']
        self.keys = list(self.param_group.keys())
        self.length = self.waveforms_h1.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load waveforms
        h1 = torch.tensor(self.waveforms_h1[idx], dtype=torch.float32)
        l1 = torch.tensor(self.waveforms_l1[idx], dtype=torch.float32)

        # Load parameters as a dictionary
        params = {k: torch.tensor(self.param_group[k][idx], dtype=torch.float32) for k in self.keys}
        return (h1, l1, params)