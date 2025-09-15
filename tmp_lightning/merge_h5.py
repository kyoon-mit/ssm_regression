import h5py
import glob
import numpy as np

input_files = sorted(glob.glob('/n/holystore01/LABS/iaifi_lab/Lab/kyoon/models/BNS/sig_*.h5'))
output_file = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/models/BNS/bns_new_waveforms.h5'

# Open first file to get structure
with h5py.File(input_files[0], 'r') as f0:
    def copy_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            shape = (0,) + obj.shape[1:]  # initial empty dataset with same shape
            dtype = obj.dtype
            f_out.create_dataset(name, shape=shape, maxshape=(None, *obj.shape[1:]), dtype=dtype, chunks=True)

with h5py.File(output_file, 'w') as f_out:
    # Recursively copy structure
    with h5py.File(input_files[0], 'r') as f0:
        f0.visititems(lambda name, obj: copy_structure(name, obj))
    
    # Now append data from all files
    for fname in input_files:
        with h5py.File(fname, 'r') as f_in:
            for name, dset in f_in.items():
                out_dset = f_out[name]
                # Resize along first dimension
                old_len = out_dset.shape[0]
                new_len = old_len + dset.shape[0]
                out_dset.resize(new_len, axis=0)
                out_dset[old_len:new_len] = dset[:]
