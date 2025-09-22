import h5py
import numpy as np
from scipy import stats
import argparse
import os

def normalize_to_gaussian(data, target_mean=0, target_std=1):
    """
    Normalize data to have a Gaussian distribution using quantile transformation.
    
    Parameters:
    -----------
    data : array-like
        Input data to normalize
    target_mean : float
        Target mean for the normalized distribution
    target_std : float
        Target standard deviation for the normalized distribution
    
    Returns:
    --------
    normalized_data : numpy array
        Data transformed to follow a Gaussian distribution
    """
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(data)
    valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        raise ValueError("No valid data points found")
    
    # Sort the data and compute ranks
    sorted_indices = np.argsort(valid_data)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(valid_data))
    
    # Convert ranks to uniform distribution [0, 1]
    uniform_values = (ranks + 0.5) / len(valid_data)
    
    # Transform uniform to standard Gaussian using inverse CDF
    gaussian_values = stats.norm.ppf(uniform_values)
    
    # Scale to target mean and std
    normalized_values = gaussian_values * target_std + target_mean
    
    # Handle invalid values by setting them to the target mean
    result = np.full_like(data, target_mean, dtype=np.float32)
    result[valid_mask] = normalized_values
    
    return result

def process_h5_file(input_file, output_file, sample_size=None, target_mean=0, target_std=1):
    """
    Process H5 file to normalize specified parameters to Gaussian distributions.
    
    Parameters:
    -----------
    input_file : str
        Path to input H5 file
    output_file : str
        Path to output H5 file
    sample_size : int or None
        Number of samples to keep (None to keep all)
    target_mean : float
        Target mean for normalized distributions
    target_std : float
        Target standard deviation for normalized distributions
    """
    
    # Parameters to normalize
    params_to_normalize = ['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']
    
    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    
    with h5py.File(input_file, 'r') as infile:
        # Get the original sample size
        original_size = infile['chirp_mass'].shape[0]
        print(f"Original sample size: {original_size}")
        
        # Determine final sample size
        if sample_size is None:
            final_size = original_size
            indices = np.arange(original_size)
        else:
            final_size = min(sample_size, original_size)
            # Randomly select indices to maintain data diversity
            indices = np.random.choice(original_size, size=final_size, replace=False)
            indices = np.sort(indices)  # Sort for better I/O performance
            
        print(f"Final sample size: {final_size}")
        
        with h5py.File(output_file, 'w') as outfile:
            # Process each dataset
            for dataset_name in infile.keys():
                print(f"Processing dataset: {dataset_name}")
                
                # Read the original data
                if len(infile[dataset_name].shape) == 1:
                    # 1D datasets
                    original_data = infile[dataset_name][indices]
                elif len(infile[dataset_name].shape) == 3:
                    # 3D datasets (like 'data')
                    original_data = infile[dataset_name][indices, :, :]
                else:
                    # Handle other dimensionalities if needed
                    original_data = infile[dataset_name][indices]
                
                # Normalize if it's one of the target parameters
                if dataset_name in params_to_normalize:
                    print(f"  Normalizing {dataset_name} to Gaussian distribution...")
                    print(f"  Original range: [{np.min(original_data):.6f}, {np.max(original_data):.6f}]")
                    print(f"  Original mean: {np.mean(original_data):.6f}, std: {np.std(original_data):.6f}")
                    
                    normalized_data = normalize_to_gaussian(original_data, target_mean, target_std)
                    
                    print(f"  Normalized mean: {np.mean(normalized_data):.6f}, std: {np.std(normalized_data):.6f}")
                    print(f"  Normalized range: [{np.min(normalized_data):.6f}, {np.max(normalized_data):.6f}]")
                    
                    outfile.create_dataset(dataset_name, data=normalized_data, 
                                         dtype=np.float32, compression='gzip')
                else:
                    # Copy other datasets without modification
                    outfile.create_dataset(dataset_name, data=original_data, 
                                         dtype=infile[dataset_name].dtype, compression='gzip')
            
            # Add metadata about the normalization
            outfile.attrs['normalized_parameters'] = params_to_normalize
            outfile.attrs['target_mean'] = target_mean
            outfile.attrs['target_std'] = target_std
            outfile.attrs['original_sample_size'] = original_size
            outfile.attrs['final_sample_size'] = final_size
    
    print(f"Processing complete. Output saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Normalize H5 file parameters to Gaussian distributions')
    parser.add_argument('input_file', help='Input H5 file path')
    parser.add_argument('-o', '--output', help='Output H5 file path (default: adds _normalized suffix)')
    parser.add_argument('-n', '--sample_size', type=int, help='Target sample size (default: keep all samples)')
    parser.add_argument('--mean', type=float, default=0.0, help='Target mean for normalized distributions (default: 0.0)')
    parser.add_argument('--std', type=float, default=1.0, help='Target std for normalized distributions (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output = f"{base_name}_normalized.h5"
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    # Process the file
    try:
        process_h5_file(
            input_file=args.input_file,
            output_file=args.output,
            sample_size=args.sample_size,
            target_mean=args.mean,
            target_std=args.std
        )
    except Exception as e:
        print(f"Error processing file: {e}")
        return

if __name__ == "__main__":
    main()