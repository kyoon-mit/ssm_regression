#!/usr/bin/env python3
import argparse, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate BNS regression run script.')
    parser.add_argument('-d', '--d_model', type=int, required=True, help='Model dimension')
    parser.add_argument('-n', '--n_layers', type=int, required=True, help='Number of layers')
    parser.add_argument('-b', '--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--d_output', type=int, default=18, help='Number of output nodes')
    parser.add_argument('--downsample_factor', type=int, default=2, help='Downsampling factor')
    parser.add_argument('--duration', type=int, default=4, help='Duration of the coalescence')
    parser.add_argument('--scale_factor', type=float, default=1., help='Scale factor')
    parser.add_argument('--normalize', type=bool, default=False, help='Normalize data')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--comment', type=str, default='', help='Additional comment for the run')
    args = parser.parse_args()
    print(args)

    script_dir = Path(__file__).resolve().parent.parent
    ssm_dir = script_dir.parent
    model_dir = os.path.join(ssm_dir.parent, 'models')
    bash_dir = script_dir / 'bash_scripts'
    filename = bash_dir / f'bns_run_regression_d{args.d_model}_n{args.n_layers}_o{args.d_output}.sh'

    script_content = f'''#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="{ssm_dir}"
MODEL_DIR="{model_dir}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ${{BASE_DIR}}/ligo/call_regression_bns.py -t BNS \\
--hdf5-path ${{MODEL_DIR}}/BNS/bns_waveforms.hdf5 \\
--split-indices-file ${{MODEL_DIR}}/BNS/bns_data_indices.npz \\
--batch_size {args.batch_size} \\
--device cuda \\
--epochs {args.epochs} \\
--d_output {args.d_output} \\
--d_model {args.d_model} \\
--n_layers {args.n_layers} \\
--dropout {args.dropout} \\
--loss NLLGaussian \\
--downsample_factor {args.downsample_factor} \\
--duration {args.duration} \\
--scale_factor {args.scale_factor} \\
--normalize {args.normalize} \\
--lr {args.learning_rate} \\
--logfile="${{BASE_DIR}}/slurm/logs/bns_regression.log" \\
--comment="BNS injection (20000 samples); loss=NLLGaussian;
d_model={args.d_model}, n_layers={args.n_layers}; d_output={args.d_output};
downsample_factor={args.downsample_factor}, duration={args.duration}, scale_factor={args.scale_factor};
normalize={args.normalize};
{args.comment}"
'''

    with open(filename, 'w') as f:
        f.write(script_content)

    print(f'Script "{filename}" created successfully.')

if __name__ == '__main__':
    main()