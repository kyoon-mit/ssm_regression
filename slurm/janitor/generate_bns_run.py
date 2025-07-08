#!/usr/bin/env python3
import argparse, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate BNS regression run script.')
    parser.add_argument('-d', '--d_model', type=int, required=True, help='Model dimension')
    parser.add_argument('-n', '--n_layers', type=int, required=True, help='Number of layers')
    parser.add_argument('-b', '--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, required=True, help='Number of epochs')
    args = parser.parse_args()
    print(args)

    script_dir = Path(__file__).resolve().parent.parent
    ssm_dir = script_dir.parent
    model_dir = os.path.join(ssm_dir.parent, 'models')
    bash_dir = script_dir / 'bash_scripts'
    filename = bash_dir / f'bns_run_regression_d{args.d_model}_n{args.n_layers}.sh'

    script_content = f'''#!/bin/bash
sleep $(( RANDOM % 91 ))
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
--d_model {args.d_model} \\
--n_layers {args.n_layers} \\
--loss NLLGaussian \\
--logfile="${{BASE_DIR}}/slurm/logs/bns_regression.log" \\
--comment="BNS injection (20000 samples); loss=NLLGaussian; d_model={args.d_model}, n_layers={args.n_layers}"
'''

    with open(filename, 'w') as f:
        f.write(script_content)

    print(f'Script "{filename}" created successfully.')

if __name__ == '__main__':
    main()