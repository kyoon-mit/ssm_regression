#!/usr/bin/env python3
import argparse, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate bilby run script.')
    parser.add_argument('-d', '--datatype', type=str, choices=['SHO', 'SineGaussian'])
    parser.add_argument('--datasfx', type=str, default='_sigma0.4_gaussian', help='Suffix of the data file.')
    parser.add_argument('--num_points', type=int, default=200, help='Number of points in the t values.')
    parser.add_argument('--t_vals_start', type=int, default=-1)
    parser.add_argument('--t_vals_stop', type=int, default=10)
    args = parser.parse_args()
    print(args)

    if args.datatype=='SHO': datatag='sho'
    elif args.datatype=='SineGaussian': datatag='sg'
    script_dir = Path(__file__).resolve().parent.parent
    bash_dir = script_dir / 'bash_scripts'

    for i in range(0, 10000, 100):
        start, end = f'{i:05d}', f'{i+99:05d}'
        filename = bash_dir / f'{datatag}_run_bilby_id{start}-{end}.sh'
        script_content = f'''#!/bin/bash
source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate ssm
BASE_DIR="/n/holystore01/LABS/iaifi_lab/Lab/kyoon"
python ${{BASE_DIR}}/ssm_regression/toy/call_fitting.py -t {args.datatype} -j bilby -b {start} {end} \\
--datasfx {args.datasfx} \\
--num_points {args.num_points} --t_vals_start {args.t_vals_start} --t_vals_stop {args.t_vals_stop} \\
--logfile="${{BASE_DIR}}/logs/bilby_{datatag}/id{start}-{end}.log"
'''
        with open(filename, 'w') as f:
            f.write(script_content)
        print(f'Script "{filename}" created successfully.')

if __name__ == '__main__':
    main()