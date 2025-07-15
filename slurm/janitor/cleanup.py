#!/usr/bin/env python3

import sys
import time
from pathlib import Path
import argparse

def cleanup_logs(log_dir: Path, n_days: int):
    if not log_dir.is_dir():
        print(f'Error: Directory {log_dir} does not exist.')
        return

    now = time.time()
    cutoff = now - n_days * 86400  # seconds in a day

    deleted = 0
    for filepath in log_dir.iterdir():
        if filepath.is_file():
            mtime = filepath.stat().st_mtime
            if mtime < cutoff:
                try:
                    filepath.unlink()
                    print(f'Deleted: {filepath}')
                    deleted += 1
                except Exception as e:
                    print(f'Failed to delete {filepath}: {e}')
    print(f'Done. Deleted {deleted} file(s).')

def main():
    parser = argparse.ArgumentParser(
        description='Delete files older than N days in the logs directory.'
    )
    parser.add_argument('days', type=int, help='Number of days (files older than this will be deleted)')

    args = parser.parse_args()

    if args.days < 0:
        print('Error: Number of days must be non-negative.')
        sys.exit(1)

    # Determine log directory relative to script location
    script_dir = Path(__file__).resolve().parent
    slurm_dir = script_dir.parent
    log_dir = slurm_dir / 'logs'

    cleanup_logs(log_dir, args.days)

if __name__ == '__main__':
    main()