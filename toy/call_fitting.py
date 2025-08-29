import os
import logging
import argparse
from fitting import Fitting

def _none_or_float(value):
    if value == 'None':
        return None
    return float(value)

def parse_args():
    parser = argparse.ArgumentParser(prog='fitting.py')
    parser.add_argument('-t', '--datatype', type=str,
                        choices=['SHO', 'SineGaussian', 'LIGO'],
                        help='Data type or dataset.')
    parser.add_argument('-j', '--jobtype', type=str,
                        choices=['lmfit', 'bilby', 'pink'],
                        help='Name of the job to run.')
    parser.add_argument('-d', '--device', type=str,
                        default='cpu',
                        help='Device to run on.')
    parser.add_argument('-b', '--batch-indices', nargs=2, type=int,
                        default=[0,1000],
                        help='Indices of the event id to run on.')
    parser.add_argument('--datasfx', type=str,
                        default='_sigma0.4_gaussian',
                        help='Suffix of the data file.')
    parser.add_argument('--num_points', type=int,
                        default=200,
                        help='Number of points in the t values.')
    parser.add_argument('--t_vals_start', type=int,
                        default=-1)
    parser.add_argument('--t_vals_stop', type=int,
                        default=10)
    parser.add_argument('--sigma', type=_none_or_float,
                        default='None')
    parser.add_argument('--logfile', type=str, default=None, help='Name of the log file.')
    parser.add_argument('--loglevel', type=str, default='info',
                        choices=['notset', 'debug', 'info', 'warning', 'error', 'critical'],
                        help='Log level.')
    args = parser.parse_args()
    print(args)
    return args

def configure_logging(logfile, loglevel):
    logger = logging.getLogger('fitting')
    # Remove all existing handlers
    logger.handlers.clear()
    match loglevel:
        case 'notset':
            logger.setLevel(logging.NOTSET)
        case 'debug':
            logger.setLevel(logging.DEBUG)
        case 'info':
            logger.setLevel(logging.INFO)
        case 'warning':
            logger.setLevel(logging.WARNING)
        case 'error':
            logger.setLevel(logging.ERROR)
        case 'critical':
            logger.setLevel(logging.CRITICAL)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)  # Ensure directory exists
        file_handler = logging.FileHandler(logfile, 'w+')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return

def run_bilby(datatype, device, batch_indices, datasfx, num_points, t_vals_start, t_vals_stop, sigma):
    fitter = Fitting(datatype=datatype, device=device, batch_indices=batch_indices, sigma=sigma,
                     datasfx=datasfx, num_points=num_points, t_vals_start=t_vals_start, t_vals_stop=t_vals_stop)
    fitter.run_bilby()
    return

def run_bilby_psd(datatype, device, batch_indices, datasfx, num_points, t_vals_start, t_vals_stop, sigma):
    fitter = Fitting(datatype=datatype, device=device, batch_indices=batch_indices, sigma=sigma,
                     datasfx=datasfx, num_points=num_points, t_vals_start=t_vals_start, t_vals_stop=t_vals_stop)
    fitter.run_bilby_psd()
    return

def run_lmfit(datatype, device, batch_indices, datasfx, num_points, t_vals_start, t_vals_stop, sigma):
    fitter = Fitting(datatype=datatype, device=device, batch_indices=batch_indices, sigma=sigma,
                     datasfx=datasfx, num_points=num_points, t_vals_start=t_vals_start, t_vals_stop=t_vals_stop)
    fitter.run_lmfit()
    return

def main():
    args = parse_args()
    configure_logging(logfile=args.logfile, loglevel=args.loglevel)
    if args.jobtype == 'lmfit':
        run_lmfit(datatype=args.datatype, device=args.device, batch_indices=tuple(args.batch_indices), sigma=args.sigma,
                  datasfx=args.datasfx, num_points=args.num_points, t_vals_start=args.t_vals_start, t_vals_stop=args.t_vals_stop)
    elif args.jobtype == 'bilby':
        run_bilby(datatype=args.datatype, device=args.device, batch_indices=tuple(args.batch_indices), sigma=args.sigma,
                  datasfx=args.datasfx, num_points=args.num_points, t_vals_start=args.t_vals_start, t_vals_stop=args.t_vals_stop)
    elif args.jobtype == 'pink':
        run_bilby_psd(datatype=args.datatype, device=args.device, batch_indices=tuple(args.batch_indices), sigma=args.sigma,
                  datasfx=args.datasfx, num_points=args.num_points, t_vals_start=args.t_vals_start, t_vals_stop=args.t_vals_stop)

if __name__=='__main__':
    main()