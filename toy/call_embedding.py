import wandb
import logging
import argparse
from embedding import Embedding

import os, sys
import torch
from datetime import datetime

from pathlib import Path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent, 'modules'))
from utils import configure_logging

def parse_args():
    parser = argparse.ArgumentParser(prog='embedding.py')
    parser.add_argument('-t', '--datatype', type=str,
                        choices=['SHO', 'SineGaussian', 'LIGO'],
                        help='Data type or dataset.')
    parser.add_argument('-d', '--device', type=str,
                        default='cuda',
                        help='Device to run on.')
    parser.add_argument('-e', '--epochs', type=int,
                        default=150,
                        help='Number of epochs.')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=1000,
                        help='Batch size for training and validation.')
    parser.add_argument('-s', '--suffix', type=str,
                        default='',
                        help='Suffix for the dataset, e.g., "_sigma0.4_gaussian".')
    parser.add_argument('-l', '--num_hidden_layers_h', type=int,
                        default=2,
                        help='Number of hidden layers in the embedding model.')
    parser.add_argument('--logfile', type=str, default=None, help='Name of the log file.')
    parser.add_argument('--loglevel', type=str, default='info',
                        choices=['notset', 'debug', 'info', 'warning', 'error', 'critical'],
                        help='Log level.')
    parser.add_argument('--comment', type=str, default='',
                        help='Comment for the run.')
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    configure_logging(logname='embedding', logfile=args.logfile, loglevel=args.loglevel, timestamp=timestamp)
    
    datatype = args.datatype
    datatag = 'toy'  # default tag for toy datasets
    if datatype not in ['SHO', 'SineGaussian', 'LIGO']:
        raise ValueError(f"Invalid datatype: {datatype}. Choose from 'SHO', 'SineGaussian', or 'LIGO'.")
    if datatype=='SHO': datatag = 'sho'
    elif datatype=='SineGaussian': datatag = 'sg'
    elif datatype=='LIGO': datatag = 'ligo'

    # os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project=f'embedding_{datatag}', name=f'embedding_{datatag}_{timestamp}')

    task = Embedding(datatype=datatype, datasfx=args.suffix,
                    device=args.device, batch_size=args.batch_size,
                    num_hidden_layers_h=args.num_hidden_layers_h)
    task.build_model()

    print('Start training...')

    EPOCHS = args.epochs

    for epoch_number in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        wt_repr, wt_cov, wt_std = (1, 1, 1)

        if epoch_number < 20:
            wt_repr, wt_cov, wt_std = (5, 1, 5)
        elif epoch_number < 45:
            wt_repr, wt_cov, wt_std = (2, 2, 1)

        print(f"VicReg wts: {wt_repr=} {wt_cov=} {wt_std=}")
        # Gradient tracking
        task.model.train(True)
        avg_train_loss = task.train_one_epoch(epoch_number, wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)

        # no gradient tracking, for validation
        task.model.train(False)
        avg_val_loss = task.val_one_epoch(epoch_number, wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)

        print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

        modelname = os.path.join(task.modeldir, f'embedding.CNN.{datatype}.{timestamp}.pt')
        wandb.log({
            'epoch': epoch_number,
            'train_loss': avg_train_loss,
            'val_accuracy': avg_val_loss,
            'lr': task.optimizer.param_groups[0]['lr'],
            'wt_repr': wt_repr,
            'wt_cov': wt_cov,
            'wt_std': wt_std,
            'num_hidden_layers_h': args.num_hidden_layers_h,
            'datatype': datatype,
            'datasfx': args.suffix,
            'device': args.device,
            'batch_size': args.batch_size,
            'timestamp': timestamp,
            'modeldir': task.modeldir,
            'modelname': modelname,
            'comment': args.comment
        })

        epoch_number += 1
        try:
            task.scheduler.step(avg_val_loss)
        except TypeError:
            task.scheduler.step()

    wandb.finish()

    torch.save(task.model.state_dict(), modelname)

if __name__=='__main__':
    main()