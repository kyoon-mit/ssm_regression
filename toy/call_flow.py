import wandb
import argparse
from parameter_estimation import NormalizingFlow

import os, sys
import torch
from torch import optim
from datetime import datetime

from pathlib import Path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent, 'modules'))
from utils import configure_logging

def parse_args():
    parser = argparse.ArgumentParser(prog='parameter_estimation.py')
    parser.add_argument('-t', '--datatype', type=str,
                        choices=['SHO', 'SineGaussian', 'LIGO'],
                        help='Data type or dataset.')
    parser.add_argument('--embed_model', type=str,
                        help='Full path to the embedding model.')
    parser.add_argument('-s', '--suffix', type=str,
                        default='',
                        help='Suffix for the dataset, e.g., "_sigma0.4_gaussian".')
    parser.add_argument('-d', '--device', type=str,
                        default='cuda',
                        help='Device to run on.')
    parser.add_argument('-e', '--epochs', type=int,
                        default=60,
                        help='Number of epochs.')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=1000,
                        help='Batch size for training and validation.')
    parser.add_argument('--logfile', type=str,
                        default=None,
                        help='Name of the log file.')
    parser.add_argument('--loglevel', type=str,
                        default='info',
                        choices=['notset', 'debug', 'info', 'warning', 'error', 'critical'],
                        help='Log level.')
    parser.add_argument('--comment', type=str,
                        default='',
                        help='Comment for the run.')
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    logger = configure_logging(logname='flow', logfile=args.logfile, loglevel=args.loglevel, timestamp=timestamp) 
    
    datatype = args.datatype
    datatag = 'toy'  # default tag for toy datasets
    if datatype not in ['SHO', 'SineGaussian', 'LIGO']:
        raise ValueError(f"Invalid datatype: {datatype}. Choose from 'SHO', 'SineGaussian', or 'LIGO'.")
    if datatype=='SHO': datatag = 'sho'
    elif datatype=='SineGaussian': datatag = 'sg'
    elif datatype=='LIGO': datatag = 'ligo'

    # os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project=f'flow_{datatag}', name=f'flow_{datatag}_{timestamp}')

    task = NormalizingFlow(embed_model=args.embed_model, datatype=datatype, datasfx=args.suffix,
                           device=args.device, batch_size=args.batch_size)
    task.build_flow()
    flow = task.flow

    wandb.init(project=f'flow_{task.datatype}', name=f'flow_{task.datatype}_embed{task.embed_timestamp}_{timestamp}')

    EPOCHS = args.epochs

    for epoch_number in range(EPOCHS):
        logger.info(f'EPOCH {epoch_number + 1}')
        avg_train_loss = task.train_one_epoch(epoch_number)
        avg_val_loss = task.val_one_epoch(epoch_number)

        logger.info(f"Train/Val flow Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

        wandb.log({
            'epoch': epoch_number,
            'train_loss': avg_train_loss,
            'val_accuracy': avg_val_loss,
            'lr': task.optimizer.param_groups[0]['lr'],
            'trainable_params': sum(p.numel() for p in flow.parameters() if p.requires_grad),
            'fixed_params': sum(p.numel() for p in flow._embedding_net.parameters() if not p.requires_grad),
            'total_params': sum(p.numel() for p in flow.parameters()),
            'datatype': datatype,
            'timestamp': timestamp,
            'embed_timestamp': task.embed_timestamp,
            'embed_model': args.embed_model,
            'comment': args.comment
        })

        for param_group in task.optimizer.param_groups:
            logger.info("Current LR = {:.3e}".format(param_group['lr']))
        epoch_number += 1
        try:
            task.scheduler.step(avg_val_loss)
        except TypeError:
            task.scheduler.step()
    wandb.finish()

    torch.save(flow.state_dict(), os.path.join(task.modeldir, f'flow.CNN.{task.datatype}.embed{task.embed_timestamp}.{timestamp}.pt'))

if __name__=='__main__':
    main()