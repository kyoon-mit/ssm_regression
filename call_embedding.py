import wandb
import logging
import argparse
from embedding import Embedding

import os
import torch
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(prog='embedding.py')
    parser.add_argument('-t', '--datatype', type=str,
                        choices=['SHO', 'SineGaussian', 'LIGO'],
                        help='Data type or dataset.')
    parser.add_argument('-d', '--device', type=str,
                        default='cpu',
                        help='Device to run on.')
    parser.add_argument('-e', '--epochs', type=int,
                        default=220,
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
    args = parser.parse_args()
    print(args)
    return args

def configure_logging(logfile, loglevel):
    logger = logging.getLogger('embedding')
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

def main():
    args = parse_args()
    configure_logging(logfile=args.logfile, loglevel=args.loglevel)
    
    datatype = args.datatype
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')

    # os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project=f'embedding_{datatype}', name=f'embedding_{datatype}_{timestamp}')

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

        wandb.log({
            'epoch': epoch_number,
            'train_loss': avg_train_loss,
            'val_accuracy': avg_val_loss,
            'lr': task.optimizer.param_groups[0]['lr'],
            'wt_repr': wt_repr,
            'wt_cov': wt_cov,
            'wt_std': wt_std
        })

        epoch_number += 1
        try:
            task.scheduler.step(avg_val_loss)
        except TypeError:
            task.scheduler.step()

    wandb.finish()

    torch.save(task.model.state_dict(), os.path.join(task.modeldir, f'model.CNN.{datatype}.{timestamp}.path'))

if __name__=='__main__':
    main()