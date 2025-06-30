import wandb
import logging
import argparse
from regression_bns import SSMRegression
from tqdm.auto import tqdm

import os
import torch
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(prog='regression.py')
    parser.add_argument('-t', '--datatype', type=str,
                        choices=['BNS'],
                        help='Data type or dataset.')
    parser.add_argument('-l', '--loss', type=str,
                        choices=['NLLGaussian', 'Quantile'],
                        help='Type of loss function to use.')
    parser.add_argument('-d', '--device', type=str,
                        default='cuda',
                        help='Device to run on.')
    parser.add_argument('-e', '--epochs', type=int,
                        default=180,
                        help='Number of epochs.')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=1000,
                        help='Batch size for training.')
    parser.add_argument('-p', '--hdf5-path', type=str,
                        default='',
                        help='Path to the HDF5 dataset file.')
    parser.add_argument('-s', '--split-indices-file', type=str,
                        default='',
                        help='Path to the precomputed split indices file (npz format).')
    parser.add_argument('--d_model', type=int,
                        default=6,
                        help='Dimension of the model.')
    parser.add_argument('--n_layers', type=int,
                        default=4,
                        help='Number of layers in the model.')
    parser.add_argument('--dropout', type=float,
                        default=0.0,
                        help='Dropout rate.')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--logfile', type=str, default=None, help='Name of the log file.')
    parser.add_argument('--loglevel', type=str, default='info',
                        choices=['notset', 'debug', 'info', 'warning', 'error', 'critical'],
                        help='Log level.')
    parser.add_argument('--comment', type=str, default='',
                        help='Comment for the run.')
    args = parser.parse_args()
    print(args)
    return args

def configure_logging(logfile, loglevel):
    logger = logging.getLogger('ssm_regression')
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
    datatag = 'bns'  # default tag for toy datasets
    loss = args.loss
    
    if datatype not in ['BNS']:
        raise ValueError(f"Unsupported datatype: {datatype}. Supported types are 'BNS'.")
    if loss not in ['NLLGaussian', 'Quantile']:
        raise ValueError(f"Unsupported loss type: {loss}. Supported types are 'NLLGaussian', 'Quantile'.")
    
    # Initialize the regression model
    task = SSMRegression(
        d_model=args.d_model,
        n_layers=args.n_layers,
        datatype=args.datatype,
        loss=args.loss,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        device=args.device,
        hdf5_path=args.hdf5_path,
        split_indices_file=args.split_indices_file
    )
    task.build_model()
    task.setup_optimizer(lr=args.lr, weight_decay=args.weight_decay)

    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    wandb.init(project=f'ssm_{datatag}_regression', name=f'ssm_{datatag}_{timestamp}')
    EPOCHS = args.epochs

    pbar = tqdm(range(task.start_epoch, EPOCHS))
    for epoch_number in pbar:
        avg_train_loss = task.train()
        avg_val_loss = task.eval(epoch_number, checkpoint=True)
        if epoch_number == 0:
            pbar.set_description('Epoch: %d' % (epoch_number))
        else:
            pbar.set_description('Epoch: %d | Val loss: %1.3f' % (epoch_number, avg_val_loss))
        task.scheduler.step()

        wandb.log({
            'epoch': epoch_number,
            'train_loss': avg_train_loss,
            'val_accuracy': avg_val_loss,
            'learning_rate': task.optimizer.param_groups[0]['lr'],
            'doc_loss': task.doc_loss,
            'doc_val': task.doc_val,
            'model': task.model.state_dict(),
            'best_loss': task.best_loss,
            'timestamp': timestamp,
            'datatype': task.datatype,
            'loss': task.loss,
            'd_input': task.d_input,
            'd_model': task.d_model,
            'n_layers': task.n_layers,
            'dropout': task.dropout,
            'prenorm': task.prenorm,
            'device': str(task.device),
            'train_data_size': len(task.train_data),
            'val_data_size': len(task.val_data),
            'train_batch_size': task.TRAIN_BATCH_SIZE,
            'val_batch_size': task.VAL_BATCH_SIZE,
            'modeldir': task.modeldir,
            'datadir': task.datadir,
            'start_epoch': task.start_epoch,
            'num_epochs': EPOCHS,
            'optimizer': str(task.optimizer),
            'scheduler': str(task.scheduler),
            'comment': args.comment,
        })

    wandb.finish()
    torch.save(task.model.state_dict(), os.path.join(task.modeldir, f'model.SSM.{task.datatype}.{task.loss}.{timestamp}.path'))

if __name__ == '__main__':
    main()