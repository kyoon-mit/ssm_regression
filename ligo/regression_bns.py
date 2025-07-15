import logging
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import sys
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent, 'modules'))

from models import S4Model
from losses import QuantileLoss

import psutil, os

logger = logging.getLogger('ssm_regression')

class SSMRegression():
    def __raise__(self, msg, exc_type=Exception):
        logging.error(msg)
        raise exc_type(msg)

    def __init__(
        self,
        d_input=2,
        d_output=18,
        d_model=6,
        n_layers=4,
        downsample_factor=1,
        duration=64,
        scale_factor=1.,
        dropout=0.0,
        train_batch_size=100,
        val_batch_size=100,
        prenorm=False,
        device=None,
        datatype='BNS', # 'BNS'
        hdf5_path='', # path to the HDF5 dataset
        split_indices_file='', # path to precomputed split indices file
        # If split_indices_file is empty, it will compute indices on the fly
        # and save them to 'bns_data_indices.npz' in the current directory.
        # If you want to use precomputed indices, provide the path to the .npz
        # file containing 'train_indices', 'val_indices', and 'test_indices'.
        # If you want to use the default split, set split_indices_file to ''.
        loss='NLLGaussian', # 'NLLGaussian', 'Quantile'
    ):
        # Load datasets
        if datatype=='BNS':
            from data_bns import get_dataloaders
        else:
            self.__raise__(f'Unsupported datatype: {datatype}', exc_type=ValueError)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f"Using device={self.device}")

        self.d_input = d_input # number of channels
        self.d_output = d_output
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.prenorm = prenorm
        self.datatype = datatype
        self.loss = loss

        if loss=='NLLGaussian':
            self.n_output = int(d_output // 2)
        elif loss=='Quantile':
            self.n_output = int(d_output // 3)
        else:
            self.__raise__(f'Unknown loss type: {self.loss}. Supported losses are "NLLGaussian" and "Quantile".', ValueError)

        self.datadir = os.path.join(Path(__file__).resolve().parent.parent.parent, 'models', self.datatype)
        self.modeldir = os.path.join(self.datadir, 'output')
        if not os.path.exists(self.modeldir):
            os.makedirs(self.modeldir)
            logger.info(f'Created model directory: {self.modeldir}')

        self.TRAIN_BATCH_SIZE = train_batch_size
        self.VAL_BATCH_SIZE = val_batch_size
        train_split, test_split = 0.8, 0.1
        random_seed = 42

        self.train_data_loader, self.val_data_loader, _ = get_dataloaders(
            hdf5_path=hdf5_path,
            downsample_factor=downsample_factor,
            duration=duration,
            scale_factor=scale_factor,
            train_batch_size=self.TRAIN_BATCH_SIZE,
            val_batch_size=self.VAL_BATCH_SIZE,
            train_split=train_split,
            test_split=test_split,
            split_indices_file=split_indices_file,
            random_seed=random_seed
        )

        self.model = None
        self.optimizer, self.scheduler = None, None

        # Create a dummy tensor to ensure CUDA is initialized if using GPU
        if self.device.type == 'cuda':
            torch.tensor([0.0], device=self.device)
            torch.cuda.empty_cache()

        logger.info(f'''
        d_input={self.d_input},
        d_output={self.d_output},
        d_model={self.d_model},
        n_layers={self.n_layers},
        downsample_factor={downsample_factor},
        duration={duration},
        scale_factor={scale_factor},
        dropout={self.dropout},
        train_batch_size={self.TRAIN_BATCH_SIZE},
        val_batch_size={self.VAL_BATCH_SIZE},
        prenorm={self.prenorm},
        device={self.device},
        datatype={self.datatype},
        hdf5_path={hdf5_path},
        split_indices_file={split_indices_file},
        loss='{self.loss},
        train_split={train_split},
        test_split={test_split},
        random_seed={random_seed}
        ''')

    def build_model(self):
        """
        Build the S4 model for regression.
        """
        logger.info('==> Building S4 model...')
        # Define the model
        self.model = S4Model(d_input=self.d_input, d_output=self.d_output, loss=self.loss,
                             d_model=self.d_model, n_layers=self.n_layers, dropout=self.dropout,
                             prenorm=self.prenorm)
        self.model = self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            logger.info(f'Using {torch.cuda.device_count()} GPUs')
            self.model = nn.DataParallel(self.model)
        # Requires setup with torch.distributed.launch or torchrun
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        logger.info('...done!')

        # Count parameters
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info(f'Total trainable parameters: {params}')
        return

    def setup_optimizer(self, lr=0.01, weight_decay=0.01, epochs=10):
        all_parameters = list(self.model.parameters())
        params = [p for p in all_parameters if not hasattr(p, "_optim")]
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None)==hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            logger.info(' | '.join([
                f"Optimizer group {i}",
                f"{len(g['params'])} tensors",
            ] + [f"{k} {v}" for k, v in group_hps.items()]))

        self.optimizer, self.scheduler = optimizer, scheduler

        self.best_loss = np.inf  # best test loss
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.doc_loss = []
        self.doc_val = []

    def split_outputs(self, n_output, outputs):
        """
        If loss is 'NLLGaussian', outputs are of shape (B, 2*n_output) with mean and uncertainties.
        If loss is 'Quantile', outputs are of shape (B, 3*n_output) with mean, q25, and q75.
        """
        if self.loss == 'NLLGaussian':
            return {
                'mean': outputs[:,:n_output],
                'sigma': outputs[:,n_output:2*n_output],
            }
        elif self.loss=='Quantile':
            return {
                'mean': outputs[:,:n_output],
                'q25':  outputs[:,n_output:n_output*2],
                'q75':  outputs[:,n_output*2:3*n_output],
            }
        else:
            self.__raise__(f'Unknown loss type: {self.loss}. Supported losses are "NLLGaussian" and "Quantile".', ValueError)

    # Define loss function
    def compute_loss(self, outputs, targets):
        """
        Define the loss function based on the specified loss type.
        Supported losses are 'NLLGaussian' and 'Quantile'.
        """
        if self.loss == 'NLLGaussian':
            criterion = nn.GaussianNLLLoss(reduction='mean', full=False, eps=1e-7)
            loss_fn = criterion(outputs['mean'], targets, outputs['sigma'])
        elif self.loss == 'Quantile':
            mean_loss = nn.MSELoss(reduction='mean')
            q25_loss = QuantileLoss(quantile=0.25)
            q75_loss = QuantileLoss(quantile=0.75)
            loss_fn = (
                mean_loss(outputs['mean'], targets) +
                q25_loss(outputs['q25'], targets) +
                q75_loss(outputs['q75'], targets)
            )
        else:
           self.__raise__(f'Unknown loss type: {self.loss}. Supported losses are "NLLGaussian" and "Quantile".', ValueError)
        return loss_fn

    def stack_inputs_targets(self, vals):
        h1, l1, params, idx = vals
        mass_1 = params['mass_1'].to(self.device)
        mass_2 = params['mass_2'].to(self.device)
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
        # mass_ratio = mass_2 / mass_1
        # total_mass = mass_1 + mass_2
        # right_ascension = params['ra'].to(self.device)
        # declination = params['dec'].to(self.device)
        # redshift = params['redshift'].to(self.device)
        # theta_jn = params['theta_jn'].to(self.device)

        inputs = torch.stack([h1.to(self.device), l1.to(self.device)], dim=2)
        targets = torch.stack([chirp_mass], dim=1)
        # targets = torch.stack([mass_1, mass_2], dim=1)
        # targets = torch.stack([mass_1, mass_2,
        #                        chirp_mass, mass_ratio, total_mass,
        #                        right_ascension, declination, redshift, theta_jn], dim=1)
        return inputs, targets

    # Training
    def train(self):
        self.model.train()
        train_loss = 0
        pbar = tqdm(enumerate(self.train_data_loader))
        for batch_idx, vals in pbar:
            process = psutil.Process(os.getpid())
            logger.info(f'Memory usage in training (MB): {process.memory_info().rss / 1024 / 1024:.2f}')
            inputs, targets = self.stack_inputs_targets(vals)
            self.optimizer.zero_grad()

            preds = self.model(inputs)
            outputs = self.split_outputs(n_output=self.n_output, outputs=preds)
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f ' %
                (batch_idx, len(self.train_data_loader), train_loss/(batch_idx+1))
            )
            self.doc_loss.append(train_loss/(batch_idx+1))
            torch.cuda.empty_cache()
        return train_loss / len(self.train_data_loader)

    def eval(self, epoch, checkpoint=False):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_data_loader))
            for batch_idx, vals in pbar:
                process = psutil.Process(os.getpid())
                logger.info(f'Memory usage in training (MB): {process.memory_info().rss / 1024 / 1024:.2f}')
                inputs, targets = self.stack_inputs_targets(vals)
                preds = self.model(inputs)
                outputs = self.split_outputs(n_output=self.n_output, outputs=preds)

                # Compute loss
                loss = self.compute_loss(outputs, targets)
                eval_loss += loss.item()

                pbar.set_description(
                    'Batch Idx: (%d/%d) | Loss: %.3f' %
                    (batch_idx, len(self.val_data_loader), eval_loss/(batch_idx+1))
                )
                self.doc_val.append(eval_loss/(batch_idx+1))
                torch.cuda.empty_cache()
        return eval_loss / len(self.val_data_loader)


if __name__=='__main__':
    import wandb
    from datetime import datetime

    task = SSMRegression(d_model=6, n_layers=4, datatype='BNS', loss='NLLGaussian',
                         train_batch_size=100, val_batch_size=100,
                         device='cuda',
                         hdf5_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/models/BNS/bns_waveforms.hdf5',
                         split_indices_file='/ceph/submit/data/user/k/kyoon/KYoonStudy/models/BNS/bns_data_indices.npz')  # Use precomputed indices file
    task.build_model()
    task.setup_optimizer(lr=0.001, weight_decay=0.01)

    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    logger.info(f'{timestamp=}')
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
            'str_timestamp': timestamp, # timestamp in str format
            'datatype': task.datatype,
            'loss': task.loss,
            'd_input': task.d_input,
            'd_output': task.d_output,
            'd_model': task.d_model,
            'n_layers': task.n_layers,
            'dropout': task.dropout,
            'prenorm': task.prenorm,
            'device': str(task.device),
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
    save_model_name = os.path.join(task.modeldir, f'model.SSM.{task.datatype}.{task.loss}.d{task.d_model}.n{task.n_layers}.{timestamp}.path')
    logger.info(f'Saving to {save_model_name}.')
    torch.save(task.model.state_dict(), save_model_name)