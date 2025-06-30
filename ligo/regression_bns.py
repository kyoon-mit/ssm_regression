import logging
import os, sys
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

sys.path.append('/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/modules')
from models import S4Model
from losses import QuantileLoss

logger = logging.getLogger('ssm_regression')

class SSMRegression():
    def __raise__(self, msg, exc_type=Exception):
        logging.error(msg)
        raise exc_type(msg)

    def __init__(
        self,
        d_input=1,
        d_model=6,
        n_layers=4,
        dropout=0.0,
        train_batch_size=10,
        val_batch_size=10,
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
            from data_bns import DataGenerator, get_dataloaders
        else:
            self.__raise__(f'Unsupported datatype: {datatype}', exc_type=ValueError)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f"Using device={self.device}")

        self.d_input = d_input # number of channels (here only one time series -> 1)
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.prenorm = prenorm
        self.datatype = datatype
        self.loss = loss

        self.datadir = f'/ceph/submit/data/user/k/kyoon/KYoonStudy/models/{self.datatype}'
        self.modeldir = os.path.join(self.datadir, 'output')

        self.TRAIN_BATCH_SIZE = train_batch_size
        self.VAL_BATCH_SIZE = val_batch_size

        self.train_data_loader, self.val_data_loader, _ = get_dataloaders(
            hdf5_path=hdf5_path,
            train_batch_size=self.TRAIN_BATCH_SIZE,
            val_batch_size=self.VAL_BATCH_SIZE,
            train_split=0.8,
            test_split=0.1,
            split_indices_file='',  # No precomputed indices file
            random_seed=42
        )

        self.model = None
        self.optimizer, self.scheduler = None, None

    def build_model(self):
        """
        Build the S4 model for regression.
        """
        logger.info('==> Building S4 model...')
        # Define the model
        self.model = S4Model(d_input=self.d_input, loss=self.loss, d_model=self.d_model,
                        n_layers=self.n_layers, dropout=self.dropout, prenorm=self.prenorm)
        self.model = self.model.to(self.device)
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

    def split_outputs(self, outputs):
        """
        If loss is 'NLLGaussian', outputs are of shape (B, 4) with mean and uncertainties.
        If loss is 'Quantile', outputs are of shape (B, 6) with mean, q25, and q75.
        Split model output of shape (B, 6) into mean, 25% quantile, and 75% quantile predictions.
        Returns a dict with 'mean', 'q25', and 'q75' tensors each of shape (B, 2).
        """
        if self.loss == 'NLLGaussian':
            return {
                'mean': outputs[:,:2], # mean predictions on the two parameters
                'sigma': outputs[:,2:4], # uncertainties on the two parameters
            }
        elif self.loss=='Quantile':
            return {
                'mean': outputs[:,:2], # mean predictions on the two parameters
                'q25':  outputs[:,2:4],
                'q75':  outputs[:,4:6],
            }
        else:
            msg = f"Unknown loss type: {self.loss}. Supported losses are 'NLLGaussian' and 'Quantile'."
            logger.error(msg)
            raise ValueError(msg)

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
            msg = f"Unknown loss type: {self.loss}. Supported losses are 'NLLGaussian' and 'Quantile'."
            logger.error(msg)
            raise ValueError(msg)
        return loss_fn

    # Training
    def train(self):
        self.model.train()
        train_loss = 0
        pbar = tqdm(enumerate(self.train_data_loader))
        for batch_idx, vals in pbar:
            h1, l1, params, idx = vals
            mass_1 = params['mass_1'].to(self.device)
            mass_2 = params['mass_2'].to(self.device)
            chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
            inputs = h1.unsqueeze(-1).to(self.device)
            targets = torch.stack([mass_1, chirp_mass], dim=1)
            self.optimizer.zero_grad()

            preds = self.model(inputs)
            outputs = self.split_outputs(preds)
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f ' %
                (batch_idx, len(self.train_data_loader), train_loss/(batch_idx+1))
            )
            self.doc_loss.append(train_loss/(batch_idx+1))
        return train_loss / len(self.train_data_loader)

    def eval(self, epoch, checkpoint=False):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_data_loader))
            for batch_idx, vals in pbar:
                h1, l1, params, idx = vals
                mass_1 = params['mass_1'].to(self.device)
                mass_2 = params['mass_2'].to(self.device)
                chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
                inputs = h1.unsqueeze(-1).to(self.device)
                targets = torch.stack([mass_1, chirp_mass], dim=1)
                
                preds = self.model(inputs)
                outputs = self.split_outputs(preds)
                # Compute loss
                loss = self.compute_loss(outputs, targets)

                eval_loss += loss.item()

                pbar.set_description(
                    'Batch Idx: (%d/%d) | Loss: %.3f' %
                    (batch_idx, len(self.val_data_loader), eval_loss/(batch_idx+1))
                )
                self.doc_val.append(eval_loss/(batch_idx+1))
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

    wandb.init(project=f'ssm_{task.datatype}_regression', name=f'ssm_{task.datatype}_{timestamp}')

    EPOCHS = 100

    pbar = tqdm(range(task.start_epoch, EPOCHS))
    for epoch_number in pbar:
        avg_train_loss = task.train()
        avg_val_loss = task.eval(epoch_number, checkpoint=True)
        if epoch_number == 0:
            pbar.set_description('Epoch: %d' % (epoch_number))
        else:
            pbar.set_description('Epoch: %d | Val loss: %1.3f' % (epoch_number, avg_val_loss))
        #eval(epoch, test_loader)
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
            'wandb_run_id': wandb.run.id,
        })

    wandb.finish()

    torch.save(task.model.state_dict(), os.path.join(task.modeldir, f'model.SSM.{task.datatype}.{task.loss}.{timestamp}.path'))