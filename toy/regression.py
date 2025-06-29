import logging
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from models import S4Model
from losses import QuantileLoss

logger = logging.getLogger('ssm_regression')

class SSMRegression():
    def __init__(
        self,
        d_input=1,
        d_model=6,
        n_layers=4,
        dropout=0.0,
        prenorm=False,
        device=None,
        datatype='SHO', # 'SineGaussian', 'SHO', or 'LIGO'
        datasfx='', # suffix for the dataset, e.g., '_sigma0.4_gaussian'
        loss='NLLGaussian', # 'NLLGaussian', 'Quantile'
    ):
        # Load datasets
        if datatype=='SineGaussian':
            from data_sinegaussian import DataGenerator
        elif datatype=='SHO':
            from data_sho import DataGenerator
        elif datatype=='LIGO':
            pass # TODO: implement LIGO data loading

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

        self.train_dict = torch.load(os.path.join(self.datadir, f'train{datasfx}.pt'), map_location=self.device, weights_only=True)
        self.val_dict = torch.load(os.path.join(self.datadir, f'val{datasfx}.pt'), map_location=self.device, weights_only=True)

        self.train_data = DataGenerator(self.train_dict)
        self.val_data = DataGenerator(self.val_dict)

        self.TRAIN_BATCH_SIZE = 1000
        self.VAL_BATCH_SIZE = 1000

        self.train_data_loader = DataLoader(
            self.train_data, batch_size=self.TRAIN_BATCH_SIZE,
            shuffle=True
        )
        self.val_data_loader = DataLoader(
            self.val_data, batch_size=self.VAL_BATCH_SIZE,
            shuffle=True
        )

        self.model = None
        self.optimizer, self.scheduler = None, None

    def reshaping(self, batch, input_dim=1, output_dim=2):
        theta_u, theta_s, data_u, data_s, \
        data_clean_u, data_noise_u, data_clean_s, data_noise_s, \
        t_vals, event_id = batch

        # remove repeat (take only first repeat for unshifted data)
        if input_dim==1:
            inputs = data_u[:, 0, :].unsqueeze(-1) if data_u.ndim == 3 else data_u.unsqueeze(-1)  # [B, 200, input_dim]
        else:
            inputs = data_u[:, 0, :input_dim] if data_u.ndim == 3 else data_u[:, :input_dim]
        targets = theta_u[:, 0, :output_dim] if theta_u.ndim == 3 else theta_u[:, :output_dim]  # [B, output_dim]

        return inputs, targets

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
            inputs, targets = self.reshaping(vals, output_dim=2)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
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
                inputs, targets = self.reshaping(vals)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
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

    task = SSMRegression(d_model=12, n_layers=8, datatype='SineGaussian', loss='NLLGaussian')
    task.build_model()
    task.setup_optimizer(lr=0.001, weight_decay=0.01)

    timestamp = datetime.now().strftime('%y%m%d%H%M%S')

    wandb.init(project=f'ssm_{task.datatype}_regression', name=f'ssm_{task.datatype}_{timestamp}')

    EPOCHS = 240

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