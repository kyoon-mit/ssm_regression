import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import wandb
from datetime import datetime
from tqdm.auto import tqdm
from models import S4Model
from losses import QuantileLoss

class SSMRegression():
    def __init__(
        self,
        d_input=1,
        d_output=6,
        d_model=6,
        n_layers=4,
        dropout=0.0,
        prenorm=False,
        datatype='SineGaussian', # 'SineGaussian', 'SHO', or 'LIGO'
        loss='NLLGaussian', # 'NLLGaussian', 'Quantile'
    ):
        self.d_input = d_input # number of channels (here only one time series -> 1)
        self.d_output = d_output # number of outputs (here regression, so one output, can be several, if we want to regress several quantities)
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.prenorm = prenorm
        self.datatype = datatype
        self.loss = loss

        self.datadir = f'/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/{self.datatype}'
        self.modeldir = f'/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/{self.datatype}/models'

        # Load datasets
        if datatype=='SineGaussian':
            from data_sinegaussian import DataGenerator
        elif datatype=='SHO':
            from data_sho import DataGenerator
        elif datatype=='LIGO':
            pass # TODO: implement LIGO data loading

        self.train_dict = torch.load(os.path.join(self.datadir, 'train.pt'))
        self.val_dict = torch.load(os.path.join(self.datadir, 'val.pt'))

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

        self.device = None
        self.model = None

        self.optimizer, self.scheduler = None, None

        self.best_loss = None
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.doc_loss = []
        self.doc_val = []

    def reshaping(self, batch, input_dim=1, output_dim=2):
        theta_u, theta_s, data_u, data_s = batch

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
        print('==> Building model..')
        # Define the model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = S4Model(d_input=self.d_input, d_output=self.d_output, d_model=self.d_model,
                        n_layers=self.n_layers, dropout=self.dropout, prenorm=self.prenorm)
        self.model = self.model.to(self.device)
        print('...done!')

        # Count parameters
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Total trainable parameters: {params}')
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
            print(' | '.join([
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
                'sigma': torch.exp(outputs[:,2:4]).clamp(min=1e-7), # uncertainties on the two parameters (model outputs log values)
            }
        elif self.loss=='Quantile':
            return {
                'mean': outputs[:,:2], # mean predictions on the two parameters
                'q25':  outputs[:,2:4],
                'q75':  outputs[:,4:6],
            }
        else: raise ValueError(f"Unknown loss type: {self.loss}. Supported losses are 'NLLGaussian' and 'Quantile'.")

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
        else: raise ValueError(f"Unknown loss type: {self.loss}. Supported losses are 'NLLGaussian' and 'Quantile'.")
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

        # Save checkpoint.
        if checkpoint:
            if eval_loss < self.best_loss:
                state = {
                    'model': self.model.state_dict(),
                    'loss': eval_loss,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.pth')
                self.best_loss = loss
            return loss

if __name__=='__main__':

    task = SSMRegression(d_output=4, datatype='SHO', loss='NLLGaussian')
    task.build_model()
    task.setup_optimizer(lr=0.001, weight_decay=0.01)

    timestamp = datetime.now().strftime('%y%m%d%H%M%S')

    wandb.init(project='ssm_uncertainty_sho_regression', name=f'{timestamp}_regression.py')

    print('Start training...')

    EPOCHS = 200

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
        })

    wandb.finish()

    torch.save(task.model.state_dict(), os.path.join(task.modeldir, f'model.SSM.{task.datatype}.{task.loss}.{timestamp}.path'))