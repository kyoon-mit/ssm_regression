import os
from s4d import S4D
import torch.nn as nn
dropout_fn = nn.Dropout2d
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import wandb
from datetime import datetime

d_input = 1 # number of channels (here only one time series -> 1)
d_output = 2 # number of outputs (here regression, so one output, can be several, if we want to regress several quantities)

# datadir = '/n/holystore01/LABS/iaifi_lab/Users/creissel/SHO/'
# modeldir = '/n/holystore01/LABS/iaifi_lab/Users/creissel/SHO/models/'
datadir = '/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SHO'
modeldir = '/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SHO/models'

# Load datasets
from data import DataGenerator

train_dict = torch.load(os.path.join(datadir, 'train.pt'))
val_dict = torch.load(os.path.join(datadir, 'val.pt'))

train_data = DataGenerator(train_dict)
val_data = DataGenerator(val_dict)

TRAIN_BATCH_SIZE = 1000
VAL_BATCH_SIZE = 1000

train_data_loader = DataLoader(
    train_data, batch_size=TRAIN_BATCH_SIZE,
    shuffle=True
)
val_data_loader = DataLoader(
    val_data, batch_size=VAL_BATCH_SIZE,
    shuffle=True
)

def reshaping(batch):
    theta_u, theta_s, data_u, data_s = batch

    # remove repeat (take only first repeat for unshifted data)
    inputs = data_u[:, 0, :].unsqueeze(-1) if data_u.ndim == 3 else data_u.unsqueeze(-1)  # [B, 200, 1]
    targets = theta_u[:, 0, :2] if theta_u.ndim == 3 else theta_u[:, :2]  # [B, 2]

    return inputs, targets

# definition of SSM here
class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x

# Model
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Building model..')
model = S4Model(d_input=d_input, d_output=d_output, d_model=6, n_layers=4, dropout=0.0, prenorm=False)
model = model.to(device)
print('...done!')

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

import torch.optim as optim
def setup_optimizer(model, lr, weight_decay, epochs):
    all_parameters = list(model.parameters())
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    #optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))
    return optimizer, scheduler

criterion = nn.MSELoss()
optimizer, scheduler = setup_optimizer(model, lr=0.01, weight_decay=0.01, epochs=10)

best_loss = np.inf  # best test loss
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
doc_loss = []
doc_val = []

# Training
from tqdm.auto import tqdm
def train():
    model.train()
    train_loss = 0
    pbar = tqdm(enumerate(train_data_loader))
    for batch_idx, vals in pbar:
        inputs, targets = reshaping(vals)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f ' %
            (batch_idx, len(train_data_loader), train_loss/(batch_idx+1))
        )
        doc_loss.append(train_loss/(batch_idx+1))
    return train_loss / len(train_data_loader)

def eval(epoch, dataloader, checkpoint=False):
    global best_loss
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, vals in pbar:
            inputs, targets = reshaping(vals)
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1))
            )
            doc_val.append(eval_loss/(batch_idx+1))
    return eval_loss / len(dataloader)

    # Save checkpoint.
    if checkpoint:
        if loss < best_loss:
            state = {
                'model': model.state_dict(),
                'loss': loss,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_loss = loss
        return loss

if __name__=='__main__':

    timestamp = datetime.now().strftime('%y%m%d%H%M%S')

    wandb.init(project='ssm_regression', name=f'{timestamp}_regression.py')

    print('Start training...')

    EPOCHS = 200

    pbar = tqdm(range(start_epoch, EPOCHS))
    for epoch_number in pbar:
        avg_train_loss = train()
        avg_val_loss = eval(epoch_number, val_data_loader, checkpoint=True)
        if epoch_number == 0:
            pbar.set_description('Epoch: %d' % (epoch_number))
        else:
            pbar.set_description('Epoch: %d | Val loss: %1.3f' % (epoch_number, avg_val_loss))
        #eval(epoch, test_loader)
        scheduler.step()

        wandb.log({
            'epoch': epoch_number,
            'train_loss': avg_train_loss,
            'val_accuracy': avg_val_loss,
        })

    wandb.finish()

    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), os.path.join(modeldir, f'model.SSM.{timestr}.path'))