import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform, RandomPermutation

import wandb
from datetime import datetime

import nflows.utils as torchutils
from models import SimilarityEmbedding

# datadir = '/n/holystore01/LABS/iaifi_lab/Users/creissel/SineGaussian/'
# pretraining = '/n/holystore01/LABS/iaifi_lab/Users/creissel/SineGaussian/models/model.CNN.20250408-111215.path'
# modeldir = '/n/holystore01/LABS/iaifi_lab/Users/creissel/SineGaussian/models/'
datadir = '/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SineGaussian'
modeldir = '/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SineGaussian/models'
pretraining = '/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SineGaussian/models/model.CNN.SineGaussian.20250527-232830.path'

num_transforms = 5
num_blocks = 4
hidden_features = 30
context_features = 3 # needs to fit the pretraining embedding dimensionality
num_points = 200 # length of time series
num_repeats = 10 # number of augmentations

timestamp = datetime.now().strftime('%y%m%d%H%M%S')

wandb.init(project='ssm_sg_regression', name=f'{timestamp}_parameter_estimation.py')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

# Load datasets
from data_sinegaussian import DataGenerator

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

# define model
class EmbeddingNet(nn.Module):
    """Wrapper around the similarity embedding defined above"""
    def __init__(self, pretraining, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.representation_net = SimilarityEmbedding(num_hidden_layers_h=2)
        self.representation_net.load_state_dict(torch.load(pretraining, map_location=device))

        # the expander network is unused and hence don't track gradients
        for name, param in self.representation_net.named_parameters():
            if 'expander_layer' in name or 'layers_h' in name or 'final_layer' in name:
                param.requires_grad = False

        # freeze part of the conv layer of embedding_net
        for name, param in self.representation_net.named_parameters():
            if 'layers_f.blocks.0' in name or 'layers_f.blocks.1' in name:
                param.requires_grad = False

        self.context_layer = nn.Identity()

    def forward(self, x):
        batch_size, _, dims = x.shape
        x = x.reshape(batch_size, 1, dims).repeat(1, num_repeats, 1)
        _, rep = self.representation_net(x)
        return self.context_layer(rep.reshape(batch_size, context_features))

base_dist = StandardNormal([2])
transforms = []
for _ in range(num_transforms):
    block = [
        MaskedAffineAutoregressiveTransform(
            features=2,  # 2-dim posterior
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            activation=torch.tanh,
            use_batch_norm=False,
            use_residual_blocks=True,
            dropout_probability=0.01,
        ),
        RandomPermutation(features=2)
    ]
    transforms += block

transform = CompositeTransform(transforms)

embedding_net = EmbeddingNet(pretraining)

flow = Flow(transform, base_dist, embedding_net).to(device=device)

# print number of parameters
print('Total number of NOT fixed weights in embedding net', sum(p.numel() for p in flow._embedding_net.parameters() if p.requires_grad))
print("Total number of trainable parameters: ", sum(p.numel() for p in flow.parameters() if p.requires_grad))

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for idx, val in enumerate(train_data_loader, 1):
        augmented_theta, _, augmented_data, _ = val
        augmented_theta = augmented_theta[...,0:2]

        theta = augmented_theta.reshape(-1, 2)
        data = augmented_data.reshape(-1, 1, num_points)

        flow_loss = -flow.log_prob(theta, context=data).mean()

        optimizer.zero_grad()
        flow_loss.backward()
        optimizer.step()

        running_loss += flow_loss.item()
        if idx % 10 == 0:
            last_loss = running_loss / 10 # avg loss
            print(' Avg. train loss/batch after {} batches = {:.4f}'.format(idx, last_loss))
            tb_x = epoch_index * len(train_data_loader) + idx
            running_loss = 0.
    return last_loss


def val_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for idx, val in enumerate(val_data_loader, 1):
        augmented_theta, _, augmented_data, _ = val
        augmented_theta = augmented_theta[...,0:2]

        theta = augmented_theta.reshape(-1, 2)
        data = augmented_data.reshape(-1, 1, num_points)

        flow_loss = -flow.log_prob(theta, context=data).mean()
        loss = flow_loss.item()

        running_loss += flow_loss.item()
        if idx % 5 == 0:
            last_loss = running_loss / 5
            tb_x = epoch_index * len(val_data_loader) + idx + 1

            running_loss = 0.
    return last_loss

if __name__=='__main__':

    optimizer = optim.SGD(flow.parameters(), lr=1e-4, momentum=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, threshold=0.01)

    EPOCHS = 200

    for epoch_number in range(EPOCHS):
        print(f'EPOCH {epoch_number + 1}')
        # Gradient tracking
        flow.train(True)
        avg_train_loss = train_one_epoch(epoch_number)

        # no gradient tracking, for validation
        flow.train(False)
        avg_val_loss = val_one_epoch(epoch_number)

        print(f"Train/Val flow Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

        wandb.log({
            'epoch': epoch_number,
            'train_loss': avg_train_loss,
            'val_accuracy': avg_val_loss,
        })

        for param_group in optimizer.param_groups:
            print("Current LR = {:.3e}".format(param_group['lr']))
        epoch_number += 1
        try:
            scheduler.step(avg_val_loss)
        except TypeError:
            scheduler.step()

    wandb.finish()

    torch.save(flow.state_dict(), os.path.join(modeldir, f'flow.CNN.{timestamp}.path'))
