import logging

import os, sys
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform, RandomPermutation

import wandb

sys.path.append('/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/modules')
from utils import extract_timestamp
from models import EmbeddingNet

logger = logging.getLogger('flow')

class NormalizingFlow():
    def __init__(
            self,
            datatype='SHO',
            embed_model='', # Path to the pretraining model
            datasfx='', # suffix for the dataset, e.g., '_sigma0.4_gaussian'
            batch_size=1000,
            num_transforms=5,
            num_blocks=4,
            hidden_features=30,
            context_features=3,  # needs to fit the pretraining embedding dimensionality
            num_points=200,  # length of time series
            num_repeats=10,  # number of augmentations
            device=None,
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

        self.datatype = datatype
        self.datadir = f'/ceph/submit/data/user/k/kyoon/KYoonStudy/models/{self.datatype}'
        self.modeldir = os.path.join(self.datadir, 'output')
        if not embed_model:
            raise ValueError("Pretraining path must be provided.")
        self.pretraining = embed_model

        self.embed_timestamp = extract_timestamp(embed_model, sep='')
        self.num_transforms = num_transforms
        self.num_blocks = num_blocks
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.num_points = num_points
        self.num_repeats = num_repeats

        self.train_dict = torch.load(os.path.join(self.datadir, f'train{datasfx}.pt'),
                                     map_location=self.device, weights_only=True)
        self.val_dict = torch.load(os.path.join(self.datadir, f'val{datasfx}.pt'),
                                     map_location=self.device, weights_only=True)
        self.train_data = DataGenerator(self.train_dict)
        self.val_data = DataGenerator(self.val_dict)
        self.TRAIN_BATCH_SIZE = batch_size
        self.VAL_BATCH_SIZE = batch_size
        self.train_data_loader = DataLoader(
            self.train_data, batch_size=self.TRAIN_BATCH_SIZE,
            shuffle=True
        )
        self.val_data_loader = DataLoader(
            self.val_data, batch_size=self.VAL_BATCH_SIZE,
            shuffle=True
        )
        self.flow = nn.Module()
        self.optimizer = None
        self.scheduler = None

    def build_flow(self):
        print('Building flow.')
        # Define the base distribution
        base_dist = StandardNormal([2])
        transforms = []
        for _ in range(self.num_transforms):
            block = [
                MaskedAffineAutoregressiveTransform(
                    features=2,  # 2-dim posterior
                    hidden_features=self.hidden_features,
                    context_features=self.context_features,
                    num_blocks=self.num_blocks,
                    activation=torch.tanh,
                    use_batch_norm=False,
                    use_residual_blocks=True,
                    dropout_probability=0.01,
                ),
                RandomPermutation(features=2)
            ]
            transforms += block
        transform = CompositeTransform(transforms)
        embedding_net = EmbeddingNet(self.pretraining, device=self.device)
        self.flow = Flow(transform, base_dist, embedding_net).to(device=self.device)
        # print number of parameters
        print('Total number of NOT fixed weights in embedding net', sum(p.numel() for p in self.flow._embedding_net.parameters() if p.requires_grad))
        print("Total number of trainable parameters: ", sum(p.numel() for p in self.flow.parameters() if p.requires_grad))
        self.optimizer = optim.SGD(self.flow.parameters(), lr=1e-4, momentum=0.5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, threshold=0.01)

    def train_one_epoch(self, epoch_index):
        self.flow.train(True) # gradient tracking
        running_loss = 0.
        last_loss = 0.

        for idx, val in enumerate(self.train_data_loader, 1):
            _, augmented_theta, _, augmented_data, _, _, _, _, _, _ = val
            augmented_theta = augmented_theta[...,0:2]

            theta = augmented_theta.reshape(-1, 2)
            data = augmented_data.reshape(-1, 1, self.num_points)

            flow_loss = - self.flow.log_prob(theta, context=data).mean()

            self.optimizer.zero_grad()
            flow_loss.backward()
            self.optimizer.step()

            running_loss += flow_loss.item()
            if idx % 10 == 0:
                last_loss = running_loss / 10 # avg loss
                print(' Avg. train loss/batch after {} batches = {:.4f}'.format(idx, last_loss))
                tb_x = epoch_index * len(self.train_data_loader) + idx
                running_loss = 0.
        return last_loss

    def val_one_epoch(self, epoch_index):
        self.flow.train(False) # no gradient tracking, for validation
        running_loss = 0.
        last_loss = 0.

        for idx, val in enumerate(self.val_data_loader, 1):
            _, augmented_theta, _, augmented_data, _, _, _, _, _, _ = val
            augmented_theta = augmented_theta[...,0:2]

            theta = augmented_theta.reshape(-1, 2)
            data = augmented_data.reshape(-1, 1, self.num_points)

            flow_loss = - self.flow.log_prob(theta, context=data).mean()
            loss = flow_loss.item()

            running_loss += flow_loss.item()
            if idx % 5 == 0:
                last_loss = running_loss / 5
                tb_x = epoch_index * len(self.val_data_loader) + idx + 1

                running_loss = 0.
        return last_loss

if __name__=='__main__':

    embed_model = '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/embedding.CNN.SineGaussian.250717075146.pt'
    datatype = 'SineGaussian'
    datasfx = '_sigma0.4_gaussian'

    task = NormalizingFlow(embed_model=embed_model, datatype=datatype, datasfx=datasfx)
    task.build_flow()
    flow = task.flow

    wandb.init(project=f'flow_{task.datatype}', name=f'flow_{task.datatype}_{task.timestamp}')

    EPOCHS = 200

    for epoch_number in range(EPOCHS):
        print(f'EPOCH {epoch_number + 1}')
        avg_train_loss = task.train_one_epoch(epoch_number)
        avg_val_loss = task.val_one_epoch(epoch_number)

        print(f"Train/Val flow Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

        wandb.log({
            'epoch': epoch_number,
            'train_loss': avg_train_loss,
            'val_accuracy': avg_val_loss,
            'lr': task.optimizer.param_groups[0]['lr'],
            'trainable_params': sum(p.numel() for p in flow.parameters() if p.requires_grad),
            'fixed_params': sum(p.numel() for p in flow._embedding_net.parameters() if not p.requires_grad),
            'total_params': sum(p.numel() for p in flow.parameters())
        })

        for param_group in task.optimizer.param_groups:
            print("Current LR = {:.3e}".format(param_group['lr']))
        epoch_number += 1
        try:
            task.scheduler.step(avg_val_loss)
        except TypeError:
            task.scheduler.step()

    wandb.finish()

    torch.save(flow.state_dict(), os.path.join(task.modeldir, f'flow.CNN.{task.datatype}.{task.embed_timestamp}.pt'))