import os, sys
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

class SSMRegression(L.LightningModule):
    def __init__(
        self,
        d_input=2,
        d_output=18,
        d_model=6,
        n_layers=4,
        downsample_factor=1,
        duration=64,
        scale_factor=1.,
        normalize=False,
        dropout=0.0,
        train_batch_size=100,
        val_batch_size=100,
        prenorm=False,
        device=None,
    ):
    super().__init()__

    def training_step(self, batch, batch_idx):