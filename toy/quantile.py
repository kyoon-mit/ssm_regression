# An implementation of quantile regression (arXiv:2505.18311) on toy data

import logging
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from models import QRModel
from losses import MultiQuantileLoss

logger = logging.getLogger('quantile_regression')

class QuantileRegression():
    def __init__(
        self,
        d_input=1
    ):
        pass