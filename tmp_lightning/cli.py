import os, sys
import torch
from torch import nn
import torch.nn.functional as F
from lightning.pytorch.cli import LightningCLI

if __name__=='__main__':
    torch.set_float32_matmul_precision('medium')
    torch.cuda.memory_summary(device=None, abbreviated=False)
    torch.cuda.empty_cache()
    LightningCLI(save_config_callback=None)