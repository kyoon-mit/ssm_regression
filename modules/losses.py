import torch
import torch.nn.functional as F
from torch import nn, optim

# implemented from https://github.com/violatingcp/codec
class VICRegLoss_modified(nn.Module):
    def forward(self, x, y, wt_repr=1.0, wt_cov=1.0, wt_std=1.0):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        N = x.size(0)
        D = x.size(1)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        x = (x-x.mean(dim=0))/x.std(dim=0)
        y = (y-y.mean(dim=0))/y.std(dim=0)

        # keep batch dim i.e. 0, unchanged
        #cov_x = (x.transpose(1, 2) @ x) / (N - 1)
        #cov_y = (y.transpose(1, 2) @ y) / (N - 1)
        cov_x = (x.T @ x) / (N - 1)
        cov_y = (y.T @ y) / (N - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D)
        cov_loss += self.off_diagonal(cov_y).pow_(2).sum().div(D)

        s = wt_repr*repr_loss + wt_cov*cov_loss + wt_std*std_loss
        return s, repr_loss, cov_loss, std_loss

    def off_diagonal(self,x):
        num_batch, n = x.shape
        #assert n == m
        # All off diagonal elements from complete batch flattened
        #return x.flatten(start_dim=1)[...,:-1].view(num_batch, n - 1, n + 1)[...,1:].flatten()
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICRegLoss(nn.Module):
    def forward(self, x, y, wt_repr=1.0, wt_cov=1.0, wt_std=1.0):
        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        N = x.size(0)
        D = x.size(1)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        x = (x-x.mean(dim=0))/x.std(dim=0)
        y = (y-y.mean(dim=0))/y.std(dim=0)

        # keep batch dim i.e. 0, unchanged
        cov_x = (x.transpose(1, 2) @ x) / (N - 1)
        cov_y = (y.transpose(1, 2) @ y) / (N - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D)
        cov_loss += self.off_diagonal(cov_y).pow_(2).sum().div(D)

        s = wt_repr*repr_loss + wt_cov*cov_loss + wt_std*std_loss
        return s, repr_loss, cov_loss, std_loss

    def off_diagonal(self,x):
        num_batch, n, m = x.shape
        assert n == m
        # All off diagonal elements from complete batch flattened
        return x.flatten(start_dim=1)[...,:-1].view(num_batch, n - 1, n + 1)[...,1:].flatten()

class QuantileLoss(nn.Module):
    """Quantile Loss for regression tasks.
    This loss is useful when you want to predict a quantile of the target distribution.

    Args:
        quantile (float): The quantile to predict. Default is 0.25 for the first quantile.
        reduction (str): The reduction method to apply to the loss. Default is 'mean'.
            Valid options are 'mean', 'sum', or 'none'.

    Raises:
        ValueError: If the reduction method is not one of the valid options.

    Example:
        >>> loss_fn = QuantileLoss(quantile=0.25, reduction='mean')
        >>> pred = torch.tensor([0.5, 1.0, 1.5])
        >>> target = torch.tensor([0.0, 1.0, 2.0])
        >>> loss = loss_fn(pred, target)
        >>> print(loss)  # Output will be the quantile loss value
    """
    def __init__(self, quantile=0.25, reduction='mean'):
        super().__init__()
        self.quantile = quantile
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose from 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, pred, target):
        z = target - pred
        loss = torch.where(z > 0, self.quantile * z, (self.quantile - 1) * z)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class MultiQuantileLoss(nn.Module):
    """Multi-quantile loss function.

    See (3) of arXiv:2505.18311.
    """
    def __init__(self, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], reduction='mean'):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose from 'mean', 'sum', or 'none'.")
        self.reduction = reduction
    
    def forward(self, pred, target):
        z = target - pred
        loss = torch.where(z > 0, torch.outer(self.quantiles, z), torch.outer((self.quantiles - 1), z))
        loss = loss.mean(dim=0)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


####### Compound loss functions #######

class NLLGaussianUncertainties(nn.Module):
    """NLLGaussian to predict the values and uncertainties of the parameters.

    Args:
        (None)

    Forward Args:
        preds (torch.Tensor): The predicted values.
        targets (torch.Tensor): The truth values.
        variances (torch.Tensor): The uncertainties squared. Must be positive.
    
    Raises:
        ValueError: If the reduction method is not one of the valid options.

    Example:
        >>> loss_fn = NLLGaussianUncertainties()
        >>> preds   = torch.tensor([0.5, 1.0, 1.5])
        >>> targets = torch.tensor([1.0, -1.0, -0.5])
        >>> variances = torch.tensor([0.2, 0.3, 0.15])
        >>> loss    = loss_fn(preds, targets, variances)
        >>> print(loss)  # Output will be the quantile loss value
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.GaussainNLLLoss(reduction='mean', full=False, eps=1e-9)

    def forward(self, preds, targets, variances):
        return self.criterion(preds, targets, variances)

class QuantileUncertainties(nn.Module):
    """Quantile to predict the values and uncertainties of the parameters.
    1 sigma = (q84 - q16)/2

    Args:
        reduction (str): Reduction method.
            Valid options are 'mean' (default), 'sum', or 'none'.

    Forward Args:
        preds_means (torch.Tensor): The predicted mean values.
        preds_q16 (torch.Tensor): The predicted q16 values.
        preds_q84 (torch.Tensor): The predicted q84 values.
        targets (torch.Tensor): The truth values.
    
    Raises:
        ValueError: If the reduction method is not one of the valid options.

    Example:
        >>> loss_fn = QuantileUncertainties()
        >>> loss    = loss_fn(pred_means, pred_q16, pred_q84, targets)
        >>> print(loss)  # Output will be the quantile loss value
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.l_mean = nn.MSELoss(reduction='mean')
        self.l_q16 = QuantileLoss(quantile=0.1587, reduction=reduction)
        self.l_q84 = QuantileLoss(quantile=0.8413, reduction=reduction)

    def forward(self, preds_mean, preds_q16, preds_q84, targets):
        loss_fn = (
            self.l_mean(preds_mean, targets),
            self.l_q16(preds_q16, targets),
            self.l_q84(preds_q84, targets)
        ) / 3
        return loss_fn
        