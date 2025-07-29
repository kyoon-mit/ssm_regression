from torch import nn
from s4d import S4D
import lightning as L

class LitS4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        loss:str,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        lr=0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        dropout_fn = nn.Dropout1d
        self.encoder = nn.Linear(d_input, d_model)
        self.decoder = nn.Linear(d_model, d_output)
        # Stack S4 layers as residual blocks
        self.s4_layers, self.norms, self.dropouts =\
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        if loss=='NLLGaussian':
            if not (d_output % 2 == 0): raise ValueError(f'If {loss=}, d_output must be an even number.')
            from losses import NLLGaussianUncertainties
            self.criterion = NLLGaussianUncertainties()
        else:
            raise ValueError(f'Invalid option for {loss=}.')

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.hparams.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.hparams.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        if self.hparams.loss=='NLLGaussian':
            mid_idx = int(self.d_output/2)
            x_uncertainties = F.softplus(x[..., mid_idx:])
            x = torch.cat([x[..., :mid_idx], x_uncertainties], dim=-1)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optim_lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.criterion(y_hat, y)
        self.log("train/loss",
                loss,
                on_step=False,
                on_epoch=True,
                reduce_fx='mean',
                logger=True,
                prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.criterion(y_hat, y)
        self.log("val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                reduce_fx='mean',
                logger=True,
                prog_bar=True)
        return loss