from s4d import S4D
#from s4 import S4Block as S4
from torch import nn, optim
import torch
import torch.nn.functional as F

num_dim = 3 # dimensionality of embedding space
num_repeats = 10 # number of augmentations
num_points = 200 # length of time series

class ConvResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=5,
        activation=F.relu,
        dropout_probability=0.1,
        use_batch_norm=True,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(channels, eps=1e-3) for _ in range(2)]
            )
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size=kernel_size, padding='same') for _ in range(2)] #2 is for 2 conv layers
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            nn.init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        return inputs + temps

class ConvResidualNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_blocks=2,
        kernel_size=5,
        activation=F.relu,
        dropout_probability=0.1,
        use_batch_norm=True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.initial_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding='same',
        )
        self.blocks = nn.ModuleList(
            [
                ConvResidualBlock(
                    channels=hidden_channels,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    kernel_size=kernel_size,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Conv1d(
            hidden_channels, out_channels, kernel_size=1, padding='same'
        )

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs

# SimilarityEmbedder Model (backbone: CNN) 
class SimilarityEmbedding(nn.Module):
    """Simple Dense embedding"""
    def __init__(self, num_hidden_layers_h=1, num_points=num_points, activation=torch.relu):
        super().__init__()
        self.num_hidden_layers_h = num_hidden_layers_h

        self.layers_f = ConvResidualNet(in_channels=num_repeats, out_channels=1,
                                        hidden_channels=20, num_blocks=4,
                                        kernel_size=21)
        self.contraction_layer = nn.Sequential(
            nn.Linear(num_points, 100),
            nn.ReLU(),
            nn.Linear(100, num_dim)
        )
        self.expander_layer = nn.Linear(num_dim, 20)
        self.layers_h = nn.ModuleList([nn.Linear(20, 20) for _ in range(num_hidden_layers_h)])
        self.final_layer = nn.Linear(20, 6)

        self.activation = activation

    def forward(self, x):
        x = self.layers_f(x)
        x = self.contraction_layer(x)
        representation = torch.clone(x) #copy
        x = self.activation(self.expander_layer(x))
        for layer in self.layers_h:
            x = layer(x)
            x = self.activation(x) #activation of layer(x) in layers_h
        x = self.final_layer(x)
        return x, representation
    
# definition of EmbeddingNet
class EmbeddingNet(nn.Module):
    """Wrapper around the similarity embedding defined above"""
    def __init__(self,
                 pretraining,
                 num_hidden_layers_h=2,
                 context_features=3,  # needs to fit the pretraining embedding dimensionality
                 num_repeats=10,  # number of augmentations
                 device=None,
                 *args,
                 **kwargs
    ):    
        super().__init__(*args, **kwargs)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.representation_net = SimilarityEmbedding(num_hidden_layers_h=num_hidden_layers_h)
        self.representation_net.load_state_dict(torch.load(pretraining, map_location=device, weights_only=True))

        # the expander network is unused and hence don't track gradients
        for name, param in self.representation_net.named_parameters():
            if 'expander_layer' in name or 'layers_h' in name or 'final_layer' in name:
                param.requires_grad = False

        # freeze part of the conv layer of embedding_net
        for name, param in self.representation_net.named_parameters():
            if 'layers_f.blocks.0' in name or 'layers_f.blocks.1' in name:
                param.requires_grad = False

        self.context_layer = nn.Identity()
        self.context_features = context_features
        self.num_repeats = num_repeats

    def forward(self, x):
        batch_size, _, dims = x.shape
        x = x.reshape(batch_size, 1, dims).repeat(1, self.num_repeats, 1)
        _, rep = self.representation_net(x)
        return self.context_layer(rep.reshape(batch_size, self.context_features))

# definition of SSM here
class S4Model(nn.Module):
    def __init__(
        self,
        d_input,
        d_output,
        loss:str,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()
        dropout_fn = nn.Dropout1d
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
        self.d_output = d_output
        self.loss=loss
        if loss=='NLLGaussian' and (d_output % 2 != 0):
            raise ValueError(f'If {loss=}, d_output must be an even number.')
        # Linear decoder
        self.decoder = nn.Linear(d_model, self.d_output)

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
        if self.loss=='NLLGaussian':
            mid_idx = int(self.d_output/2)
            x_uncertainties = F.softplus(x[..., mid_idx:])
            x = torch.cat([x[..., :mid_idx], x_uncertainties], dim=-1)
        return x
    
# Definition of quantile regression (arXiv:2505.18311)
class QRModel(nn.Module):
    """
    Neural Network for Quantile Regression as described in arXiv:2505.18311
    """
    def __init__(self, input_dim=8, quantiles=(0.1, 0.5, 0.9), target='chirp_mass'):
        """
        Args:
            input_dim (int): Number of input features (8 in the paper)
            quantiles (tuple): Quantiles to estimate
            target (str): One of 'chirp_mass', 'mass_ratio', 'total_mass'
        """
        super(QRModel, self).__init__()
        self.quantiles = quantiles
        self.target = target
        self.num_quantiles = len(quantiles)

        # Preprocessing
        self.norm = nn.LayerNorm(input_dim)

        # Hidden layers
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 12)
        self.dropout = nn.Dropout(p=0.20)
        self.leaky_relu = nn.LeakyReLU()

        # Output: raw quantile estimates
        self.output = nn.Linear(12, self.num_quantiles)

    def forward(self, x, recovered_param=None):
        """
        Args:
            x: Input tensor (batch_size, input_dim)
            recovered_param: Recovered parameter from pipeline (batch_size,)
                             Needed for chirp_mass and total_mass scaling
        Returns:
            Quantile predictions (batch_size, num_quantiles)
        """
        # Normalize
        x = self.norm(x)

        # Hidden layers
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))

        # Raw quantiles
        q_raw = self.output(x)

        # Sort outputs in ascending order (SoftSort can be approximated as sort)
        q_sorted, _ = torch.sort(q_raw, dim=1)

        if self.target == 'mass_ratio':
            # Constrain to [0,1]
            q_final = torch.sigmoid(q_sorted)
        elif self.target in ['chirp_mass', 'total_mass']:
            # Apply exp and scale
            if recovered_param is None:
                raise ValueError("recovered_param must be provided for chirp_mass or total_mass")
            q_exp = torch.exp(q_sorted)
            q_final = recovered_param.unsqueeze(1) * q_exp
        else:
            raise ValueError("Invalid target specified")

        return q_final