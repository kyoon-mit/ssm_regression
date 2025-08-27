import numpy as np
from bilby.core.prior import Uniform
import torch

def damped_sho(t, omega_0, beta, shift=0):
    # beta less than 1 for underdamped
    envel = beta * omega_0
    osc = torch.sqrt(1 - beta**2) * omega_0
    tau = t - shift
    data = torch.exp(-envel * tau) * torch.cos(osc * tau)
    data[tau < 0] = 0  # assume oscillator starts at tau = 0
    return data

def get_data(omega_0=None, beta=None, shift=None, num_points=1):
    """Sample omega, beta, shift and return a batch of data with noise"""
    omega_0 = priors['omega_0'].sample() if omega_0 is None else omega_0
    omega_0 = torch.tensor(omega_0).to(dtype=torch.float32)
    beta = priors['beta'].sample() if beta is None else beta
    beta = torch.tensor(beta).to(dtype=torch.float32)
    shift = priors['shift'].sample() if shift is None else shift
    shift = torch.tensor(shift).to(dtype=torch.float32)

    t_vals = torch.linspace(-1, 10, num_points).to(dtype=torch.float32) #

    y = damped_sho(t_vals, omega_0=omega_0, beta=beta, shift=shift)
    y += sigma * torch.randn(size=y.size()).to(dtype=torch.float32)

    return t_vals, y, omega_0, beta, shift
    


num_points = 200
t_vals = np.linspace(-1, 10, num_points)
sigma = 0.4
priors = dict()

priors['omega_0'] = Uniform(0.1, 2, name='omega_0', latex_label='$\omega_0$') #np array
priors['beta'] = Uniform(0, 0.5, name='beta', latex_label='$\\beta$') #np array
priors['shift'] = Uniform(-4, 4, name='shift', latex_label='$\Delta\;t$')

num_simulations = 100000
num_repeats = 10
theta_unshifted_vals = []
theta_shifted_vals = []
data_unshifted_vals = []
data_shifted_vals = []

for ii in range(num_simulations):
    # generated data with a fixed shift
    t_vals, y_unshifted, omega, beta, shift = get_data(num_points=num_points, shift=1)
    # create repeats
    theta_unshifted = torch.tensor([omega, beta, shift]).repeat(num_repeats, 1).to(device=device)
    theta_unshifted_vals.append(theta_unshifted)
    data_unshifted_vals.append(y_unshifted.repeat(num_repeats, 1).to(device=device))
    # generate shifted data
    theta_shifted = []
    data_shifted = []
    for _ in range(num_repeats):
        t_val, y_shifted, _omega, _beta, shift = get_data(
            omega_0=omega, beta=beta,  # omega and beta same as above
            shift=None,
            num_points=num_points
        )
        theta_shifted.append(torch.tensor([omega, beta, shift]))
        data_shifted.append(y_shifted)
    theta_shifted_vals.append(torch.stack(theta_shifted).to(device=device))
    data_shifted_vals.append(torch.stack(data_shifted).to(device=device))

class DataGenerator(Dataset):
    def __len__(self):
        return num_simulations

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # return shifted and unshifted theta and data
        return (
            theta_shifted_vals[idx].to(dtype=torch.float32),
            theta_unshifted_vals[idx].to(dtype=torch.float32),
            data_shifted_vals[idx].to(dtype=torch.float32),
            data_unshifted_vals[idx].to(dtype=torch.float32)
        )
    
dataset = DataGenerator()

train_set_size = int(0.8 * num_simulations)
val_set_size = int(0.1 * num_simulations)
test_set_size = int(0.1 * num_simulations)

train_data, val_data, test_data = torch.utils.data.random_split(
    dataset, [train_set_size, val_set_size, test_set_size])

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

test_data_loader = DataLoader(
    test_data, batch_size=1,
    shuffle=False
)

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
    
vicreg_loss = VICRegLoss()

# modified 
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
    
num_dim = 3

class SimilarityEmbedding(nn.Module):
    """Simple Dense embedding"""
    def __init__(self, num_hidden_layers_h=1, activation=torch.relu):
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
    
similarity_embedding = SimilarityEmbedding(num_hidden_layers_h=2).to(device)

optimizer = optim.Adam(similarity_embedding.parameters(), lr=5e-3)
scheduler_1 = optim.lr_scheduler.ConstantLR(optimizer, total_iters=2)
scheduler_2 = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=200, max_lr=5e-4)
scheduler_3 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[50, 150])

embed, rep = similarity_embedding(data_aug)
embed.shape, rep.shape

emb_aug, rep_aug = similarity_embedding(data_aug)
emb_orig, rep_orig = similarity_embedding(data_orig)
vicreg_loss(emb_aug, emb_orig)

# from tqdm.auto import tqdm

def train_one_epoch(epoch_index, tb_writer, **vicreg_kwargs):
    running_sim_loss = 0.
    last_sim_loss = 0.

    #for idx, val in enumerate(tqdm(train_data_loader, desc='train', leave=False), 1):
    for idx, val in enumerate(train_data_loader):
        augmented_theta, _, augmented_data, unshifted_data = val
        embedded_values_aug, _ = similarity_embedding(augmented_data)
        embedded_values_orig, _ = similarity_embedding(unshifted_data)
        sim_loss = 0

        similar_embedding_loss, _repr, _cov, _std = vicreg_loss(
            embedded_values_aug,
            embedded_values_orig,
            **vicreg_kwargs
        )
        sim_loss += similar_embedding_loss
        optimizer.zero_grad()
        sim_loss.backward()
        optimizer.step()
        # Gather data and report
        running_sim_loss += sim_loss.item()
        if idx % 10 == 0:
            last_sim_loss = running_sim_loss / 10
            tb_x = epoch_index * len(train_data_loader) + idx
            tb_writer.add_scalar('SimLoss/train', last_sim_loss, tb_x)
            running_sim_loss = 0.
    return last_sim_loss


def val_one_epoch(epoch_index, tb_writer, **vicreg_kwargs):
    running_sim_loss = 0.
    last_sim_loss = 0.

    #for idx, val in enumerate(tqdm(val_data_loader, desc='val', leave=False), 1):
    for idx, val in enumerate(val_data_loader):
        augmented_theta, _, augmented_data, unshifted_data = val
        embedded_values_aug, _ = similarity_embedding(augmented_data)
        embedded_values_orig, _ = similarity_embedding(unshifted_data)
        sim_loss = 0

        similar_embedding_loss, _repr, _cov, _std = vicreg_loss(
            embedded_values_aug,
            embedded_values_orig,
            **vicreg_kwargs
        )
        sim_loss += similar_embedding_loss

        running_sim_loss += sim_loss.item()
        if idx % 5 == 0:
            last_sim_loss = running_sim_loss / 5
            tb_x = epoch_index * len(val_data_loader) + idx + 1
            tb_writer.add_scalar('SimLoss/val', last_sim_loss, tb_x)
            tb_writer.flush()
            print(f'Last {_repr.item():.2f}; {_cov.item():.2f}; {_std.item():.2f}')
            running_sim_loss = 0.
    tb_writer.flush()
    return last_sim_loss

writer = SummaryWriter("damped-harmonic-oscillator-similarity-embedding-training", flush_secs=5)
epoch_number = 0

sum_param=0
for name, param in similarity_embedding.named_parameters():
    if param.requires_grad:
        sum_param+=param.numel()
print(sum_param)

# UNCOMMENT AND RUN TO TRAIN FROM SCRATCH
EPOCHS = 500

#for epoch in tqdm(range(1, EPOCHS + 1)):
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    wt_repr, wt_cov, wt_std = (1, 1, 1)

    if epoch < 20:
        wt_repr, wt_cov, wt_std = (5, 1, 5)
    elif epoch < 45:
        wt_repr, wt_cov, wt_std = (2, 2, 1)

    print(f"VicReg wts: {wt_repr=} {wt_cov=} {wt_std=}")
    # Gradient tracking
    similarity_embedding.train(True)
    avg_train_loss = train_one_epoch(epoch_number, writer,
                                     wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)
    
    # no gradient tracking, for validation
    similarity_embedding.train(False)
    avg_val_loss = val_one_epoch(epoch_number, writer,
                                 wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)
    
    print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

    epoch_number += 1
    try:
        scheduler.step(avg_val_loss)
    except TypeError:
        scheduler.step()

PATH = './damped-harmonic-oscillator-similarity-embedding-weights.neurips.pth'
torch.save(similarity_embedding.state_dict(), PATH)