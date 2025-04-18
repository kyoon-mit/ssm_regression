import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device=}")

datadir = '/n/holystore01/LABS/iaifi_lab/Users/creissel/SHO/'
modeldir = '/n/holystore01/LABS/iaifi_lab/Users/creissel/SHO/models/'

# Load datasets
from data import DataGenerator

train_data = torch.load(datadir+'train.pt')
val_data = torch.load(datadir+'val.pt')

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

print('Finished loading data! Start loading model...')

# Load loss and model
from losses import VICRegLoss
from models import SimilarityEmbedding
vicreg_loss = VICRegLoss()
similarity_embedding = SimilarityEmbedding(num_hidden_layers_h=2).to(device)
# setup scheduler
optimizer = optim.Adam(similarity_embedding.parameters(), lr=5e-3)
scheduler_1 = optim.lr_scheduler.ConstantLR(optimizer, total_iters=2)
scheduler_2 = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=200, max_lr=5e-4)
scheduler_3 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[50, 150])

print('...done loading model!')

# print number of trainable parameters
sum_param=0
for name, param in similarity_embedding.named_parameters():
    if param.requires_grad:
        sum_param+=param.numel()
print('trainable parameters:',sum_param)

# Definition of training cicle
def train_one_epoch(epoch_index, **vicreg_kwargs):
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
            #tb_writer.add_scalar('SimLoss/train', last_sim_loss, tb_x)
            print('SimLoss/train', last_sim_loss, tb_x)
            running_sim_loss = 0.
    return last_sim_loss


def val_one_epoch(epoch_index, **vicreg_kwargs):
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
            #tb_writer.add_scalar('SimLoss/val', last_sim_loss, tb_x)
            print('SimLoss/val', last_sim_loss, tb_x)
            #tb_writer.flush()
            print(f'Last {_repr.item():.2f}; {_cov.item():.2f}; {_std.item():.2f}')
            running_sim_loss = 0.
    #tb_writer.flush()
    return last_sim_loss

print('Start training...')

EPOCHS = 10 

for epoch_number in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    wt_repr, wt_cov, wt_std = (1, 1, 1)

    if epoch_number < 20:
        wt_repr, wt_cov, wt_std = (5, 1, 5)
    elif epoch_number < 45:
        wt_repr, wt_cov, wt_std = (2, 2, 1)

    print(f"VicReg wts: {wt_repr=} {wt_cov=} {wt_std=}")
    # Gradient tracking
    similarity_embedding.train(True)
    avg_train_loss = train_one_epoch(epoch_number, wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)
    
    # no gradient tracking, for validation
    similarity_embedding.train(False)
    avg_val_loss = val_one_epoch(epoch_number, wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)
    
    print(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

    epoch_number += 1
    try:
        scheduler.step(avg_val_loss)
    except TypeError:
        scheduler.step()

import time
timestr = time.strftime("%Y%m%d-%H%M%S")
torch.save(similarity_embedding.state_dict(), modeldir+'model.CNN.'+timestr+'.path')
