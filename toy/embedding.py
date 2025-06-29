import logging
import os

import torch
from torch.utils.data import DataLoader
from torch import optim
import wandb
from datetime import datetime

logger = logging.getLogger('embedding')

class Embedding():
    def __init__(
        self,
        batch_size=1000, # batch size for training and validation
        num_hidden_layers_h=2, # number of hidden layers in the embedding model
        device=None,
        datatype='SHO', # options: 'SineGaussian', 'SHO', 'LIGO'
        datasfx='', # suffix for the dataset, e.g., '_sigma0.4_gaussian'
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
        logger.info(f"Using device={self.device}")
        
        self.datatype = datatype

        self.datadir = f'/ceph/submit/data/user/k/kyoon/KYoonStudy/models/{datatype}'
        self.modeldir = os.path.join(self.datadir, 'output')

        self.train_dict = torch.load(os.path.join(self.datadir, f'train{datasfx}.pt'), map_location=device, weights_only=True)
        self.val_dict = torch.load(os.path.join(self.datadir, f'val{datasfx}.pt'), map_location=device, weights_only=True)

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

        self.model = None
        self.vicreg_loss = None
        self.num_hidden_layers_h = num_hidden_layers_h
        self.optimizer, self.scheduler = None, None

    def build_model(self):
        logger.info('==> Building embedding model...')
        # Load loss and model
        from losses import VICRegLoss
        from models import SimilarityEmbedding
        self.vicreg_loss = VICRegLoss()
        self.model = SimilarityEmbedding(num_hidden_layers_h=self.num_hidden_layers_h).to(self.device)
        # setup scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-3)
        scheduler_1 = optim.lr_scheduler.ConstantLR(self.optimizer, total_iters=20)
        scheduler_2 = optim.lr_scheduler.OneCycleLR(self.optimizer, total_steps=20, max_lr=5e-4)
        scheduler_3 = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.scheduler = optim.lr_scheduler.SequentialLR(
        self.optimizer, schedulers=[scheduler_1, scheduler_2, scheduler_3], milestones=[20, 40])
        logger.info('...done!')

        # print number of trainable parameters
        sum_param=0
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                sum_param+=param.numel()
        logger.info(f'trainable parameters: {sum_param}')

    # Definition of training cycle
    def train_one_epoch(self, epoch_index, **vicreg_kwargs):
        running_sim_loss = 0.
        last_sim_loss = 0.

        #for idx, val in enumerate(tqdm(train_data_loader, desc='train', leave=False), 1):
        for idx, val in enumerate(self.train_data_loader):
            _, _, unshifted_data, augmented_data, _, _, _, _, _, _ = val
            embedded_values_aug, _ = self.model(augmented_data)
            embedded_values_orig, _ = self.model(unshifted_data)
            sim_loss = 0

            similar_embedding_loss, _repr, _cov, _std = self.vicreg_loss(
                embedded_values_aug,
                embedded_values_orig,
                **vicreg_kwargs
            )
            sim_loss += similar_embedding_loss
            self.optimizer.zero_grad()
            sim_loss.backward()
            self.optimizer.step()
            # Gather data and report
            running_sim_loss += sim_loss.item()
            if idx % 10 == 0:
                last_sim_loss = running_sim_loss / 10
                tb_x = epoch_index * len(self.train_data_loader) + idx
                #tb_writer.add_scalar('SimLoss/train', last_sim_loss, tb_x)
                logger.info(f'SimLoss/train {last_sim_loss} {tb_x}')
                running_sim_loss = 0.
        return last_sim_loss

    def val_one_epoch(self, epoch_index, **vicreg_kwargs):
        running_sim_loss = 0.
        last_sim_loss = 0.

        #for idx, val in enumerate(tqdm(val_data_loader, desc='val', leave=False), 1):
        for idx, val in enumerate(self.val_data_loader):
            _, _, unshifted_data, augmented_data, _, _, _, _, _, _ = val
            embedded_values_aug, _ = self.model(augmented_data)
            embedded_values_orig, _ = self.model(unshifted_data)
            sim_loss = 0

            similar_embedding_loss, _repr, _cov, _std = self.vicreg_loss(
                embedded_values_aug,
                embedded_values_orig,
                **vicreg_kwargs
            )
            sim_loss += similar_embedding_loss

            running_sim_loss += sim_loss.item()
            if idx % 5 == 0:
                last_sim_loss = running_sim_loss / 5
                tb_x = epoch_index * len(self.val_data_loader) + idx + 1
                #tb_writer.add_scalar('SimLoss/val', last_sim_loss, tb_x)
                logger.info(f'SimLoss/val {last_sim_loss} {tb_x}')
                #tb_writer.flush()
                logger.info(f'Last {_repr.item():.2f}; {_cov.item():.2f}; {_std.item():.2f}')
                running_sim_loss = 0.
        #tb_writer.flush()
        return last_sim_loss

if __name__=='__main__':

    datatype = 'SHO'
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')

    os.environ['WANDB_MODE'] = 'offline'
    wandb.init(project=f'embedding_{datatype}', name=f'embedding_{datatype}_{timestamp}')

    task = Embedding(datatype=datatype, datasfx='_sigma0.4_gaussian')
    task.build_model()

    logger.info('Start training...')

    EPOCHS = 220

    for epoch_number in range(EPOCHS):
        logger.info('EPOCH {}:'.format(epoch_number + 1))
        wt_repr, wt_cov, wt_std = (1, 1, 1)

        if epoch_number < 20:
            wt_repr, wt_cov, wt_std = (5, 1, 5)
        elif epoch_number < 45:
            wt_repr, wt_cov, wt_std = (2, 2, 1)

        logger.info(f"VicReg wts: {wt_repr=} {wt_cov=} {wt_std=}")
        # Gradient tracking
        task.model.train(True)
        avg_train_loss = task.train_one_epoch(epoch_number, wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)

        # no gradient tracking, for validation
        task.model.train(False)
        avg_val_loss = task.val_one_epoch(epoch_number, wt_repr=wt_repr, wt_cov=wt_cov, wt_std=wt_std)

        logger.info(f"Train/Val Sim Loss after epoch: {avg_train_loss:.4f}/{avg_val_loss:.4f}")

        wandb.log({
            'epoch': epoch_number,
            'train_loss': avg_train_loss,
            'val_accuracy': avg_val_loss,
            'lr': task.optimizer.param_groups[0]['lr'],
            'wt_repr': wt_repr,
            'wt_cov': wt_cov,
            'wt_std': wt_std
        })

        epoch_number += 1
        try:
            task.scheduler.step(avg_val_loss)
        except TypeError:
            task.scheduler.step()

    wandb.finish()

    torch.save(task.model.state_dict(), os.path.join(task.modeldir, f'model.CNN.{datatype}.{timestamp}.path'))
