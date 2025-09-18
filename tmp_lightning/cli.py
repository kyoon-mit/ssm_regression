from pathlib import Path
import torch
from lightning.pytorch.cli import LightningCLI
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

if __name__=='__main__':
    torch.set_float32_matmul_precision('medium')
    # Print CUDA memory summary for debugging purposes
    torch.cuda.memory_summary(device=None, abbreviated=False)
    lr_monitor = LearningRateMonitor(logging_interval='step') # or 'epoch'
    wandb_logger = WandbLogger(project='bns_parameter_estimation')
    current_directory = Path.cwd()
    outdir = Path(current_directory) / wandb_logger.experiment.project / wandb_logger.experiment.id
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = Path(outdir) / 'config.yaml'
    print('Wandb run ID:', wandb_logger.experiment.id)
    trainer = Trainer(callbacks=[lr_monitor], logger=wandb_logger)
    cli = LightningCLI(save_config_kwargs={'config_filename': config_path})

    ckpt_dir = outdir / 'checkpoints'
    # Select the latest checkpoint file
    if ckpt_dir.exists():
        ckpt_files = sorted(ckpt_dir.glob('*.ckpt'), key=lambda f: f.stat().st_mtime, reverse=True)
        last_ckpt = ckpt_files[0] if ckpt_files else None
    else:
        last_ckpt = None
    
    from eval import trainer_callback
    print(config_path)
    trainer_callback(checkpoint_path=last_ckpt,
                     config_path=config_path,
                     csv_path=Path(outdir) / 'outputs.csv')
