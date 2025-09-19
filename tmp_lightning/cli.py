from pathlib import Path
from datetime import datetime
import torch
from lightning.pytorch.cli import SaveConfigCallback, LightningCLI
from lightning.fabric.utilities.cloud_io import get_filesystem

class CustomCallback(SaveConfigCallback):
    def __init__(self, parser, config, **kwargs):
        kwargs['save_to_log_dir'] = False
        logger_init_args = config.trainer.logger.init_args
        self.config_dir_overwrite = Path.cwd() / logger_init_args.project / logger_init_args.id / 'config'
        super().__init__(parser, config, **kwargs)
    
    def save_config(self, trainer, pl_module, stage):
        self.config_dir_overwrite.mkdir(parents=True, exist_ok=True)
        print('Config files will be copied to: ', self.config_dir_overwrite)
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        config_filename = f'{self.config_filename.removesuffix(".yaml")}_{now}.yaml'
        config_path = self.config_dir_overwrite / config_filename
        self.parser.save(
            self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
        )

def train():
    try:
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('medium')
        torch.cuda.memory_summary(device=None, abbreviated=False)
    except:
        print('*** CUDA not available! ***')
        pass
    cli = LightningCLI(save_config_callback=CustomCallback)

if __name__=='__main__':
    train()
    # Print CUDA memory summary for debugging purposes
    # current_directory = Path.cwd()
    # outdir = Path(current_directory) / LightningCLI.trainer_class.logger.experiment.project / LightningCLI.trainer_class.logger.experiment.id
    # outdir.mkdir(parents=True, exist_ok=True)
    # config_path = Path(outdir) / 'config.yaml'
    # print('Output directory:', outdir)

    # ckpt_dir = outdir / 'checkpoints'
    # # Select the latest checkpoint files
    # if ckpt_dir.exists():
    #     ckpt_files = sorted(ckpt_dir.glob('*.ckpt'), key=lambda f: f.stat().st_mtime, reverse=True)
    #     last_ckpt = ckpt_files[0] if ckpt_files else None
    # else:
    #     last_ckpt = None
    
    # from eval import BNSEval
    # print(config_path)
    # bns_eval = BNSEval(
    #     config_path=config_path,
    #     checkpoint_path=last_ckpt,
    #     csv_path=Path(outdir)/'outputs.csv'
    # )
    # bns_eval.trainer_callback()