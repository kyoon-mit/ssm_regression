from pathlib import Path
from datetime import datetime
import torch
from lightning.pytorch.cli import SaveConfigCallback, LightningCLI
from eval import BNSEval

class CustomCallback(SaveConfigCallback):
    def __init__(self, parser, config, **kwargs):
        kwargs['save_to_log_dir'] = False
        logger_init_args = config.trainer.logger.init_args
        log_dir = Path.cwd() / logger_init_args.project / logger_init_args.id
        self.config_dir_overwrite = log_dir / 'config'
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

def train() -> Path:
    try:
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('medium')
        torch.cuda.memory_summary(device=None, abbreviated=False)
    except:
        print('*** CUDA not available! ***')
        pass
    cli = LightningCLI(save_config_callback=CustomCallback)
    logger_init_args = cli.config.fit.trainer.logger.init_args
    log_dir = Path.cwd() / logger_init_args.project / logger_init_args.id
    return log_dir

def plot(log_dir: str | Path, save_suffix: str) -> None:
    checkpoint_path = get_latest(Path(log_dir)/'checkpoints', '*.ckpt')
    config_path = get_latest(Path(log_dir)/'config', '*.yaml')
    save_path = Path(log_dir) / 'output'
    bns_eval = BNSEval(config_path=config_path, checkpoint_path=checkpoint_path,
                       save_path=save_path, save_suffix=save_suffix, compute_on_cpu=False)
    bns_eval.trainer_callback()
    return

def get_latest(search_dir: str | Path, pattern: str='*.ckpt'):
    # Select the latest file with extension
    if Path(search_dir).exists():
        files = sorted(search_dir.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            latest = files[0]
        else:
            raise RuntimeError(f'File does not exist in {search_dir}')
    else:
        raise RuntimeError(f'{search_dir} does not exist.')
    return latest

def run():
    log_dir = train()
    plot(log_dir, save_suffix='plot')

if __name__=='__main__':
    run()