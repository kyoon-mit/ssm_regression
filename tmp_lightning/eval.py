from pathlib import Path
import yaml
import pandas as pd
import torch
from data import LitBNSDataModule

class BNSEval():
    def __init__(self, config_path: str, checkpoint_path: str, csv_path: str, compute_on_cpu: bool=False):
        self.load_config(config_path)
        self.checkpoint_path = checkpoint_path
        self.csv_path = csv_path
        self.compute_on_cpu = compute_on_cpu

    def load_config(self, config_path: str) -> tuple[dict, dict, list]:
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.data_init_args = cfg.get('data', {}).get('init_args', {})
        self.model_init_args = cfg.get('model', {}).get('init_args', {})
        self.variables = self.data_init_args.get('variables')
        self.loss = self.model_init_args.get('loss')
        if not isinstance(self.variables, list):
            self.variables = list(self.variables)
            self.data_init_args['variables'] = self.variables
        print(f'Variables: {self.variables}.')

    def load_model(self):
        if self.loss=='MSELoss':
            from model_mean_only import LitS4ModelMeanOnly
            self.model = LitS4ModelMeanOnly(**self.model_init_args)
        else:
            from model import LitS4Model
            self.model = LitS4Model(**self.model_init_args)
        checkpoint = torch.load(self.checkpoint_path, weights_only=True,
                                map_location=torch.device('cpu' if self.compute_on_cpu else 'cuda'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
    def maketensor(self, d: dict) -> dict:
        return {k: torch.tensor(v) for k, v in d.items()}

    def makesqrttensor(self, d: dict) -> dict:
        return {k: torch.sqrt(torch.tensor(v)) for k, v in d.items()}

    def dinit(self, prefix: list, variables: list) -> tuple[dict]:
        out = []
        for p in prefix:
            d = {f'{p}_{var}': [] for var in variables}
            out.append(d)
        return tuple(d)
        
    def dump_to_csv(self, outputs: dict, csv_path: str) -> pd.DataFrame:
        df = pd.DataFrame(outputs)
        df.to_csv(csv_path, index=False)
        print(f'Dumped results to {csv_path}')
        return df

    def compute_vals(self):
        self.load_model()
        bns_data_module = LitBNSDataModule(**self.data_init_args)
        bns_data_module.setup('test')
        test_data_loader = bns_data_module.test_dataloader()

        # Placeholders for values
        pred_dict, truth_dict, pred_sigma_dict = self.dinit(['pred', 'truth', 'sigma'], self.variables)
        return_dict = dict()

        it = 0
        for batch in test_data_loader:
            if it > 0: break
            h1, l1, params, idx = batch
            print('Batch iteration:', it)
            device = h1.device
            inputs = torch.stack([h1.to(device), l1.to(device)], dim=2)
            truths = torch.stack([params[v] for v in self.variables], dim=1)

            with torch.no_grad():
                preds = self.maketensormodel(inputs)

            # Move samples to CPU if required
            if self.compute_on_cpu:
                preds = preds.cpu()
                truths = truths.cpu()
            
            for i in range(len(self.variables)):
                p = self.variables[i]
                pred_dict[f'pred_{p}'].extend(preds[:, i].tolist())
                truth_dict[f'truth_{p}'].extend(truths[:, i].tolist())
                # TODO: implement for other losses
                pred_sigma_dict[f'sigma_{p}'].extend(preds[:, i+len(self.variables)].tolist())
            it += 1

        return_dict.update(self.maketensor(pred_dict))
        return_dict.update(self.maketensor(truth_dict))
        # TODO: implement for other losses
        return_dict.update(self.makesqrttensor(pred_sigma_dict))
        self.dump_to_csv(return_dict, csv_path=self.csv_path)

    def plotter(self, variables):
        for var in variables:
            continue
        pass

    def trainer_callback(self):
        self.compute_vals()

def main():
    base_path = Path('/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning')
    checkpoint_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/lightning_logs/d16qhixu/checkpoints/ckpt_epoch=19.ckpt'
    config_path = base_path / 'config_2vars.yaml'
    csv_path = base_path / 'bns_o8_d64_n32_nllgaussian_2vars_outputs.csv'
    bns_eval = BNSEval(config_path=config_path, checkpoint_path=checkpoint_path,
                       csv_path=csv_path)
    bns_eval.compute_vals()
    return

if __name__ == '__main__':
    main()