from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt
import torch
from data import LitBNSDataModule

class BNSEval():
    def __init__(
            self,
            config_path: str,
            checkpoint_path: str,
            save_path: Path | str,
            save_suffix: str,
            compute_on_cpu: bool=False):
        self.load_config(config_path)
        self.checkpoint_path = checkpoint_path
        self.compute_on_cpu = compute_on_cpu
        self.save_path = Path(save_path)
        self.csv_path = self.save_path / 'bns_eval.csv'
        self.save_suffix = save_suffix
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _set_var_map(self):
        self._var_map = {
            'mass_1': 'm_1',
            'mass_2': 'm_2',
            's1z': 'S_{1,z}',
            's2z': 'S_{2,z}',
            'distance': '\mathrm{distance}',
            'phic': '\phi_c',
            'inclination': '\mathrm{inclincation}',
            'chirp_mass': '\mathcal{M}',
            'total_mass': 'M',
            'mass_ratio': 'q'
        }

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

    def dinit(self, prefix: list, variables: list) -> tuple[dict] | dict:
        out = []
        for p in prefix:
            d = {f'{p}_{var}': [] for var in variables}
            out.append(d)
        return out[0] if len(out) == 1 else tuple(out)
        
    def dump_to_csv(self, outputs: dict, csv_path: str) -> pd.DataFrame:
        df = pd.DataFrame(outputs)
        df.to_csv(csv_path, index=False)
        print(f'Dumped results to {csv_path}')
        return df

    def compute_vals(self):
        if Path(self.csv_path).exists():
            return
        self.load_model()
        bns_data_module = LitBNSDataModule(**self.data_init_args)
        bns_data_module.setup('test')
        test_data_loader = bns_data_module.test_dataloader()

        # Placeholders for values
        pred_dict, truth_dict = self.dinit(['pred', 'truth'], self.variables)
        if not self.loss=='MSELoss':
            pred_sigma_dict = self.dinit(['sigma'], self.variables)
        return_dict = dict()

        it = 0
        for batch in test_data_loader:
            if it > 0: break # TODO: temporary
            h1, l1, params, idx = batch
            print('Batch iteration:', it*len(idx))
            device = h1.device
            inputs = torch.stack([h1.to(device), l1.to(device)], dim=2)
            truths = torch.stack([params[v] for v in self.variables], dim=1)

            with torch.no_grad():
                preds = self.model(inputs)

            # Move samples to CPU if required
            if self.compute_on_cpu:
                preds = preds.cpu()
                truths = truths.cpu()
            
            for i in range(len(self.variables)):
                p = self.variables[i]
                pred_dict[f'pred_{p}'].extend(preds[:, i].tolist())
                truth_dict[f'truth_{p}'].extend(truths[:, i].tolist())
                if not self.loss=='MSELoss':
                    pred_sigma_dict[f'sigma_{p}'].extend(preds[:, i+len(self.variables)].tolist())
            it += 1

        return_dict.update(self.maketensor(pred_dict))
        return_dict.update(self.maketensor(truth_dict))
        # TODO: implement for other losses
        if not self.loss=='MSELoss':
            return_dict.update(self.makesqrttensor(pred_sigma_dict))
        self.dump_to_csv(return_dict, csv_path=self.csv_path)
        return

    def plot(self):
        if not Path(self.csv_path).exists:
            raise RuntimeError('Please run compute_vals first.')
        df = pd.read_csv(self.csv_path)
        
        self._set_var_map()

        # Get variable names (everything after the prefix)
        pred_cols = [col for col in df.columns if col.startswith('pred_')]
        truth_cols = [col for col in df.columns if col.startswith('truth_')]
        assert len(pred_cols)==len(truth_cols)
        variables = [col.removeprefix('pred_') for col in pred_cols]

        # Create labels
        tex_variables = [self._var_map[var] for var in variables]
        label_preds = [fr'$\hat{{{tvar}}}$' for tvar in tex_variables]
        label_truths = [fr'${tvar}$' for tvar in tex_variables]
        label_residuals = [fr'$\hat{{{tvar}}} - {tvar}$' for tvar in tex_variables]
        label_z_scores  = [fr'$(\hat{{{tvar}}} - {tvar})/\hat\sigma_{{{tvar}}}$' for tvar in tex_variables]
        label_sigmas = [fr'$\hat\sigma_{tvar}' for tvar in tex_variables]

        # Get predictions and truths
        preds = np.stack([df[p] for p in pred_cols], axis=1)
        truths = np.stack([df[t] for t in truth_cols], axis=1)

        # Calculate residuals
        residuals = preds - truths
        
        # Check if sigma columns exist and stack normalized residuals
        sigma_cols = [f'sigma_{var}' for var in variables]
        sigma_exists = all(col in df.columns for col in sigma_cols)
        if sigma_exists:
            sigmas = np.stack([df[s] for s in sigma_cols], axis=1)
            z_scores = np.divide(residuals, sigmas,
                                 out=np.full_like(residuals, np.nan),
                                 where=(sigmas != 0))
        
        self._plot_corner(
            title=f'Residuals',
            fig_name=f'residuals_{self.save_suffix}',
            data=residuals,
            labels=label_residuals,
            color='purple')

        self._plot_hists(truths=truths, preds=preds,
                         label_truths=label_truths, label_preds=label_preds,
                         variables=variables)

        if sigma_exists:
            self._plot_corner(suptitle=f'Z Scores',
            fig_name=f'z_scores_{self.save_suffix}',
            data=z_scores,
            labels=label_z_scores,
            color='red')

            self._plot_corner(suptitle=f'Uncertainties',
            fig_name=f'sigmas_{self.save_suffix}',
            data=sigmas,
            labels=label_sigmas,
            color='blue')
        
        return

    def _plot_corner(self, title: str, fig_name: str | Path, data: np.array, labels: str, color: str, **kwargs) -> None:
        fig = corner.corner(
            data=data,
            labels=labels,
            color=color,
            **kwargs
        )
        fig.suptitle(title, fontsize=12)
        fig.subplots_adjust(top=.87)
        fig_name = Path(fig_name).with_suffix('.png')
        save_to = Path(self.save_path) / fig_name
        fig.savefig(save_to, bbox_inches='tight')
        print(f'Saved {save_to}')
        return
    
    def _plot_hists(self, truths: np.array, preds: np.array,
                    label_truths: list[str], label_preds: list[str], variables: list[str],
                    **kwargs) -> None:
        for i, var in enumerate(variables):
            plt.figure(figsize=(6, 4))
            plt.hist(truths[:, i], bins=30, alpha=0.5, color='gray', 
                    label=label_truths[i], density=False)
            plt.hist(preds[:, i], bins=30, alpha=1.0, color='black', 
                    label=label_preds[i], density=False)
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.legend()
            plt.title(f'{var} Distribution')
            plt.tight_layout()
            save_to = Path(self.save_path) / f'hist_{var}.png'
            plt.savefig(save_to)
            print(f'Saved to {save_to}')
            plt.close()

    def trainer_callback(self):
        self.compute_vals()
        self.plot()

def main():
    checkpoint_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/bns_parameter_estimation/bns_config_2vars_20250919_181359_36335176/checkpoints/bns-ckpt-epoch=46.ckpt'
    config_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/bns_parameter_estimation/bns_config_2vars_20250919_181359_36335176/config/config_20250919181438.yaml'
    save_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/bns_parameter_estimation/bns_config_2vars_20250919_181359_36335176/output'
    bns_eval = BNSEval(config_path=config_path, checkpoint_path=checkpoint_path,
                       save_path=save_path, save_suffix='test', compute_on_cpu=False)
    bns_eval.compute_vals()
    bns_eval.plot()
    return

if __name__ == '__main__':
    main()