import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import corner
from matplotlib.patches import Patch

import sys
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent, 'modules'))

print(sys.executable)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

class Plotter:
    def __init__(
        self,
        save_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/plots',
        datatype='BNS',
        hdf5_path='', # path to the HDF5 dataset
        split_indices_file='',
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device={self.device}')

        # Load datasets
        global get_dataloaders
        if datatype == 'BNS':
            from data_bns import get_dataloaders
        else:
            raise ValueError(f'Unknown datatype: {datatype}')
        self.datatype = datatype
        self.datadir = f'/ceph/submit/data/user/k/kyoon/KYoonStudy/models/{datatype}'
        self.hdf5_path = hdf5_path
        _, _, self.test_data_loader = get_dataloaders(
            hdf5_path=hdf5_path,
            test_batch_size=1,
            train_split=0.8,
            test_split=0.1,
            split_indices_file='',  # No precomputed indices file
            random_seed=42
        )

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            print(f'Created save path: {self.save_path}')

        self.param_list = ['mass_1', 'mass_2', 'chirp_mass', 'mass_ratio',
            'total_mass', 'right_ascension', 'declination', 'redshift', 'theta_jn']
        self.param_latex = [r'$m_1$', r'$m_2$', r'$\mathcal{M}$', r'$q$',
            r'$M$', r'$\alpha$', r'$\delta$', r'$z$', r'$\theta_{\mathrm{JN}}$']

        # Placeholders
        self.ssm_model = nn.Module()

    def extract_timestamp(self, filepath, sep='_'):
        """ Extracts the timestamp from the filename in the format '<directory>/YYYYMMDDHHMMSS.path'.
        Args:
            filepath (str): Path to the file from which to extract the timestamp.
            sep (str): Separator used in the filename. Default is '_'.
        Returns:
            str: The extracted timestamp in the format sep + 'YYYYMMDDHHMMSS', or '' if not found.
        Raises:
            ValueError: If the file path does not exist, is not a string, or does not end with '.path'.
            ValueError: If the filename does not match the expected format.
            ValueError: If the timestamp is not found in the filename.
        """
        import re
        if not os.path.exists(filepath):
            raise ValueError(f"File path '{filepath}' does not exist.")
        if not isinstance(filepath, str):
            raise ValueError("File path must be a string.")
        if not filepath.endswith('.path'):
            raise ValueError("File path must end with '.path'.")
        if not isinstance(sep, str):
            raise ValueError("Separator must be a string.")
        filename = os.path.basename(filepath)
        match = re.search(r'(\d{12})\.path$', filename)
        if match:
            return sep + match.group(1)
        else:
            return ''

    def compute_z_scores(self, pred_means, pred_stds, truth_means):
        diffs = pred_means - truth_means
        z_scores = (pred_means - truth_means) / pred_stds
        return diffs, z_scores

    def dump_to_csv(self, outputs, filename='ssm_outputs.csv'):
        """
        Dumps the outputs to a CSV file.
        Args:
            outputs (dict): Dictionary containing the outputs to be saved.
            filename (str): Name of the output CSV file.
        """
        import pandas as pd

        df = pd.DataFrame(outputs)
        df.to_csv(os.path.join(self.save_path, filename), index=False)
        print(f'Dumped results to {filename}')
        return df

    # Define loss function
    def compute_loss(self, preds, targets, loss, reduction='none'):
        """
        Define the loss function based on the specified loss type.
        Supported losses are 'NLLGaussian' and 'Quantile'.
        """
        if loss == 'NLLGaussian':
            criterion = nn.GaussianNLLLoss(reduction=reduction, full=True, eps=1e-7)
            outputs = {
                'mean': preds[:,:9],
                'sigma': preds[:,9:18],
            }
            loss_fn = criterion(outputs['mean'], targets, outputs['sigma'])
        elif loss == 'Quantile':
            from losses import QuantileLoss
            mean_loss = nn.MSELoss(reduction=reduction)
            q25_loss = QuantileLoss(quantile=0.25, reduction=reduction)
            q75_loss = QuantileLoss(quantile=0.75, reduction=reduction)
            outputs = {
                'mean': outputs[:,:9],
                'q25':  outputs[:,9:18],
                'q75':  outputs[:,18:27],
            }
            loss_fn = (
                mean_loss(outputs['mean'], targets) +
                q25_loss(outputs['q25'], targets) +
                q75_loss(outputs['q75'], targets)
            )
        else: raise ValueError(f"Unknown loss type: {loss}. Supported losses are 'NLLGaussian' and 'Quantile'.")
        return loss_fn

    def stack_inputs_truths(self, vals):
        h1, l1, params, idx = vals
        mass_1 = params['mass_1'].to(self.device)
        mass_2 = params['mass_2'].to(self.device)
        chirp_mass = (mass_1 * mass_2)**(3/5) / (mass_1 + mass_2)**(1/5)
        mass_ratio = mass_2 / mass_1
        total_mass = mass_1 + mass_2
        right_ascension = params['ra'].to(self.device)
        declination = params['dec'].to(self.device)
        redshift = params['redshift'].to(self.device)
        theta_jn = params['theta_jn'].to(self.device)

        inputs = torch.stack([h1.to(self.device), l1.to(self.device)], dim=2)
        truths = torch.stack([mass_1, mass_2,
                               chirp_mass, mass_ratio, total_mass,
                               right_ascension, declination, redshift, theta_jn], dim=1)
        return inputs, truths

    def ssm_compute_vals(self, ssm_model, loss='NLLGaussian', batch_size=16, compute_on_cpu=False, csv_output=False, timestamp=''):
        """
        Computes predictions and truths for the SSM model on the test data.

        Args:
            ssm_model (nn.Module): The SSM model to evaluate.
            loss (str): Loss function used in the model ('NLLGaussian' or 'Quantile').
            batch_size (int): Number of samples to process in each batch.
            compute_on_cpu (bool): If True, move computations to CPU.
            csv_output (bool): If True, save outputs to a CSV file.
            timestamp (str): Timestamp to append to the output filename.

        Returns:
            dict: Dictionary containing predictions and truths.
        """        
        def dinit(prefix):
            l =  self.param_list
            d = {f'{prefix}_{k}': [] for k in l}
            return d
        
        def maketensor(d):
            return {k: torch.tensor(v) for k, v in d.items()}
        
        pred_dict, truth_dict = dinit('pred'), dinit('truth')
        loss_per_sample = []
        if loss=='NLLGaussian':
            pred_sigma_dict = dinit('sigma')
        elif loss=='Quantile':
            pred_q25_dict, pred_q75_dict = dinit('q25'), dinit('q75')
        else:
            raise ValueError(f'Unknown loss function: {loss}')

        device = next(ssm_model.parameters()).device
        ssm_model.eval()

        # Redefine the test data loader to iterate over batches
        _, _, test_data_loader = get_dataloaders(
            hdf5_path=self.hdf5_path,
            test_batch_size=batch_size,
            train_split=0.8,
            test_split=0.1,
            split_indices_file='',  # No precomputed indices file
            random_seed=42
        )

        for _, vals in enumerate(test_data_loader):
            inputs, truths = self.stack_inputs_truths(vals)
            inputs = inputs.to(device) # (B, length of sequence, 2)
            truths = truths.to(device) # (B, 9)

            with torch.no_grad():
                preds = ssm_model(inputs)

            if preds.ndim == 3:
                preds = preds.squeeze(2)  # e.g., (B, N, 2) → (B, N)

            # Move samples to CPU if required
            if compute_on_cpu:
                preds = preds.cpu()
                truths = truths.cpu()
            
            for i in range(len(self.param_list)):
                p = self.param_list[i]
                pred_dict[f'pred_{p}'].extend(preds[:, i].tolist())
                truth_dict[f'truth_{p}'].extend(truths[:, i].tolist())
                if loss=='NLLGaussian':
                    pred_sigma_dict[f'sigma_{p}'].extend(preds[:, i+len(self.param_list)].tolist())
                elif loss=='Quantile':
                    pred_q25_dict[f'q25_{p}'].extend(preds[:, i+len(self.param_list)].tolist())
                    pred_q75_dict[f'q75_{p}'].extend(preds[:, i+2*len(self.param_list)].tolist())
            l = self.compute_loss(preds, truths, loss=loss, reduction='none')
            l = l.mean(dim=1, keepdim=False).tolist() # List of length B
            loss_per_sample.extend(l)

        return_dict = {'loss_per_sample': torch.tensor(loss_per_sample)}
        return_dict.update(maketensor(pred_dict))
        return_dict.update(maketensor(truth_dict))

        if loss=='NLLGaussian':
            return_dict.update(maketensor(pred_sigma_dict))
        elif loss=='Quantile':
            return_dict.update(maketensor(pred_q25_dict))
            return_dict.update(maketensor(pred_q75_dict))

        if csv_output:
            csv_name = f'ssm_{self.datatype}_{loss}{timestamp}_outputs.csv'
            self.dump_to_csv(return_dict, filename=csv_name)

        return return_dict

    def plot_ssm_predictions(self, d_input, d_model, n_layers, model_path='', batch_size=16,
                             save_prefix='ssm', loss='NLLGaussian', csv_output=False):
        timestamp = self.extract_timestamp(model_path, sep='_')
        from models import S4Model
        if loss=='NLLGaussian': d_output = 18
        elif loss=='Quantile': d_output = 27
        else: raise ValueError(f'Unknown loss function: {loss}')
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model path '{model_path}' does not exist or was not provided.")
        self.ssm_model = S4Model(d_input=d_input, d_output=d_output, d_model=d_model, loss=loss, n_layers=n_layers, dropout=0.0, prenorm=False)
        self.ssm_model = self.ssm_model.to(self.device)
        self.ssm_model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.ssm_model.eval()

        ssm_outputs = self.ssm_compute_vals(self.ssm_model, loss=loss, batch_size=batch_size,
                                            compute_on_cpu=False, csv_output=csv_output, timestamp=timestamp)

        # Get differences between predictions and truths
        ssm_diffs, ssm_z_scores, ssm_uncertainties = [], [], []
        labels_diffs, labels_z_scores, labels_uncertainties = [], [], []
        for i in range(len(self.param_list)):
            param, platex = self.param_list[i], self.param_latex[i]
            diff, z_score = self.compute_z_scores(ssm_outputs[f'pred_{param}'], torch.ones_like(ssm_outputs[f'pred_{param}']), ssm_outputs[f'truth_{param}'])
            ssm_diffs.append(diff.numpy())
            ssm_z_scores.append(z_score.numpy())
            if loss=='NLLGaussian':
                uncertainty = ssm_outputs[f'sigma_{param}']
            elif loss=='Quantile':
                uncertainty = ssm_outputs[f'q75_{param}'] - ssm_outputs[f'q25_{param}']
            else:
                raise ValueError(f'Invalid {loss=}.')
            ssm_uncertainties.append(uncertainty.numpy())
            labels_diffs.append(fr'$\hat{platex} - {platex}$')
            labels_z_scores.append(fr'$(\hat{platex} - {platex})/\sigma_{platex}$')
            labels_uncertainties.append(fr'$\sigma_{platex}')
        ssm_diffs_stacked, ssm_z_scores_stacked, ssm_uncertainties_stacked =\
            np.stack(ssm_diffs, axis=1), np.stack(ssm_z_scores, axis=1), np.stack(ssm_uncertainties, axis=1)
        
        # Get loss per sample
        loss_per_sample = ssm_outputs['loss_per_sample'].numpy()

        # Plot uncertainties
        figure_uncertainties = corner.corner(
            ssm_uncertainties_stacked,
            labels=labels_uncertainties,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.2f',
            color='C3'
        )
        figure_uncertainties.suptitle(f'Data: {self.datatype}, Loss: {loss}', fontsize=12)
        figure_uncertainties.subplots_adjust(top=0.87)

        # Plot diffs
        figure_diffs = corner.corner(
            ssm_diffs_stacked,
            quantiles=[0.16, 0.5, 0.84],
            labels=labels_diffs,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.2f',
            color='C5'
        )

        # Plot z_scores
        figure_z_scores = corner.corner(
            ssm_z_scores_stacked,
            quantiles=[0.16, 0.5, 0.84],
            labels=labels_z_scores,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.2f',
            color='C4'
        )
        figure_z_scores.suptitle(f'Data: {self.datatype}, Loss: {loss}', fontsize=12)
        figure_z_scores.subplots_adjust(top=0.87)

        # Plot loss per sample
        figure_loss = corner.corner(
            loss_per_sample,
            quantiles=[0.16, 0.5, 0.84],
            labels=[f'{loss} loss per sample'],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.2f',
            color='C7'
        )
        figure_loss.suptitle(f'Data: {self.datatype}, Loss: {loss}', fontsize=12)

        # Save the figures
        if self.save_path is not None:
            figure_diffs.savefig(os.path.join(self.save_path, f'{save_prefix}_{self.datatype}_{loss}{timestamp}_diffs.png'), bbox_inches='tight')
            figure_uncertainties.savefig(os.path.join(self.save_path, f'{save_prefix}_{self.datatype}_{loss}{timestamp}_uncertainties.png'), bbox_inches='tight')
            figure_z_scores.savefig(os.path.join(self.save_path, f'{save_prefix}_{self.datatype}_{loss}{timestamp}_z_scores.png'), bbox_inches='tight')
        else:
            figure_diffs.tight_layout()
            figure_diffs.show()
            figure_uncertainties.tight_layout()
            figure_uncertainties.show()
            figure_z_scores.tight_layout()
            figure_z_scores.show()

        print(f'Saved SSM predictions plots to {self.save_path}')
        print(f'SSM predictions computed with {loss=}')

        return

    def plot_bilby(self, bilby_dir='/ceph/submit/data/user/k/kyoon/KYoonStudy/fitresults'):
        """
        Placeholder for bilby plotting function.
        This function should be implemented to plot bilby results.
        """
        import pandas as pd
        import glob
        if self.datatype=='SHO':
            bilby_dir = os.path.join(bilby_dir, 'bilby_sho')
        elif self.datatype=='SineGaussian':
            bilby_dir = os.path.join(bilby_dir, 'bilby_sg')
        bilby_parquet = sorted(glob.glob(os.path.join(bilby_dir, f'{self.datatype}_bilby_id*.parquet')))
        if not bilby_parquet:
            raise FileNotFoundError(f'No bilby parquet files found in {bilby_dir} for datatype {self.datatype}')
        parquet_name = os.path.join(bilby_dir, f'{self.datatype}_bilby_combined.parquet')
        if not os.path.exists(parquet_name):
            dfs = []
            for f in bilby_parquet:
                _df = pd.read_parquet(f)
                print(f'Loaded {f} with shape {_df.shape}')
                _df = _df.reset_index()
                _df['event_id'] = _df['event_id'].ffill()
                _df = _df.set_index('event_id')
                dfs.append(_df)           
            combined_df = pd.concat(dfs, axis=0, ignore_index=False)
            print(f'Combined {len(bilby_parquet)} parquet files into a dataframe with shape {combined_df.shape}')
            combined_df.to_parquet(parquet_name)
            print(f'Saved combined dataframe to {parquet_name}')
        else:
            combined_df = pd.read_parquet(parquet_name)
            print(f'Loaded combined dataframe from {parquet_name} with shape {combined_df.shape}')

if __name__ == "__main__":
    model_path = '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/BNS/output/model.SSM.BNS.NLLGaussian.250707141027.path'
    plotter = Plotter(datatype='BNS',
                      hdf5_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/models/BNS/bns_waveforms.hdf5',
                      split_indices_file='/ceph/submit/data/user/k/kyoon/KYoonStudy/models/BNS/bns_data_indices.npz')
    plotter.plot_ssm_predictions(d_input=2, d_model=4, n_layers=4, batch_size=256,
                                 model_path=model_path, csv_output=True)