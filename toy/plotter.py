import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import corner
from matplotlib.patches import Patch

import sys
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).resolve().parent.parent, 'modules'))
from utils import extract_timestamp

class Plotter:
    def __init__(
        self,
        save_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/plots',
        datatype='SineGaussian', # 'SineGaussian', 'SHO', or 'LIGO',
        datasfx='',
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device={self.device}')

        # Load datasets
        self.datatype = datatype
        self.datasfx = datasfx

        if datatype == 'SineGaussian':
            self.datadir = '/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/data/SG'
        elif datatype == 'SHO':
            self.datadir = '/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/data/DHO'
        else:
            raise ValueError(f'Unknown datatype: {datatype}')
        
        self.__loaddata__(os.path.join(self.datadir, f'test{self.datasfx}.pt'))

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            print(f'Created save path: {self.save_path}')
        
        # Placeholders
        self.embedding_model = nn.Module()  # Placeholder for the model, replace with actual model loading
        self.flow_model = nn.Module()  # Placeholder for the flow model, replace with actual model loading
        self.ssm_model = nn.Module()

    def __loaddata__(self, datapath):
        # Load datasets
        if self.datatype == 'SineGaussian':
            from data_sinegaussian import DataGenerator
        elif self.datatype == 'SHO':
            from data_sho import DataGenerator
        else:
            raise ValueError(f'Unknown datatype: {self.datatype}')

        self.test_dict  = torch.load(datapath, map_location=torch.device(self.device), weights_only=True)
        self.test_data  = DataGenerator(self.test_dict)

        self.test_data_loader = DataLoader(
            self.test_data, batch_size=1, num_workers=4,
            shuffle=False
        )

    def __savefig__(self, plot, save_name):
        save_name = os.path.join(self.save_path, save_name)
        plot.savefig(save_name, bbox_inches='tight')
        print(f'Saved {save_name}.')
        return

    def similarity_outputs(self, param_index, bounds, var_name, similarity_embedding):
        """
        GPT-4.1 generated.
        Collects similarity outputs for test samples, grouped by intervals defined in bounds.
        For each interval (bounds[i], bounds[i+1]), collects outputs where bounds[i] < val < bounds[i+1].

        Example usage:
            bounds = [0.2, 0.4, 0.6, 0.8]
            outputs = similarity_outputs(param_index=0, bounds=bounds, similarity_embedding=similarity_embedding)
            # outputs[0] contains outputs for (0.2, 0.4), outputs[1] for (0.4, 0.6), etc.

        Args:
            param_index (int): Index in theta_test to cut on (e.g., 0 for omega, 1 for beta).
            bounds (list): List of bounds (must be sorted).
            var_name (str): Name of the variable to be used in the plot legend.
            similarity_embedding (nn.Module): Model to compute similarity embedding.

        Returns:
            outputs_by_interval (list(list)): Each sublist contains similarity outputs for the corresponding interval.
            legends (list): List of legends for each interval.
        """
        outputs_by_interval = [[] for _ in range(len(bounds) - 1)]
        legends = [fr'{var_name} $\in$ ({bounds[i]}, {bounds[i + 1]})' for i in range(len(bounds) - 1)]
        for _, theta_test, data_test_orig, data_test, _ in self.test_data_loader:
            val = theta_test[0][0][param_index]
            for i in range(len(bounds) - 1):
                lower, upper = bounds[i], bounds[i + 1]
                if lower < val < upper:
                    with torch.no_grad():
                        _, similarity_output = similarity_embedding(data_test)
                    outputs_by_interval[i].append(similarity_output)
                    break  # Only add to one interval
        return outputs_by_interval, legends

    def similarity_outputs_2d(self, param_indices, bounds_list, var1_name, var2_name, similarity_embedding):
        """
        GPT-4.1 generated.
        Collects similarity outputs for test samples, grouped by 2D intervals defined in bounds_list.
        For each interval (x_bounds[i], x_bounds[i+1]) and (y_bounds[j], y_bounds[j+1]), collects outputs where
        x_bounds[i] < val_x < x_bounds[i+1] and y_bounds[j] < val_y < y_bounds[j+1].

        Example usage:
            x_bounds = [0.25, 0.5, 0.75]
            y_bounds = [0.2, 0.3, 0.4]
            outputs_2d = similarity_outputs_2d(
                param_indices=(0, 1), 
                bounds_list=(x_bounds, y_bounds), 
                var1_name='omega',
                var2_name='beta',
                similarity_embedding=similarity_embedding
            )
            # outputs_2d[0][0] contains outputs for (x_bounds[0], x_bounds[1]) and (y_bounds[0], y_bounds[1]), etc.

        Args:
            param_indices (tuple): Tuple of two ints, indices in theta_test to cut on (e.g., (0, 1)).
            bounds_list (tuple): Tuple of two lists, each list is the bounds for one parameter (must be sorted).
            var1_name (str): Name of the first variable to be used in the plot legend.
            var2_name (str): Name of the second variable to be used in the plot legend.
            similarity_embedding (nn.Module): Model to compute similarity embedding.

        Returns:
            outputs_by_bin (list(list(list))): 2D list where outputs_by_bin[i][j] contains similarity outputs for the corresponding bin.
            legends (list): List of legends for each bin in the 2D grid.
        """
        x_idx, y_idx = param_indices
        x_bounds, y_bounds = bounds_list
        outputs_by_bin = [[[] for _ in range(len(y_bounds)-1)] for _ in range(len(x_bounds)-1)]
        legends = [
            fr'{var1_name} $\in$ ({x_bounds[i]}, {x_bounds[i + 1]}) $\times$ {var2_name} $\in$ ({y_bounds[j]}, {y_bounds[j + 1]})'
            for i in range(len(x_bounds) - 1) for j in range(len(y_bounds) - 1)
        ]
        for _, theta_test, data_test_orig, data_test in self.test_data_loader:
            x_val = theta_test[0][0][x_idx]
            y_val = theta_test[0][0][y_idx]
            for i in range(len(x_bounds)-1):
                if not (x_bounds[i] < x_val < x_bounds[i+1]):
                    continue
                for j in range(len(y_bounds)-1):
                    if not (y_bounds[j] < y_val < y_bounds[j+1]):
                        continue
                    with torch.no_grad():
                        _, similarity_output = similarity_embedding(data_test)
                    outputs_by_bin[i][j].append(similarity_output)
                    break
                break
        return outputs_by_bin, legends

    def make_embedding_plots(self, outputs_by_interval, legends, save_name='embedding_plot.png', num_dim=3, title=None):
        """
        Create a corner plot from the similarity outputs.
        Args:
            outputs_by_interval (list): List of similarity outputs for each interval.
            legends (list): List of legends for each interval.
            save_name (str): Name of the file to save the plot.
            num_dim (int): Number of dimensions in the similarity outputs.
            title (str): Title for the plot. If not provided, will use a default title.

        Returns:
            figure (corner.corner): The created corner plot figure.
        """
        legend_labels = []
        outputs_by_interval[0], outputs_by_interval[1] =\
            outputs_by_interval[1], outputs_by_interval[0]
        for i, outputs in enumerate(outputs_by_interval):
            if len(outputs) == 0:
                continue
            outputs = torch.stack(outputs).cpu().numpy()
            print(outputs.shape)
            figure = corner.corner(
                outputs.reshape((outputs.shape[0]*outputs.shape[2], num_dim)),
                quantiles=[0.16, 0.5], color=f'C{i+1}'
            )
            legend_labels.append(Patch(color=f'C{i+1}', label=legends[i]))
        if title is None:
            title = f'Similarity Embeddings for {self.datatype} Data'
        figure.suptitle(title, fontsize=16)

        # Position legend outside top-left corner
        figure.legend(handles=legend_labels, loc='upper right', bbox_to_anchor=(1.25, 1.0), title='Range')

        # Add gridlines
        for ax in figure.get_axes():
            ax.grid(True, linestyle='--', alpha=0.3)

        # Show the plot
        if self.save_path is not None:
            figure.savefig(os.path.join(self.save_path, save_name), bbox_inches='tight')
        else:
            figure.tight_layout()
            figure.show()

        return figure

    def plot_embeddings(self, model_path='', num_hidden_layers_h=2):
        """
        Plot embeddings from the model.

        Args:
            model_path (str): Path to the trained model.
            num_hidden_layers_h (int): Number of hidden layers in the model.
                If not provided, will use the default number of hidden layers.

        Raises:
            ValueError: If the model path does not exist or is not provided.
            ValueError: If the number of hidden layers is not greater than 0.

        Returns:
            None
        """
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model path '{model_path}' does not exist or was not provided.")
        self.embedding_model = torch.load(model_path, map_location=self.device, weights_only=True)

        from models import SimilarityEmbedding
        # Initialize the model with the specified number of hidden layers
        if num_hidden_layers_h <= 0:
            raise ValueError("Number of hidden layers must be greater than 0.")
        print(f'Embeddings: Loaded model from {model_path}')
        similarity_embedding = SimilarityEmbedding(num_hidden_layers_h=2).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        similarity_embedding.load_state_dict(state_dict)
        similarity_embedding.eval()

        if self.datatype == 'SineGaussian':
            bounds = [0.2, 0.4, 0.6, 0.8, 1.6]
            var_name = r'f_0'
            plot_title = 'Sine Gaussian: fixed f_0'
        elif self.datatype == 'SHO':
            bounds = [0.2, 0.4, 0.6, 0.8, 1.4, 1.6]
            var_name = r'$\omega$'
            plot_title = r'DHO: fixed $\omega$'
        else:
            raise ValueError(f'Unknown datatype: {self.datatype}')

        # Collect similarity outputs for the test data
        similarity_outputs, legends = self.similarity_outputs(
            param_index=0, bounds=bounds,
            var_name=var_name, similarity_embedding=similarity_embedding
        )

        # Plot the similarity outputs
        timestamp = extract_timestamp(model_path, sep='_')
        fig = self.make_embedding_plots(
            outputs_by_interval=similarity_outputs,
            legends=legends,
            save_name=f'embedddings_{self.datatype}{timestamp}_space.png',
            num_dim=3,
            title=plot_title
        )
        return
    
    def flow_compute_vals(
        self,
        embed_path,
        flow_path,
        embed_hidden_layers=2,
        embed_hidden_channels=20,
        embed_kernel_size=21,
        embed_d_output=6,
        flow_hidden_features=50,
        num_points=200,
        save_prefix='flow',
        batch_size=256,
        num_samples=1000,
        csv_output=False):
        compute_on_cpu = True if self.device==torch.device('cpu') else False
        timestamp = extract_timestamp(flow_path, sep='_')
        csv_fname = f'{save_prefix}_{self.datatype}{timestamp}_outputs.csv'
        csv_name = os.path.join(self.save_path, csv_fname)
        # Check if CSV file exists
        if os.path.exists(csv_name):
            print(f'{csv_name} exists.\nOpening CSV file instead of recomputing flow model.')
            import pandas as pd
            df = pd.read_csv(csv_name)
            flow_outputs = {key: torch.tensor(value) for key, value in df.items()}
            return flow_outputs
        
        if not embed_path or not os.path.exists(embed_path):
            raise ValueError(f"Embedding model path '{embed_path}' does not exist or was not provided.")
        if not flow_path or not os.path.exists(flow_path):
            raise ValueError(f"Flow model path '{flow_path}' does not exist or was not provided.")
        timestamp = extract_timestamp(embed_path, sep='_')
        from parameter_estimation import NormalizingFlow
        nf = NormalizingFlow(datatype=self.datatype,
                             embed_model=embed_path,
                             embed_hidden_layers=embed_hidden_layers,
                             hidden_features=flow_hidden_features,
                             context_features=3,
                             num_points=num_points,
                             num_repeats=10,
                             embed_hidden_channels=embed_hidden_channels,
                             embed_kernel_size=embed_kernel_size,
                             embed_d_output=embed_d_output,
                             device=self.device,
                             load_data=False)
        nf.build_flow()
        flow = nf.flow
        flow.load_state_dict(torch.load(flow_path, map_location=self.device, weights_only=True))
        flow.eval()

        pred_omega, truth_omega = [], []
        pred_beta, truth_beta = [], []
        pred_sigma_omega = []
        pred_sigma_beta = []

        batch_contexts, batch_truths = [], []
        # Redefine the test data loader to iterate over batches
        self.test_data_loader = DataLoader(
            self.test_data, batch_size=batch_size, num_workers=0,
            shuffle=False
        )
        for idx, (_, truths, _, data_test, _) in enumerate(self.test_data_loader):
            # (B, num_repeat, 3(p1,p2,shift)), (B, num_repeat, L(t_vals))
            truths = truths[:,0,0:2] # (B, 2(p1,p2))
            # data_test = data_test.reshape(-1, 1, data_test.shape[2]) # (B, 1, L(t_vals))
            data_test = data_test[:,0:1,:] # (B, 1, L(t_vals))

            with torch.no_grad():
                samples = flow.sample(num_samples, context=data_test) # (B, num_samples, 2)

            # Move samples to CPU if required
            if compute_on_cpu:
                samples = samples.cpu()
                truths = truths.cpu()
            
            preds  = samples.mean(dim=1) # (B, 2)
            sigmas = samples.std(dim=1) # (B, 2)
            
            pred_omega.extend(preds[:, 0].tolist())
            pred_beta.extend(preds[:, 1].tolist())
            truth_omega.extend(truths[:, 0].tolist())
            truth_beta.extend(truths[:, 1].tolist())
            pred_sigma_omega.extend(sigmas[:, 0].tolist())
            pred_sigma_beta.extend(sigmas[:, 1].tolist())

        pred_omega, truth_omega = torch.tensor(pred_omega), torch.tensor(truth_omega)
        pred_beta, truth_beta = torch.tensor(pred_beta), torch.tensor(truth_beta)
        pred_sigma_omega = torch.tensor(pred_sigma_omega)
        pred_sigma_beta = torch.tensor(pred_sigma_beta)

        flow_outputs = {
            'pred_omega': pred_omega,
            'truth_omega': truth_omega,
            'pred_beta': pred_beta,
            'truth_beta': truth_beta,
            'pred_sigma_omega': pred_sigma_omega,
            'pred_sigma_beta': pred_sigma_beta,
        }

        if csv_output:
            self.dump_to_csv(flow_outputs, filename=csv_fname)

        return flow_outputs

    def plot_flow(
        self,
        num_points=200,
        flow_path='',
        flow_hidden_features=30,
        embed_path='', 
        embed_hidden_layers=2,
        embed_hidden_channels=20,
        embed_kernel_size=21,
        embed_d_output=6,
        save_prefix='flow',
        csv_output=True):
        """
        GPT-4.1 generated.
        Plots the flow model outputs for the given parameters.

        Args:
            num_points (int): Number of points in the data.
            flow_path (str): Path to the flow model.
            flow_hidden_features (int): Number of hidden features in the flow.
            embed_path (str): Path to the embedding model.
            embed_hidden_layers (int): Number of hidden layers in the embedding model.
            embed_hidden_channels (int): Number of hidden channels in ConvResidualNet.
            embed_kernel_size (int): Kernel size in ConvResidualNet.
            embed_d_output (int): Embedding output dimensions.
            save_prefix (str): Prefix for saving the figures.
            csv_output (bool): If True, saves outputs to a CSV file.

        Raises:
            ValueError: If the flow path or embedding path is not provided or does not exist.
            FileNotFoundError: If the flow model or embedding model files are not found.

        Returns:
            None
        """
        flow_outputs = self.flow_compute_vals(
            num_points=num_points,
            flow_path=flow_path,
            flow_hidden_features=flow_hidden_features,
            embed_path=embed_path,
            embed_hidden_layers=embed_hidden_layers,
            embed_hidden_channels=embed_hidden_channels,
            embed_kernel_size=embed_kernel_size,
            embed_d_output=embed_d_output,
            save_prefix=save_prefix,
            csv_output=csv_output)
        timestamp = extract_timestamp(flow_path, sep='_')
        omega_diffs, omega_z_scores = self.compute_z_scores(flow_outputs['pred_omega'], flow_outputs['pred_sigma_omega'], flow_outputs['truth_omega'])
        beta_diffs, beta_z_scores = self.compute_z_scores(flow_outputs['pred_beta'], flow_outputs['pred_sigma_beta'], flow_outputs['truth_beta'])
        # Stack into (N, 2) array
        diffs_stacked = np.stack([omega_diffs.numpy(), beta_diffs.numpy()], axis=1)
        z_scores_stacked = np.stack([omega_z_scores.numpy(), beta_z_scores.numpy()], axis=1)
        uncertainties_stacked = np.stack([flow_outputs['pred_sigma_omega'].numpy(), flow_outputs['pred_sigma_beta'].numpy()], axis=1)
        labels_diffs = [r'$\hat{\omega}_0 - \omega_0$', r'$\hat{\beta} - \beta$']
        labels_z_scores = [r'$(\hat{\omega}_0 - \omega_0)$/$\hat{\sigma}_{\omega_0}$', r'$(\hat{\beta} - \beta)$/$\hat{\sigma}_\beta$']
        labels_uncertainties = [r'$\hat{\sigma}_{\omega_0}$', r'$\hat{\sigma}_\beta$']
        figure_diffs = corner.corner(
            diffs_stacked,
            quantiles=[0.16, 0.5, 0.84],
            labels=labels_diffs,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.4f',
            color='C5'
        )
        figure_z_scores = corner.corner(
            z_scores_stacked,
            quantiles=[0.16, 0.5, 0.84],
            labels=labels_z_scores,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.4f',
            color='C5'
        )
        figure_uncertainties = corner.corner(
            uncertainties_stacked,
            labels=labels_uncertainties,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.4f',
            color='C3'
        )
        figure_uncertainties.suptitle(f'Data: {self.datatype} (NF)', fontsize=12)
        figure_uncertainties.subplots_adjust(top=0.87)
        figure_diffs.suptitle(f'Data: {self.datatype} (NF)', fontsize=12)
        figure_diffs.subplots_adjust(top=0.87)
        figure_z_scores.suptitle(f'Data: {self.datatype} (NF)', fontsize=12)
        figure_z_scores.subplots_adjust(top=0.87)

        # Save the figures
        if self.save_path is not None:
            self.__savefig__(figure_diffs, f'{save_prefix}_{self.datatype}{timestamp}_diffs.png')
            self.__savefig__(figure_z_scores, f'{save_prefix}_{self.datatype}{timestamp}_z_scores.png')
            self.__savefig__(figure_uncertainties, f'{save_prefix}_{self.datatype}{timestamp}_uncertainties.png')
        else:
            figure_diffs.tight_layout()
            figure_diffs.show()
            figure_z_scores.tight_layout()
            figure_z_scores.show()
            figure_uncertainties.tight_layout()
            figure_uncertainties.show()
        plt.close()

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
        csv_name = os.path.join(self.save_path, filename)
        df.to_csv(csv_name, index=False)
        print(f'Saved outputs to {csv_name}.')
        return df

    def ssm_reshaping(self, batch, input_dim=1, output_dim=2):
        theta_u, theta_s, data_u, data_s, _ = batch

        # remove repeat (take only first repeat for unshifted data)
        if input_dim==1:
            inputs = data_u[:, 0, :].unsqueeze(-1) if data_u.ndim == 3 else data_u.unsqueeze(-1)  # [B, 200, input_dim]
        else:
            inputs = data_u[:, 0, :input_dim] if data_u.ndim == 3 else data_u[:, :input_dim]
        targets = theta_u[:, 0, :output_dim] if theta_u.ndim == 3 else theta_u[:, :output_dim]  # [B, output_dim]

        return inputs, targets

    # Define loss function
    def compute_loss(self, loss, outputs, targets, reduction='none'):
        """
        Define the loss function based on the specified loss type.
        Supported losses are 'NLLGaussian' and 'Quantile'.
        """
        if loss == 'NLLGaussian':
            criterion = nn.GaussianNLLLoss(reduction=reduction, full=False, eps=1e-7)
            loss_fn = criterion(outputs[:,0:2], targets, outputs[:,2:4])
        elif loss == 'Quantile':
            from losses import QuantileLoss
            mean_loss = nn.MSELoss(reduction=reduction)
            q25_loss = QuantileLoss(quantile=0.1587, reduction=reduction)
            q75_loss = QuantileLoss(quantile=0.8413, reduction=reduction)
            loss_fn = (
                mean_loss(outputs[:,0:2], targets) +
                q25_loss(outputs[:,2:4], targets) +
                q75_loss(outputs[:,4:6], targets)
            )
        else: raise ValueError(f"Unknown loss type: {self.loss}. Supported losses are 'NLLGaussian' and 'Quantile'.")
        return loss_fn

    def ssm_compute_vals(self, ssm_model, loss='NLLGaussian', batch_size=16, csv_fname='',
        compute_on_cpu=False, csv_output=False, save_prefix='ssm', timestamp=''):
        """
        Computes predictions and truths for the SSM model on the test data.

        Args:
            ssm_model (nn.Module): The SSM model to evaluate.
            loss (str): Loss function used in the model ('NLLGaussian' or 'Quantile').
            batch_size (int): Number of samples to process in each batch.
            compute_on_cpu (bool): If True, move computations to CPU.
            csv_output (bool): If True, save outputs to a CSV file.
            save_prefix (str): Save prefix for the csv file.
            timestamp (str): Timestamp to append to the output filename.

        Returns:
            dict: Dictionary containing predictions and truths.
        """
        pred_param1, truth_param1 = [], []
        pred_param2, truth_param2 = [], []
        loss_per_sample = []
        if loss=='NLLGaussian':
            pred_sigma1, pred_sigma2 = [], [] # uncertainties
        elif loss=='Quantile':
            pred_q25_1, pred_q25_2 = [], []  # 25th quantile
            pred_q75_1, pred_q75_2 = [], []  # 75th quantile
        else:
            raise ValueError(f'Unknown loss function: {loss}')

        if not csv_fname:
            csv_fname = f'{save_prefix}_{self.datatype}_{loss}{timestamp}_outputs.csv'
        csv_name = os.path.join(self.save_path, csv_fname)
        # Check if CSV file exists
        if os.path.exists(csv_name):
            print(f'{csv_name} exists.\nOpening CSV file instead of recomputing ssm model.')
            import pandas as pd
            df = pd.read_csv(csv_name)
            # Make modifications here
            # df['pred_sigma1'] = np.sqrt(df['pred_sigma1'])
            # df['pred_sigma2'] = np.sqrt(df['pred_sigma2'])
            return_dict = {key: torch.tensor(value) for key, value in df.items()}
            return return_dict

        device = next(ssm_model.parameters()).device
        ssm_model.eval()

        # Redefine the test data loader to iterate over batches
        self.test_data_loader = DataLoader(
            self.test_data, batch_size=batch_size, num_workers=0,
            shuffle=False
        )

        for idx, vals in enumerate(self.test_data_loader):
            inputs, truths = self.ssm_reshaping(vals, output_dim=2) # (B, L(t_vals), 2), (B, 2)

            with torch.no_grad():
                preds = ssm_model(inputs) # (B, d_output)

            # Move samples to CPU if required
            if compute_on_cpu:
                preds = preds.cpu()
                truths = truths.cpu()
            
            pred_param1.extend(preds[:, 0].tolist())
            pred_param2.extend(preds[:, 1].tolist())
            truth_param1.extend(truths[:, 0].tolist())
            truth_param2.extend(truths[:, 1].tolist())
            l = self.compute_loss(loss, preds, truths, reduction='none')
            l = l.mean(dim=1, keepdim=False).tolist()
            loss_per_sample.extend(l)

            if loss == 'NLLGaussian':
                pred_sigma1.extend(preds[:, 2].tolist())
                pred_sigma2.extend(preds[:, 3].tolist())
            elif loss == 'Quantile':
                pred_q25_1.extend(preds[:, 2].tolist())
                pred_q25_2.extend(preds[:, 3].tolist())
                pred_q75_1.extend(preds[:, 4].tolist())
                pred_q75_2.extend(preds[:, 5].tolist())

        return_dict = {
            'pred_param1': torch.tensor(pred_param1),
            'truth_param1': torch.tensor(truth_param1),
            'pred_param2': torch.tensor(pred_param2),
            'truth_param2': torch.tensor(truth_param2),
            'loss_per_sample': torch.tensor(loss_per_sample),
        }
        if loss == 'NLLGaussian':
            return_dict['pred_sigma1'] = torch.sqrt(torch.tensor(pred_sigma1))
            return_dict['pred_sigma2'] = torch.sqrt(torch.tensor(pred_sigma2))
            # return_dict['pred_sigma1'] = torch.tensor(pred_sigma1)
            # return_dict['pred_sigma2'] = torch.tensor(pred_sigma2)
        elif loss == 'Quantile':
            return_dict['pred_q25_1'] = torch.tensor(pred_q25_1)
            return_dict['pred_q25_2'] = torch.tensor(pred_q25_2)
            return_dict['pred_q75_1'] = torch.tensor(pred_q75_1)
            return_dict['pred_q75_2'] = torch.tensor(pred_q75_2)

        if csv_output:
            self.dump_to_csv(return_dict, filename=csv_fname)

        return return_dict

    def plot_ssm_predictions(
            self,
            d_model,
            n_layers,
            model_path='',
            save_prefix='ssm',
            loss='NLLGaussian',
            csv_output=False,
            csv_fname='',
            # Flow parameters
            plot_flow=False,
            num_points=200,
            embed_path='',
            embed_hidden_layers=2,
            embed_hidden_channels=20,
            embed_kernel_size=21,
            embed_d_output=6,
            flow_path='',
            flow_hidden_features=50):
        timestamp = extract_timestamp(model_path, sep='_')
        from models import S4Model
        if loss=='NLLGaussian': d_output = 4
        elif loss=='Quantile': d_output = 6
        else: raise ValueError(f'Unknown loss function: {loss}')
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model path '{model_path}' does not exist or was not provided.")
        self.ssm_model = S4Model(d_input=1, d_output=d_output, d_model=d_model,
            loss=loss, n_layers=n_layers, dropout=0.0, prenorm=False)
        self.ssm_model = self.ssm_model.to(self.device)
        self.ssm_model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.ssm_model.eval()

        ssm_outputs = self.ssm_compute_vals(self.ssm_model, loss=loss, batch_size=1000, csv_fname=csv_fname,
            compute_on_cpu=False, csv_output=csv_output, save_prefix=save_prefix, timestamp=timestamp)
        
        if plot_flow:
            flow_outputs = self.flow_compute_vals(
                batch_size=1000,
                embed_path=embed_path,
                embed_hidden_layers=embed_hidden_layers,
                embed_hidden_channels=embed_hidden_channels,
                embed_kernel_size=embed_kernel_size,
                embed_d_output=embed_d_output,
                num_points=num_points,
                save_prefix=save_prefix,
                flow_path=flow_path,
                flow_hidden_features=flow_hidden_features,
                csv_output=csv_output)
            omega_diffs, omega_z_scores = self.compute_z_scores(flow_outputs['pred_omega'], flow_outputs['pred_sigma_omega'], flow_outputs['truth_omega'])
            beta_diffs, beta_z_scores   = self.compute_z_scores(flow_outputs['pred_beta'], flow_outputs['pred_sigma_beta'], flow_outputs['truth_beta'])
            flow_diffs_stacked = np.stack(
                [omega_diffs.numpy(), beta_diffs.numpy()],
                axis=1
            )
            flow_z_scores_stacked = np.stack(
                [omega_z_scores.numpy(), beta_z_scores.numpy()],
                axis=1
            )

        # bilby_outputs = self.get_bilby_results(bilby_dir=f'/ceph/submit/data/user/k/kyoon/KYoonStudy/fitresults/d{self.datasfx}/bilby_mcmc')
        bilby_outputs = self.get_bilby_results(bilby_dir=f'/ceph/submit/data/user/k/kyoon/KYoonStudy/fitresults/bilby_mcmc')
        bilby_omega_diffs, bilby_omega_z_scores = self.compute_z_scores(
            bilby_outputs['pred_omega'], bilby_outputs['pred_sigma_omega'], bilby_outputs['truth_omega']
        )
        bilby_beta_diffs, bilby_beta_z_scores = self.compute_z_scores(
            bilby_outputs['pred_beta'], bilby_outputs['pred_sigma_beta'], bilby_outputs['truth_beta']
        )
        bilby_z_scores_stacked = np.stack(
            [bilby_omega_z_scores, bilby_beta_z_scores],
            axis=1
        )
        bilby_diffs_stacked = np.stack(
            [bilby_omega_diffs, bilby_beta_diffs],
            axis=1
        )

        # if loss=='Quantile':
        #     ssm_outputs['pred_sigma1'] = (ssm_outputs['pred_q75_1'] - ssm_outputs['pred_q25_1']) / 2.
        #     ssm_outputs['pred_sigma2'] = (ssm_outputs['pred_q75_2'] - ssm_outputs['pred_q25_2']) / 2.

        # Get differences between predictions and truths
        param1_diff, param1_z_score = self.compute_z_scores(ssm_outputs['pred_param1'], ssm_outputs['pred_sigma1'], ssm_outputs['truth_param1'])
        param2_diff, param2_z_score = self.compute_z_scores(ssm_outputs['pred_param2'], ssm_outputs['pred_sigma2'], ssm_outputs['truth_param2'])

        # Stack into (N, 2) array
        ssm_diffs_stacked, ssm_z_scores_stacked, uncertainties_stacked =\
            np.stack([param1_diff.numpy(), param2_diff.numpy()], axis=1),\
            np.stack([param1_z_score.numpy(), param2_z_score.numpy()],axis=1),\
            np.stack([ssm_outputs['pred_sigma1'].numpy(), ssm_outputs['pred_sigma2'].numpy()],axis=1)

        # Get loss per sample
        loss_per_sample = ssm_outputs['loss_per_sample'].numpy()

        # Prepare labels for the plots
        if self.datatype == 'SHO':
            labels_diffs = [r'$\hat{\omega}_0 - \omega_0$', r'$\hat{\beta} - \beta$']
            labels_uncertainties = [r'$\hat{\sigma}_{\omega_0}$', r'$\hat{\sigma}_{\beta}$']
        elif self.datatype == 'SineGaussian':
            labels_diffs = [r'$\hat{f}_0 - f_0$', r'$\hat{\tau} - \tau$']
            labels_uncertainties = [r'$\hat{\sigma}_{f_0}$', r'$\hat{\sigma}_{\tau}$']
        labels_z_scores = [f'({labels_diffs[i]})/{labels_uncertainties[i]}' for i in range(len(labels_diffs))]

        # Plot uncertainties
        # figure_uncertainties = corner.corner(
        #     uncertainties_stacked,
        #     labels=labels_uncertainties,
        #     quantiles=[0.16, 0.5, 0.84],
        #     show_titles=True,
        #     title_kwargs={"fontsize": 12},
        #     label_kwargs={"fontsize": 12},
        #     title_fmt='.2f',
        #     color='C3'
        # )
        # figure_uncertainties.suptitle(f'Data: {self.datatype}, Loss: {loss}', fontsize=12)
        # figure_uncertainties.subplots_adjust(top=0.87)

        # Plot diffs
        figure_diffs = corner.corner(
            ssm_diffs_stacked,
            labels=labels_diffs,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.4f',
            color='C5',
            quantiles=[0.1587, 0.5, 0.8413],
            verbose=False
        )
        bilby_figure_diffs = corner.corner(
            bilby_diffs_stacked,
            labels=labels_diffs,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.4f',
            color='C5',
            quantiles=[0.1587, 0.5, 0.8413],
            verbose=False
        )

        corner_kwargs = dict(
            smooth=0.8,
            labels=labels_z_scores,
            label_kwargs=dict(fontsize=14),
            labelpad=-0.13,
            title_kwargs=dict(fontsize=14),
            tick_params=dict(labelsize=14),
            # quantiles=[0.1587, 0.5, 0.8413],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)), # 1, 2, 3 sigmas
            # levels=(0.1587, 0.5, 0.8413),
            plot_density=True,
            plot_datapoints=False,
            fill_contours=False,
            show_titles=False,
            title_fmt='.2f',
            max_n_ticks=3,
            range=[[-5, 5], [-5, 5]],
            verbose=False
        )
        colors, labels = ['gray', 'green', 'red'], ['Bilby', 'LFI', f'S4D\n({loss})']
        counts = [
            np.sum((-5 < bilby_outputs['pred_omega']) & (bilby_outputs['pred_omega'] < 5)),
            torch.sum((-5 < flow_outputs['pred_omega']) & (flow_outputs['pred_omega'] < 5)).item() if plot_flow else 0,
            torch.sum((-5 < ssm_outputs['pred_param1']) & (ssm_outputs['pred_param1'] < 5)).item()
        ]
        outliers = [
            np.sum((bilby_outputs['pred_omega'] < -5) | (bilby_outputs['pred_omega'] > 5)),
            torch.sum((flow_outputs['pred_omega'] < -5) | (flow_outputs['pred_omega'] > 5)).item() if plot_flow else 0,
            torch.sum((ssm_outputs['pred_param1'] < -5) | (ssm_outputs['pred_param1'] > 5)).item()
        ]
        print(f'Counts: {counts}, Outliers: {outliers}')
        weights = [
            np.ones(counts[0]) / counts[0],
            np.ones(counts[1]) / counts[1] if plot_flow else None,
            np.ones(counts[2]) / counts[2]
        ]
        # Plot z_scores
        if plot_flow:
            figure_z_scores = corner.corner(
                flow_z_scores_stacked,
                weights=weights[1],
                color=colors[1],
                **corner_kwargs
            )
            print('flow z-score quantiles:', np.quantile(flow_z_scores_stacked, [0.1587, 0.5, 0.8413], axis=0))
            print('flow diffs quantiles:', np.quantile(flow_diffs_stacked, [0.1587, 0.5, 0.8413], axis=0))
            corner.corner(
                bilby_z_scores_stacked,
                fig=figure_z_scores,
                weights=weights[0],
                color=colors[0],
                **corner_kwargs
            )
        else:
            figure_z_scores = corner.corner(
                bilby_z_scores_stacked,
                weights=weights[0],
                color=colors[0],
                **corner_kwargs
            )
        print('bilby z-score quantiles:', np.quantile(bilby_z_scores_stacked, [0.1587, 0.5, 0.8413], axis=0))
        print('bilby diffs quantiles:', np.quantile(bilby_diffs_stacked, [0.1587, 0.5, 0.8413], axis=0))
        print('ssm z-score quantiles:', np.quantile(ssm_z_scores_stacked, [0.1587, 0.5, 0.8413], axis=0))
        print('ssm diffs quantiles:', np.quantile(ssm_diffs_stacked, [0.1587, 0.5, 0.8413], axis=0))
        corner.corner(
            ssm_z_scores_stacked,
            fig=figure_z_scores,
            weights=weights[0],
            color=colors[2],
            **corner_kwargs
        )
        figure_z_scores.suptitle(f'{self.datatype}', fontsize=14)
        figure_z_scores.subplots_adjust(top=0.9)
        plt.legend(
            handles=[
                mlines.Line2D([], [], color=colors[i], label=labels[i])
                for i in range(len(colors))
            ],
            fontsize=14, frameon=False,
            bbox_to_anchor=(1.1, 2.2), loc="upper right"
        )
        for i, ax in enumerate(figure_z_scores.get_axes()):
            if plot_flow:
                if self.datatype=='SHO':
                    if i==0:
                        ax.set_ylim(0, 1.10*ax.get_ylim()[1])
                    elif i==3:
                        ax.set_ylim(0, 1.00*ax.get_ylim()[1])
                elif self.datatype=='SineGaussian':
                    if i==0:
                        ax.set_ylim(0, 1.30*ax.get_ylim()[1])
                    elif i==3:
                        ax.set_ylim(0, 1.10*ax.get_ylim()[1])
            ax.tick_params(labelsize=14)

        # Plot loss per sample as a histogram using matplotlib
        figure_loss, ax_loss = plt.subplots(figsize=(6, 4))
        _, _, _ = ax_loss.hist(
            loss_per_sample,
            alpha=0.5,
            label=f'{loss} loss per sample',
            bins=40,
            histtype='step',
            color='black',
        )
        figure_loss.suptitle(f'Data: {self.datatype}, Loss: {loss}', fontsize=12)

        # Save the figures
        if self.save_path is not None:
            self.__savefig__(figure_diffs, f'{save_prefix}_{self.datatype}_{loss}{timestamp}_diffs.png')
            self.__savefig__(bilby_figure_diffs, f'{save_prefix}_{self.datatype}_{loss}{timestamp}_bilby_diffs.png')
            # self.__savefig__(figure_uncertainties, f'{save_prefix}_{self.datatype}_{loss}{timestamp}_uncertainties.png')
            self.__savefig__(figure_z_scores, f'{save_prefix}_{self.datatype}_{loss}{timestamp}_z_scores.png')
            self.__savefig__(figure_loss, f'{save_prefix}_{self.datatype}_{loss}{timestamp}_hist_loss_per_sample.png')
        
        plt.close()

        print(f'Saved SSM predictions plots to {self.save_path}')
        print(f'SSM predictions computed with loss={loss}')

        return

    def compare_ssm_predictions(
            self,
            d_model1,
            n_layers1,
            d_model2,
            n_layers2,
            data1_path='',
            data2_path='',
            model1_path='',
            model2_path='',
            save_prefix1='ssm1',
            save_prefix2='ssm2',
            loss='NLLGaussian',
            csv_output=True,
            csv_fname1='',
            csv_fname2=''):
        timestamp1 = extract_timestamp(model1_path, sep='_')
        timestamp2 = extract_timestamp(model2_path, sep='_')
        from models import S4Model
        if loss=='NLLGaussian': d_output = 4
        elif loss=='Quantile': d_output = 6
        else: raise ValueError(f'Unknown loss function: {loss}')
        if not os.path.exists(model1_path):
            raise ValueError(f"Model path '{model1_path}' does not exist or was not provided.")
        if not os.path.exists(model2_path):
            raise ValueError(f"Model path '{model2_path}' does not exist or was not provided.")
        ssm_model1 = S4Model(d_input=1, d_output=d_output, d_model=d_model1,
            loss=loss, n_layers=n_layers1, dropout=0.0, prenorm=False)
        ssm_model2 = S4Model(d_input=1, d_output=d_output, d_model=d_model2,
            loss=loss, n_layers=n_layers2, dropout=0.0, prenorm=False)
        ssm_model1 = ssm_model1.to(self.device)
        ssm_model2 = ssm_model2.to(self.device)
        ssm_model1.load_state_dict(
            torch.load(model1_path, map_location=self.device, weights_only=True))
        ssm_model2.load_state_dict(
            torch.load(model2_path, map_location=self.device, weights_only=True))
        ssm_model1.eval()
        ssm_model2.eval()

        self.__loaddata__(data1_path)
        ssm_outputs1 = self.ssm_compute_vals(ssm_model1, loss=loss, batch_size=1000, csv_fname=csv_fname1,
            compute_on_cpu=False, csv_output=csv_output, save_prefix=save_prefix1, timestamp=timestamp1)
        self.__loaddata__(data2_path)
        ssm_outputs2 = self.ssm_compute_vals(ssm_model2, loss=loss, batch_size=1000, csv_fname=csv_fname2,
            compute_on_cpu=False, csv_output=csv_output, save_prefix=save_prefix2, timestamp=timestamp2)
        
        ssm1_param1_diff, ssm1_param1_z_score = self.compute_z_scores(ssm_outputs1['pred_param1'], ssm_outputs1['pred_sigma1'], ssm_outputs1['truth_param1'])
        ssm1_param2_diff, ssm1_param2_z_score = self.compute_z_scores(ssm_outputs1['pred_param2'], ssm_outputs1['pred_sigma2'], ssm_outputs1['truth_param2'])
        ssm2_param1_diff, ssm2_param1_z_score = self.compute_z_scores(ssm_outputs2['pred_param1'], ssm_outputs2['pred_sigma1'], ssm_outputs2['truth_param1'])
        ssm2_param2_diff, ssm2_param2_z_score = self.compute_z_scores(ssm_outputs2['pred_param2'], ssm_outputs2['pred_sigma2'], ssm_outputs2['truth_param2'])

        # Stack into (N, 2) array
        ssm1_diffs_stacked, ssm1_z_scores_stacked, ssm1_uncertainties_stacked =\
            np.stack([ssm1_param1_diff.numpy(), ssm1_param2_diff.numpy()], axis=1),\
            np.stack([ssm1_param1_z_score.numpy(), ssm1_param2_z_score.numpy()],axis=1),\
            np.stack([ssm_outputs1['pred_sigma1'].numpy(), ssm_outputs1['pred_sigma2'].numpy()],axis=1)
        ssm2_diffs_stacked, ssm2_z_scores_stacked, ssm2_uncertainties_stacked =\
            np.stack([ssm2_param1_diff.numpy(), ssm2_param2_diff.numpy()], axis=1),\
            np.stack([ssm2_param1_z_score.numpy(), ssm2_param2_z_score.numpy()],axis=1),\
            np.stack([ssm_outputs2['pred_sigma1'].numpy(), ssm_outputs2['pred_sigma2'].numpy()],axis=1)

        # Prepare labels for the plots
        if self.datatype == 'SHO':
            labels_diffs = [r'$\hat{\omega}_0 - \omega_0$', r'$\hat{\beta} - \beta$']
            labels_uncertainties = [r'$\hat{\sigma}_{\omega_0}$', r'$\hat{\sigma}_{\beta}$']
        elif self.datatype == 'SineGaussian':
            labels_diffs = [r'$\hat{f}_0 - f_0$', r'$\hat{\tau} - \tau$']
            labels_uncertainties = [r'$\hat{\sigma}_{f_0}$', r'$\hat{\sigma}_{\tau}$']
        labels_z_scores = [f'({labels_diffs[i]})/{labels_uncertainties[i]}' for i in range(len(labels_diffs))]

        # Plot uncertainties
        figure_uncertainties = corner.corner(
            ssm1_uncertainties_stacked,
            labels=labels_uncertainties,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.2f',
            color='blue'
        )
        corner.corner(
            ssm2_uncertainties_stacked,
            fig=figure_uncertainties,
            title_fmt='.2f',
            color='purple'
        )
        plt.legend(
            handles=[
                mlines.Line2D([], [], color='blue', label=save_prefix1),
                mlines.Line2D([], [], color='purple', label=save_prefix2),
            ],
            fontsize=14, frameon=False,
            bbox_to_anchor=(1.1, 2.2), loc="upper right"
        )
        figure_uncertainties.suptitle(f'Data: {self.datatype}, Loss: {loss}', fontsize=12)
        figure_uncertainties.subplots_adjust(top=0.87)

        corner_kwargs = dict(
            smooth=0.8,
            label_kwargs=dict(fontsize=14),
            labelpad=-0.13,
            title_kwargs=dict(fontsize=14),
            tick_params=dict(labelsize=14),
            # quantiles=[0.1587, 0.5, 0.8413],
            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)), # 1, 2, 3 sigmas
            # levels=(0.1587, 0.5, 0.8413),
            plot_density=True,
            plot_datapoints=False,
            fill_contours=False,
            show_titles=False,
            title_fmt='.2f',
            max_n_ticks=3,
            verbose=False
        )

        # Plot z_scores
        figure_z_scores = corner.corner(
            ssm1_z_scores_stacked,
            labels=labels_z_scores,
            color='blue',
            range=[[-5, 5], [-5, 5]],
            **corner_kwargs
        )
        corner.corner(
            ssm2_z_scores_stacked,
            fig=figure_z_scores,
            labels=labels_z_scores,
            color='purple',
            range=[[-5, 5], [-5, 5]],
            **corner_kwargs
        )
        figure_z_scores.suptitle(f'{self.datatype}', fontsize=14)
        figure_z_scores.subplots_adjust(top=0.87)
        plt.legend(
            handles=[
                mlines.Line2D([], [], color='blue', label=save_prefix1),
                mlines.Line2D([], [], color='purple', label=save_prefix2),
            ],
            fontsize=14, frameon=False,
            bbox_to_anchor=(1.1, 2.2), loc="upper right"
        )

        # Plot diffs
        figure_diffs = corner.corner(
            ssm2_diffs_stacked,
            labels=labels_diffs,
            color='purple',
            range=[[-1, 1], [-1, 1]],
            **corner_kwargs
        )
        corner.corner(
            ssm1_diffs_stacked,
            labels=labels_diffs,
            fig=figure_diffs,
            color='blue',
            range=[[-1, 1], [-1, 1]],
            **corner_kwargs
        )
        figure_diffs.suptitle(f'{self.datatype}', fontsize=14)
        figure_diffs.subplots_adjust(top=0.87)
        plt.legend(
            handles=[
                mlines.Line2D([], [], color='blue', label=save_prefix1),
                mlines.Line2D([], [], color='purple', label=save_prefix2),
            ],
            fontsize=14, frameon=False,
            bbox_to_anchor=(1.1, 2.2), loc="upper right"
        )

        # Save the figures
        if self.save_path is not None:
            self.__savefig__(figure_uncertainties, f'{save_prefix1}_{save_prefix2}_{self.datatype}_{loss}_uncertainties.png')
            self.__savefig__(figure_z_scores, f'{save_prefix1}_{save_prefix2}_{self.datatype}_{loss}_z_scores.png')
            self.__savefig__(figure_diffs, f'{save_prefix1}_{save_prefix2}_{self.datatype}_{loss}_diffs.png')
        
        plt.close()

        print(f'Saved SSM predictions plots to {self.save_path}')
        print(f'SSM predictions computed with loss={loss}')

        return

    def get_bilby_results(self, bilby_dir='/ceph/submit/data/user/k/kyoon/KYoonStudy/fitresults/bilby_mcmc',
                          save_tag='mcmc'):
        """
        Get bilby results.
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
        parquet_name = os.path.join(bilby_dir, f'{self.datatype}_bilby_{save_tag}_combined.parquet')
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
        # Plotting logic goes here
        if self.datatype=='SHO': params = ['omega_0', 'beta']
        elif self.datatype=='SineGaussian': params = ['f_0', 'tau']
        bilby_outputs = {
            'pred_omega': combined_df.loc[combined_df['stat'] == 'mean', params[0]].values,
            'truth_omega': combined_df.loc[combined_df['stat'] == 'truth', params[0]].values,
            'pred_beta': combined_df.loc[combined_df['stat'] == 'mean', params[1]].values,
            'truth_beta': combined_df.loc[combined_df['stat'] == 'truth', params[1]].values,
            'pred_sigma_omega': combined_df.loc[combined_df['stat'] == 'std', params[0]].values,
            'pred_sigma_beta': combined_df.loc[combined_df['stat'] == 'std', params[1]].values,
        }
        return bilby_outputs

if __name__ == "__main__":
    sho_flow_params = {
        'embed_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO/output/embedding.CNN.SHO.250804212612.pt',
        'flow_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO/output/flow.CNN.SHO.embed250804212612.250805000316.pt',
        'flow_hidden_features': 30,
        'embed_hidden_layers': 2,
        'embed_hidden_channels': 20,
        'embed_kernel_size': 21,
        'embed_d_output': 6,
        'num_points': 200
    }
    sg_flow_params = {
        'embed_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/embedding.CNN.SineGaussian.250804212612.pt',
        'flow_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/flow.CNN.SineGaussian.embed250804212612.250805000316.pt',
        'flow_hidden_features': 20,
        'embed_hidden_layers': 2,
        'embed_hidden_channels': 10,
        'embed_kernel_size': 11,
        'embed_d_output': 12,
        'num_points': 500
    }
    sho_ssm_nllgaussian_params = {
        'model_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/saved_models/DHO/model.SSM.SHO.NLLGaussian.250827165455.pink.noise.sigma0.4.pt',
        'd_model': 6,
        'n_layers': 4,
        'save_prefix': 'ssm_d6_n4',
        'loss': 'NLLGaussian',
        'csv_output': True,
        'plot_flow': True
    }
    sho_ssm_quantile_params = {
        'model_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/saved_models/DHO/model.SSM.SHO.Quantile.250828225103.pink.noise.sigma0.4.pt',
        'd_model': 6,
        'n_layers': 4,
        'save_prefix': 'ssm_d6_n4',
        'loss': 'Quantile',
        'csv_output': True,
        'plot_flow': True
    }
    sg_ssm_nllgaussian_params = {
        'model_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/saved_models/SG/model.SSM.SineGaussian.NLLGaussian.250828213816.pink.noise.sigma0.4.pt',
        'd_model': 6,
        'n_layers': 4,
        'save_prefix': 'ssm_d6_n4',
        'loss': 'NLLGaussian',
        'csv_output': True,
        'plot_flow': True
    }
    sg_ssm_quantile_params = {
        'model_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/saved_models/SG/model.SSM.SineGaussian.Quantile.250828225108.pink.noise.sigma0.4.pt',
        'd_model': 6,
        'n_layers': 4,
        'save_prefix': 'ssm_d6_n4',
        'loss': 'Quantile',
        'csv_output': True,
        'plot_flow': True
    }
    # sho_flow_params = {
    #     'embed_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO/output/embedding.CNN.SHO.250804212612.pt',
    #     'flow_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO/output/flow.CNN.SHO.embed250804212612.250805000316.pt',
    #     'flow_hidden_features': 30,
    #     'embed_hidden_layers': 2,
    #     'embed_hidden_channels': 20,
    #     'embed_kernel_size': 21,
    #     'embed_d_output': 6,
    #     'num_points': 200
    # }
    # sg_flow_params = {
    #     'embed_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/embedding.CNN.SineGaussian.250804212612.pt',
    #     'flow_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/flow.CNN.SineGaussian.embed250804212612.250805000316.pt',
    #     'flow_hidden_features': 20,
    #     'embed_hidden_layers': 2,
    #     'embed_hidden_channels': 10,
    #     'embed_kernel_size': 11,
    #     'embed_d_output': 12,
    #     'num_points': 500
    # }
    # sho_ssm_nllgaussian_params = {
    #     'model_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO/output/model.SSM.SHO.NLLGaussian.250801140434.path',
    #     'csv_fname': '/ceph/submit/data/user/k/kyoon/KYoonStudy/plots/s4d_SHO_NLLGaussian_Benedict.csv',
    #     'd_model': 6,
    #     'n_layers': 4,
    #     'save_prefix': 'ssm_d6_n4',
    #     'loss': 'NLLGaussian',
    #     'csv_output': True,
    #     'plot_flow': True
    # }
    # sho_ssm_quantile_params = {
    #     'model_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO/output/model.SSM.SHO.Quantile.250801100654.path',
    #     'csv_fname': '/ceph/submit/data/user/k/kyoon/KYoonStudy/plots/s4d_SHO_Quantile_Benedict.csv',
    #     'd_model': 6,
    #     'n_layers': 4,
    #     'save_prefix': 'ssm_d6_n4',
    #     'loss': 'Quantile',
    #     'csv_output': True,
    #     'plot_flow': True
    # }
    # sg_ssm_nllgaussian_params = {
    #     'model_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/model.SSM.SineGaussian.NLLGaussian.250801142022.path',
    #     'd_model': 6,
    #     'n_layers': 4,
    #     'save_prefix': 'ssm_d6_n4',
    #     'loss': 'NLLGaussian',
    #     'csv_output': True,
    #     'plot_flow': True
    # }
    # sg_ssm_quantile_params = {
    #     'model_path': '/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/model.SSM.SineGaussian.Quantile.250801101817.path',
    #     'd_model': 6,
    #     'n_layers': 4,
    #     'save_prefix': 'ssm_d6_n4',
    #     'loss': 'Quantile',
    #     'csv_output': True,
    #     'plot_flow': True
    # }
    
    # plotter = Plotter(datatype='SHO', datasfx='_dho_gaussian_pink_sigma0.4')
    # plotter.plot_embeddings(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO/output/embedding.CNN.SHO.250720075718.pt',
    #                         num_hidden_layers_h=2)
    # plotter.plot_flow(**sho_flow_params)
    # plotter.plot_ssm_predictions(**sho_ssm_nllgaussian_params, **sho_flow_params)
    # plotter.plot_ssm_predictions(**sho_ssm_quantile_params, **sho_flow_params)

    # plotter = Plotter(datatype='SineGaussian', datasfx='_sg_gaussian_pink_sigma0.4')
    # plotter.plot_embeddings(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO/output/embedding.CNN.SHO.250720075718.pt',
    #                         num_hidden_layers_h=2)
    # plotter.plot_flow(**sho_flow_params)
    # plotter.plot_ssm_predictions(**sg_ssm_nllgaussian_params, **sg_flow_params)
    # plotter.plot_ssm_predictions(**sg_ssm_quantile_params, **sg_flow_params)
    
    # plotter = Plotter(datatype='SineGaussian', datasfx='_sigma0.4_gaussian')
    # plotter.plot_embeddings(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/embedding.CNN.SineGaussian.250720075718.pt',
    #                         num_hidden_layers_h=2)
    # plotter.plot_flow(**sg_flow_params)
    # plotter.plot_ssm_predictions(**sg_ssm_nllgaussian_params, **sg_flow_params)
    # plotter.plot_ssm_predictions(**sg_ssm_quantile_params, **sg_flow_params)

    plotter = Plotter(datatype='SHO', datasfx='_dho_gaussian_pink_sigma0.4')
    plotter.compare_ssm_predictions(
        d_model1=6, d_model2=6, n_layers1=4, n_layers2=4,
        data1_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/data/DHO/test_dho_gaussian_smear_sigma0.4.pt',
        # data2_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/data/DHO/test_dho_gaussian_smear_sigma0.4.pt',
        data2_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/data/DHO/test_dho_gaussian_pink_sigma0.4.pt',
        model1_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SHO/output/model.SSM.SHO.NLLGaussian.250801140434.path',
        model2_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/saved_models/DHO/model.SSM.SHO.NLLGaussian.250827165455.pink.noise.sigma0.4.pt',
        save_prefix1='white_noise',
        save_prefix2='pink_noise'
    )
    plotter = Plotter(datatype='SineGaussian', datasfx='_sg_gaussian_pink_sigma0.4')
    plotter.compare_ssm_predictions(
        d_model1=6, d_model2=6, n_layers1=4, n_layers2=4,
        data1_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/data/SG/test_sg_gaussian_smear_sigma0.4.pt',
        # data2_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/data/SG/test_sg_gaussian_smear_sigma0.4.pt',
        data2_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/data/SG/test_sg_gaussian_pink_sigma0.4.pt',
        model1_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/models/SineGaussian/output/model.SSM.SineGaussian.NLLGaussian.250801142022.path',
        model2_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/neurips2025/saved_models/SG/model.SSM.SineGaussian.NLLGaussian.250828213816.pink.noise.sigma0.4.pt',
        save_prefix1='white_noise',
        save_prefix2='pink_noise'
    )