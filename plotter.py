import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import corner
from matplotlib.patches import Patch

import sys

print(sys.executable)
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

class Plotter:
    def __init__(
        self,
        save_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/plots',
        datatype='SineGaussian', # 'SineGaussian', 'SHO', or 'LIGO'
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device={self.device}')

        # Load datasets
        if datatype == 'SineGaussian':
            from data_sinegaussian import DataGenerator
        elif datatype == 'SHO':
            from data_sho import DataGenerator
        elif datatype == 'LIGO':
            pass  # TODO: Placeholder for LIGO data generator, implement as needed
        else:
            raise ValueError(f'Unknown datatype: {datatype}')
        self.datatype = datatype
        self.datadir = f'/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/{datatype}'
        self.train_dict = torch.load(os.path.join(self.datadir, 'train.pt'), map_location=torch.device(self.device))
        self.train_data = DataGenerator(self.train_dict)

        self.val_dict   = torch.load(os.path.join(self.datadir, 'val.pt'), map_location=torch.device(self.device))
        self.val_data   = DataGenerator(self.val_dict)

        self.test_dict  = torch.load(os.path.join(self.datadir, 'test.pt'), map_location=torch.device(self.device))
        self.test_data  = DataGenerator(self.test_dict)

        self.test_data_loader = DataLoader(
            self.test_data, batch_size=1, num_workers=0,
            shuffle=False
        )

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            print(f'Created save path: {self.save_path}')
        
        # Placeholders
        self.embedding_model = nn.Module()  # Placeholder for the model, replace with actual model loading
        self.ssm_model = nn.Module()

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
        for _, theta_test, data_test, data_test_orig in self.test_data_loader:
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
        for _, theta_test, data_test, data_test_orig in self.test_data_loader:
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
        self.embedding_model = torch.load(model_path, map_location=self.device)

        from models import SimilarityEmbedding
        # Initialize the model with the specified number of hidden layers
        if num_hidden_layers_h <= 0:
            raise ValueError("Number of hidden layers must be greater than 0.")
        print(f'Embeddings: Loaded model from {model_path}')
        similarity_embedding = SimilarityEmbedding(num_hidden_layers_h=2).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
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
        fig = self.make_embedding_plots(
            outputs_by_interval=similarity_outputs,
            legends=legends,
            save_name=f'{self.datatype}_embeddings_plot.png',
            num_dim=3,
            title=plot_title
        )
        return

    def compute_z_scores(self, predictions, truths):
        mean = predictions.mean()
        std = predictions.std()
        z_scores = (predictions - truths) / std
        return z_scores

    def ssm_reshaping(self, batch, input_dim=1, output_dim=2):
        theta_u, theta_s, data_u, data_s = batch

        # remove repeat (take only first repeat for unshifted data)
        if input_dim==1:
            inputs = data_u[:, 0, :].unsqueeze(-1) if data_u.ndim == 3 else data_u.unsqueeze(-1)  # [B, 200, input_dim]
        else:
            inputs = data_u[:, 0, :input_dim] if data_u.ndim == 3 else data_u[:, :input_dim]
        targets = theta_u[:, 0, :output_dim] if theta_u.ndim == 3 else theta_u[:, :output_dim]  # [B, output_dim]

        return inputs, targets

    def ssm_compute_vals(self, ssm_model, loss='NLLGaussian', batch_size=16, compute_on_cpu=False):
        pred_param1, truth_param1 = [], []
        pred_param2, truth_param2 = [], []
        if loss=='NLLGaussian':
            pred_sigma1, pred_sigma2 = [], [] # uncertainties
        elif loss=='Quantile':
            pred_q25_1, pred_q25_2 = [], []  # 25th quantile
            pred_q75_1, pred_q75_2 = [], []  # 75th quantile
        else:
            raise ValueError(f'Unknown loss function: {loss}')

        batch_contexts, batch_truths = [], []

        device = next(ssm_model.parameters()).device
        ssm_model.eval()

        # Redefine the test data loader to iterate over batches
        self.test_data_loader = DataLoader(
            self.test_data, batch_size=256, num_workers=0,
            shuffle=False
        )

        for idx, vals in enumerate(self.test_data_loader):
            inputs, truths = self.ssm_reshaping(vals, output_dim=2)
            batch_contexts.append(inputs)  # shape: (1, 1, 200)
            batch_truths.append(truths)  # shape: (1, 2)

            # Process in batches
            if (idx + 1) % batch_size == 0 or (idx + 1) == len(self.test_data_loader):
                contexts = torch.cat(batch_contexts, dim=0).to(device)   # (B, 1, 200)
                truths = torch.cat(batch_truths).to(device)                 # (B, 2)

                with torch.no_grad():
                    preds = ssm_model(contexts)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)  # e.g., (B, 1, N) → (B, N)

                # Move samples to CPU if required
                if compute_on_cpu:
                    preds = preds.cpu()
                    truths = truths.cpu()                       # (B, 2)
                
                pred_param1.extend(preds[:, 0].tolist())
                pred_param2.extend(preds[:, 1].tolist())
                truth_param1.extend(truths[:, 0].tolist())
                truth_param2.extend(truths[:, 1].tolist())

                if loss == 'NLLGaussian':
                    pred_sigma1.extend(preds[:, 2].tolist())
                    pred_sigma2.extend(preds[:, 3].tolist())
                elif loss == 'Quantile':
                    pred_q25_1.extend(preds[:, 2].tolist())
                    pred_q25_2.extend(preds[:, 3].tolist())
                    pred_q75_1.extend(preds[:, 4].tolist())
                    pred_q75_2.extend(preds[:, 5].tolist())

                # Clear for next batch
                batch_contexts.clear()
                batch_truths.clear()

        return_dict = {
            'pred_param1': torch.tensor(pred_param1),
            'truth_param1': torch.tensor(truth_param1),
            'pred_param2': torch.tensor(pred_param2),
            'truth_param2': torch.tensor(truth_param2),
        }
        if loss == 'NLLGaussian':
            return_dict['pred_sigma1'] = torch.tensor(pred_sigma1)
            return_dict['pred_sigma2'] = torch.tensor(pred_sigma2)
        elif loss == 'Quantile':
            return_dict['pred_q25_1'] = torch.tensor(pred_q25_1)
            return_dict['pred_q25_2'] = torch.tensor(pred_q25_2)
            return_dict['pred_q75_1'] = torch.tensor(pred_q75_1)
            return_dict['pred_q75_2'] = torch.tensor(pred_q75_2)

        return return_dict

    def plot_ssm_predictions(self, model_path='', save_prefix='ssm', loss='NLLGaussian'):
        from models import S4Model
        if loss=='NLLGaussian':
            d_input, d_output, d_model, n_layers = 1, 4, 6, 4
        elif loss=='Quantile':
            d_input, d_output, d_model, n_layers = 1, 6, 6, 4
        else:
            raise ValueError(f'Unknown loss function: {loss}')
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model path '{model_path}' does not exist or was not provided.")
        self.ssm_model = S4Model(d_input=d_input, d_output=d_output, d_model=d_model, n_layers=n_layers, dropout=0.0, prenorm=False)
        self.ssm_model = self.ssm_model.to(self.device)
        self.ssm_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.ssm_model.eval()

        ssm_outputs = self.ssm_compute_vals(self.ssm_model, loss=loss, batch_size=16, compute_on_cpu=False)

        # Compute differences between predictions and truths
        ssm_diffs = {
            'param1_diff': ssm_outputs['pred_param1'] - ssm_outputs['truth_param1'],
            'param2_diff': ssm_outputs['pred_param2'] - ssm_outputs['truth_param2'],
        }

        # Stack into (N, 2) array
        ssm_diffs_stacked = np.stack(
            [ssm_diffs['param1_diff'].numpy(), ssm_diffs['param2_diff'].numpy()],
            axis=1
        )

        # Create a corner plot
        if self.datatype=='SHO':
            labels_diffs = [r'$\Delta \omega_0$', r'$\Delta \beta$']
        elif self.datatype=='SineGaussian':
            labels_diffs = [r'$\Delta f_0$', r'$\Delta \tau$']
        figure_diffs = corner.corner(
            ssm_diffs_stacked,
            labels=labels_diffs,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.2f',
            color='C1'
        )

        # Plot uncertainties
        if loss == 'NLLGaussian':
            uncertainties = np.sqrt(np.exp(
                np.stack(
                    [ssm_outputs['pred_sigma1'].numpy(), ssm_outputs['pred_sigma2'].numpy()],
                    axis=1
                )
            ))
        elif loss == 'Quantile':
            uncertainties = np.stack(
                [ssm_outputs['pred_q75_1'].numpy() - ssm_outputs['pred_q25_1'].numpy(),
                 ssm_outputs['pred_q75_2'].numpy() - ssm_outputs['pred_q25_2'].numpy()],
                axis=1
            )
        if self.datatype == 'SHO':
            labels_uncertainties = [r'$\sigma_{\omega_0}$', r'$\sigma_{\beta}$']
        elif self.datatype == 'SineGaussian':
            labels_uncertainties = [r'$\sigma_{f_0}$', r'$\sigma_{\tau}$']
        figure_uncertainties = corner.corner(
            uncertainties,
            labels=labels_uncertainties,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 12},
            title_fmt='.2f',
            color='C3'
        )

        # Save the figures
        if self.save_path is not None:
            figure_diffs.savefig(os.path.join(self.save_path, f'{save_prefix}_{self.datatype}_diffs.png'), bbox_inches='tight')
            figure_uncertainties.savefig(os.path.join(self.save_path, f'{save_prefix}_{self.datatype}_uncertainties_{loss}.png'), bbox_inches='tight')
        else:
            figure_diffs.tight_layout()
            figure_diffs.show()
            figure_uncertainties.tight_layout()
            figure_uncertainties.show()

        return

if __name__ == "__main__":
    # plotter = Plotter(datatype='SHO')
    # plotter.plot_embeddings(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SHO/models/model.CNN.20250503-220716.path',
    #                         num_hidden_layers_h=2)
    # plotter.plot_ssm_predictions(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SHO/models/model.SSM.SHO.NLLGaussian.20250528-005313.path',
    #                              save_prefix='ssm', loss='NLLGaussian')
    # plotter.plot_ssm_predictions(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SHO/models/model.SSM.SHO.Quantile.20250528-013120.path',
    #                              save_prefix='ssm', loss='Quantile')
    plotter = Plotter(datatype='SineGaussian')
    # plotter.plot_embeddings(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SHO/models/model.CNN.20250503-220716.path',
    #                         num_hidden_layers_h=2)
    plotter.plot_ssm_predictions(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SineGaussian/models/model.SSM.SineGaussian.NLLGaussian.20250527-211554.path',
                                 save_prefix='ssm', loss='NLLGaussian')
    plotter.plot_ssm_predictions(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SineGaussian/models/model.SSM.SineGaussian.Quantile.20250528-005641.path',
                                 save_prefix='ssm', loss='Quantile')
    # Replace '/path/to/your/model.pth' with the actual path to your trained model