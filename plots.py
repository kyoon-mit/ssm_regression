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
        self.model = nn.Module()  # Placeholder for the model, replace with actual model loading

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
        self.model = torch.load(model_path, map_location=self.device)

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

if __name__ == "__main__":
    plotter = Plotter(datatype='SHO')
    plotter.plot_embeddings(model_path='/ceph/submit/data/user/k/kyoon/KYoonStudy/ssm_regression/SHO/models/model.CNN.20250503-220716.path',
                            num_hidden_layers_h=2)
    # Replace '/path/to/your/model.pth' with the actual path to your trained model