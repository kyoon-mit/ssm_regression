import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader

logger = logging.getLogger('fitting')

class Fitting:
    """
    Class for fitting models with configurable parameters.
    """
    def __init__(
        self,
        datatype,
        default_shift=1,
        device=None,
        batch_indices=(0, None),
        **kwargs
    ):
        """
        Parameters
        ----------
        datatype : str
            Data type or dataset. Must be one of {'SHO', 'SineGaussian', 'LIGO'}.
        default_shift : int, optional
            Default shift value. Default is 1.
        device : str or None, optional
            Device to use ('cpu', 'cuda', or None). If None, automatically selects 'cuda' if available, else 'cpu'.
        batch_indices : tuple, optional
            Indices for test data. Default is (0, None).

        Other Parameters
        ----------------
        n_repeats : int, optional
            Number of repeats. Default is 10.
        num_points : int, optional
            Number of points. Default is 200.

        Raises
        ------
        ValueError
            If `datatype` is not one of the supported types.
        """
        logger.info(f'Instantiated Fitting() with parameters:\n'
            f'{datatype=}, {default_shift=}, '
            f'{device=}, {batch_indices=}, '
            f'{", ".join([f"{k}={v}" for k, v in kwargs.items()])}')
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f'Using device: {self.device}')
        self.datatype = datatype
        if self.datatype=='SHO':
            from data_sho import damped_sho_np as func
            from data_sho import DataGenerator
        elif self.datatype=='SineGaussian':
            from data_sinegaussian import sine_gaussian_np as func
            from data_sinegaussian import DataGenerator
        elif self.datatype=='LIGO':
            pass #TODO
        else:
            msg = f'Unknown datatype: {self.datatype}'
            logger.error(msg)
            raise ValueError(msg)
        self.func = func
        self.basedir = kwargs.get('basedir', f'/ceph/submit/data/user/k/kyoon/KYoonStudy')
        self.modeldir = kwargs.get('modeldir', os.path.join(self.basedir, 'models', self.datatype))
        self.savedir = kwargs.get('savedir', os.path.join(self.basedir, 'fitresults'))
        self.test_dict = torch.load(os.path.join(self.modeldir, kwargs.get('testfile', 'test.pt')),
                                    map_location=self.device, weights_only=True)
        self.start_idx = 0 if batch_indices[0] is None else batch_indices[0]
        self.end_idx = self.test_dict['data_unshifted'].shape[0] if batch_indices[1] is None else batch_indices[1]
        self.test_data = DataGenerator(self.test_dict)
        self.test_data_subset = Subset(self.test_data, list(range(self.start_idx, self.end_idx)))
        self.test_dataloader = DataLoader(
            self.test_data_subset,
            batch_size=1,
            shuffle=False
        )
        self.shift = default_shift
        self.num_points = kwargs.get('num_points', 200)
        self.n_repeats = kwargs.get('n_repeats', 10)
        self.sigma = kwargs.get('sigma', 0.4)

        self.t_vals_np = np.linspace(start=-1, stop=10, num=self.num_points)

    def summarize_bilby_event(self, result, truth, event_id):
        desc = result.posterior.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        stats_df = desc.loc[['count', 'mean', 'std', '5%', '25%', '50%', '75%', '95%']]
        stats_df['stat'] = stats_df.index
        stats_df['event_id'] = event_id
        result.plot_corner()
        # append truth info
        truth_row = {'stat': 'truth', 'event_id': event_id}
        if self.datatype=='SHO':
            truth_row['omega_0'] = truth[0]
            truth_row['beta'] = truth[1]
        elif self.datatype=='SineGaussian':
            truth_row['f_0'] = truth[0]
            truth_row['tau'] = truth[1]
        stats_df = pd.concat([stats_df, pd.DataFrame([truth_row])])
        return stats_df.set_index('event_id', append=False)

    def summarize_lmfit_event(self, result, truth, event_id):
        """
        Summarize the results of an lmfit fit for a single event.
        Parameters
        ----------
        result : lmfit.model.ModelResult
            The result object from lmfit.
        truth : np.ndarray
            The ground truth parameters for the event.
        event_id : int
            The ID of the event being summarized.
        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing the fit results and truth values.
        """
        summary = {}
        result_dict = result.summary()
        var_names = result_dict['var_names']
        params = result_dict['params']
        for vn in var_names:
            # params is a list of tuples, so we need to search for the tuple with name == vn
            row = {}
            param_tuple = next((p for p in params if p[0] == vn), None)
            if param_tuple is not None:
                row['mean'] = param_tuple[1]
                row['std']  = param_tuple[7]
                summary[vn] = row
        # Convert summary to DataFrame
        stats_df = pd.DataFrame(summary)
        stats_df['stat'] = stats_df.index
        stats_df['event_id'] = event_id
        # append truth info
        truth_row = {'stat': 'truth', 'event_id': event_id}
        if self.datatype=='SHO':
            truth_row['omega_0'] = truth[0]
            truth_row['beta'] = truth[1]
        elif self.datatype=='SineGaussian':
            truth_row['f_0'] = truth[0]
            truth_row['tau'] = truth[1]
        stats_df = pd.concat([stats_df, pd.DataFrame([truth_row])])
        return stats_df.set_index('event_id', append=False)

    def run_lmfit(self, max_events=None, max_workers=4):
        from lmfit import Model, Parameters
        model = Model(self.func)
        params = Parameters()
        params.add('shift', value=1.0, vary=False) # Default shift value
        if self.datatype=='SHO':
            params.add('omega_0', min=0.1, max=2.0)
            params.add('beta', min=0., max=0.5)
        elif self.datatype=='SineGaussian':
            params.add('f_0', min=0.1, max=1.1)
            params.add('tau', min=1., max=5.)
        elif self.datatype=='LIGO':
            pass # TODO: Implement LIGO model parameter hints
        
        summaries = [] # Container for the event summary dataframes
        for idx, batch in enumerate(self.test_dataloader):
            theta_u, theta_s, data_u, data_s = batch
            event_id = idx + self.start_idx
            truth = theta_u[0][0].to(device='cpu')
            truth_np = truth.numpy()
            y = data_u[0][0].to(device='cpu')
            y_np = y.numpy()
            result = model.fit(y_np, params, t=self.t_vals_np)
            stats_df = self.summarize_lmfit_event(result, truth_np, event_id)
            logger.info(stats_df.head())
            summaries.append(stats_df)
        summary_df = pd.concat(summaries)
        summary_df.to_parquet(os.path.join(self.savedir, f'lmfit_{"sho" if self.datatype=="SHO" else "sg"}', f'{self.datatype}_lmfit_id{self.start_idx:05d}-{self.end_idx:05d}.parquet'))
    
    def run_bilby(self, nlive=1000, sampler='dynesty', max_workers=4):
        import bilby
        from bilby.core.prior import Uniform
        from bilby.core.likelihood import GaussianLikelihood
        priors = {}
        summaries = [] # Container for the event summary dataframes
        if self.datatype=='SHO':
            priors['omega_0'] = Uniform(0.1, 1.9, name='omega_0', latex_label=r'$\omega_0$')
            priors['beta'] = Uniform(0, 0.5, name='beta', latex_label=r'$\beta$')
            injection_parameters = dict(omega_0=1., beta=0.3)
        elif self.datatype=='SineGaussian':
            priors['f_0'] = Uniform(0.1, 1.9, name='f_0', latex_label=r'$f_0$')
            priors['tau'] = Uniform(1., 4., name='tau', latex_label=r'$\tau$')
            injection_parameters = dict(f_0=1., tau=2.5)
        for idx, batch in enumerate(self.test_dataloader):
            theta_u, theta_s, data_u, data_s = batch
            event_id = idx + self.start_idx
            truth = theta_u[0][0].to(device='cpu')
            truth_np = truth.numpy()
            y = data_u[0][0].to(device='cpu')
            y_np = y.numpy()
            log_l = GaussianLikelihood(x=self.t_vals_np, y=y_np, func=self.func,
                                       sigma=self.sigma, shift=self.shift)
            result = bilby.run_sampler(
                likelihood=log_l, priors=priors, sampler=sampler,
                nlive=nlive, npool=max_workers,
                injection_parameters=injection_parameters,
                outdir=os.path.join(self.savedir, 'bilby', f'{self.datatype}_id{event_id:05d}'),
                label=f'{self.datatype}_id{event_id:05d}'
            )
            stats_df = self.summarize_bilby_event(result, truth_np, event_id)
            logger.info(stats_df.head())
            summaries.append(stats_df)
        summary_df = pd.concat(summaries)
        summary_df.to_parquet(os.path.join(self.savedir, 'bilby', f'{self.datatype}_bilby_id{self.start_idx:05d}-{self.end_idx:05d}.parquet'))