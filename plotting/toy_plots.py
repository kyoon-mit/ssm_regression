import os
import numpy as np
import corner

def savefig(plot, savedir, name):
    path = os.path.join(savedir, name)
    plot.savefig(path, bbox_inches='tight')
    print(f'Saved {path}.')
    return

def toy_plots(outputs, targets, datatype, prefix='plot', savedir='.'):
    """Make all plots and save in directory.

    Args:
        outputs (numpy.ndarray of shape (num_samples, 4)): Model output.
            Four parameters are param_1, param_2, uncertainty_1, uncertainty_2
        targets (numpy.ndarray of shape (num_samples, 2)): Truth values.
            truth_1, truth_2
        datatype (str): 'dho' or 'sg'
        prefix (str): Prefix for the file names to be saved.
        savedir (str): Saving directory

    Return:
        None
    """
    preds, sigmas = outputs[:,:2], outputs[:,2:4]
    diffs = preds - targets
    z_scores = (preds-targets)/sigmas

    # Create labels
    if datatype=='dho':
        labels_diffs = [r'$\hat{\omega}_0 - \omega_0$', r'$\hat{\beta} - \beta$']
        labels_sigmas = [r'$\hat{\sigma}_{\omega_0}$', r'$\hat{\sigma}_{\beta}$']
    elif datatype=='sg':
        labels_diffs = [r'$\hat{f}_0 - f_0$', r'$\hat{\tau} - \tau$']
        labels_sigmas = [r'$\hat{\sigma}_{f_0}$', r'$\hat{\sigma}_{\tau}$']
    else: raise ValueError(f'Unknown {datatype=}.')
    labels_z_scores = [f'({labels_diffs[i]})/{labels_sigmas[i]}' for i in range(len(labels_diffs))]

    # Plot
    corner_kwargs = dict(
        show_title=True,
        smooth=0.8,
        label_kwargs=dict(fontsize=14),
        labelpad=-0.13,
        title_kwargs=dict(fontsize=14),
        tick_params=dict(labelsize=14),
        quantiles=[0.1587, 0.5, 0.8413],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)), # 1, 2, 3 sigmas
        plot_density=True,
        plot_datapoints=False,
        fill_contours=False,
        show_titles=False,
        title_fmt='.3f',
        max_n_ticks=3,
        range=[[-5, 5], [-5, 5]],
        verbose=False
    )
    figure_diffs = corner.corner(
        diffs,
        labels=labels_diffs,
        color='black',
        **corner_kwargs
    )
    figure_sigmas = corner.corner(
        sigmas,
        labels=labels_sigmas,
        color='blue',
        **corner_kwargs
    )
    figure_z_scores = corner.corner(
        z_scores,
        labels=labels_z_scores,
        color='red',
        **corner_kwargs
    )
    figure_diffs.suptitle(f'diffs: {datatype}', fontsize=14)
    figure_sigmas.suptitle(f'sigmas: {datatype}', fontsize=14)
    figure_z_scores.suptitle(f'z_scores: {datatype}', fontsize=14)
    savefig(figure_diffs, savedir, f'{prefix}_diffs.png')
    savefig(figure_sigmas, savedir, f'{prefix}_sigmas.png')
    savefig(figure_z_scores, savedir, f'{prefix}_z_scores.png')
    return