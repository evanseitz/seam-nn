import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm, Normalize
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_cluster_summary_matrix(mave_df, maps, clusters, figure, column='Entropy', sort=None, delta=False, mut_rate=10, alphabet=None, sort_index=None):
    """
    Visualize cluster summary matrix using MetaExplainer and Clusterer.

    Parameters
    ----------
    mave_df : pandas.DataFrame
        DataFrame containing sequences and their scores.
    maps : np.ndarray
        Attribution maps, shape (n_sequences, seq_length, n_characters).
    clusters : np.ndarray or list
        Cluster labels for each sequence.
    figure : matplotlib.figure.Figure
        Figure to plot on.
    column : str
        Which summary metric to plot ('Entropy', 'Reference', 'Consensus').
    sort : str or None
        Sorting method: 'hierarchical', 'predefined', or None.
    delta : bool
        If True and column='Entropy', plot delta entropy.
    mut_rate : float
        Mutation rate (as a percentage, e.g., 10 for 10%).
    alphabet : list or None
        Alphabet to use. If None, inferred from mave_df.
    sort_index : list or None
        Predefined cluster order if using 'predefined' sort.
    """
    from seam.meta_explainer import MetaExplainer
    from seam.clusterer import Clusterer
    import numpy as np

    nS = len(mave_df)
    if alphabet is None:
        alphabet = sorted(list(set(mave_df['Sequence'][0:100].apply(list).sum())))
    maps_reshaped = maps.reshape((nS, -1, len(alphabet)))
    clusterer = Clusterer(maps_reshaped, method='umap', gpu=True)
    clusterer.cluster_labels = np.array(clusters)
    if sort == 'hierarchical':
        sort_method = 'hierarchical'
    elif sort == 'predefined':
        sort_method = 'predefined'
    else:
        sort_method = None
    meta = MetaExplainer(
        clusterer=clusterer,
        mave_df=mave_df,
        attributions=maps_reshaped,
        sort_method=sort_method,
        mut_rate=mut_rate/100.0,
        alphabet=alphabet
    )
    meta.generate_msm()
    _, _, cluster_order, revels = meta.plot_msm(
        column=column,
        delta_entropy=delta,
        gui=True
    )
    ax = figure.add_subplot(111)
    # Use the same plotting logic as in MetaExplainer.plot_msm (for GUI)
    nC = len(cluster_order)
    nP = revels.shape[1]
    if column == 'Entropy' and delta:
        from matplotlib import colors
        divnorm = colors.TwoSlopeNorm(vmin=revels.min().min(), vcenter=0., vmax=revels.max().max())
        heatmap = ax.pcolormesh(revels, cmap='seismic', norm=divnorm)
        label = 'ΔH (bits)'
    else:
        if column == 'Entropy':
            palette = sns.color_palette('rocket', n_colors=100)
            palette.reverse()
            label = 'Shannon entropy (bits)'
        elif column == 'Reference':
            palette = sns.color_palette('Blues_r', n_colors=100)
            label = 'Mismatches to reference sequence'
        elif column == 'Consensus':
            palette = sns.color_palette('rocket', n_colors=100)
            label = 'Matches to per-cluster consensus'
        heatmap = ax.pcolormesh(revels, cmap=mpl.colors.ListedColormap(list(palette)))
    ax.set_xlabel('Position', fontsize=6)
    ax.set_ylabel('Cluster', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=4)
    ax.set_title('Click on a cell to view its sequence variation', fontsize=5)
    ax.invert_yaxis()
    # X ticks
    if nP > 100:
        x_skip = 10
    elif nP > 1000:
        x_skip = 20
    else:
        x_skip = 1
    xtick_labels = []
    xtick_range = np.arange(0.5, nP, x_skip)
    for i in xtick_range:
        if int(i)%x_skip == 0:
            xtick_labels.append(str(int(i-0.5)))
    ax.set_xticks(xtick_range)
    ax.set_xticklabels(xtick_labels, rotation=0, minor=False)
    # Y ticks
    if nC > 10:
        y_skip = 10
    else:
        y_skip = 1
    ytick_labels = []
    ytick_range = np.arange(0.5, nC, y_skip)
    y_idx = 0
    for i in ytick_range:
        if int(i)%y_skip == 0:
            ytick_labels.append(str(cluster_order[y_idx]))
        y_idx += y_skip
    ax.set_yticks(ytick_range)
    ax.set_yticklabels(ytick_labels, rotation=0, minor=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if column == 'Reference' or column == 'Consensus':
        cbar = figure.colorbar(heatmap, cax=cax, orientation='vertical', format='%.0f%%')
        heatmap.set_clim(0, 100)
    elif column == 'Entropy':
        cbar = figure.colorbar(heatmap, cax=cax, orientation='vertical')
        if not delta:
            heatmap.set_clim(0, 2)
    cbar.ax.set_ylabel(label, rotation=270, fontsize=6, labelpad=9)
    cbar.ax.tick_params(labelsize=6)
    return (ax, cax, cluster_order, revels)