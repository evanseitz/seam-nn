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
import logomaker
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy
import squid.utils as squid_utils # pip install squid-nn
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def plot_logo(logo_df, logo_type, axis=None, aesthetic_lvl=1, ref_seq=None, fontsize=4, figsize=None, y_min_max=None, center_values=True):
    """Function to plot attribution maps or sequence statistics as logos.

    Parameters
    ----------
    logo_df : pandas.dataframe 
        Dataframe containing attribution map values or sequence statistics with proper column headings for alphabet.
        See function 'arr2pd' in utilities.py for conversion of array to Pandas dataframe.
    logo_type : str {'attribution', 'sequence'}
        Data type of input dataframe
    axis : matplotlib.pyplot axis
        Axis object for plotting logo
    aesthetic_lvl : int {1, 2, 3}
        Increasing this value increases the aesthetic appearance of the logo, with the caveat that higher
        aesthetic levels have higher computation times.
    ref_seq : str or None
        The reference sequence (e.g., wild type) for the ensemble of mutagenized sequences.
    fontsize : int
        The font size for the axis ticks.
    figsize : [float, float]
        The figure size, if no axis is provided.
    y_min_max : [float, float]
        The global attribution minimum and maximum value over all attribution maps (not just the current cluster).
    center_values : boole

    Returns
    -------
    logo : pandas.dataframe
    axis : matplotlib.pyplot axis
    """
    map_length = len(logo_df)
    color_scheme = 'classic'
    if logo_type == 'sequence' and ref_seq is not None:
        color_scheme = 'dimgray'
    
    if axis is None:
        fig, axis = plt.subplots()
        if figsize is not None:
            fig.set_size_inches(figsize) # e.g., [20, 2.5] or [10, 1.5] for logos < 100 nt

    if aesthetic_lvl > 0:
        if aesthetic_lvl == 2: # pretty logomaker mode (very slow)
            logo = logomaker.Logo(df=logo_df,
                                ax=axis,
                                fade_below=.5,
                                shade_below=.5,
                                width=.9,
                                center_values=center_values,
                                font_name='Arial Rounded MT Bold', # comp time bottleneck
                                color_scheme=color_scheme)
        elif aesthetic_lvl == 1: # plain logomaker mode (faster)
            if logo_type == 'sequence' and ref_seq is not None:
                logo = logomaker.Logo(df=logo_df,
                                    ax=axis,
                                    width=.9,
                                    center_values=center_values,
                                    color_scheme=color_scheme)
            else:
                logo = logomaker.Logo(df=logo_df,
                                    ax=axis,
                                    fade_below=.5,
                                    shade_below=.5,
                                    width=.9,
                                    center_values=center_values,
                                    color_scheme=color_scheme)
        if logo_type == 'sequence' and ref_seq is not None:
            logo.style_glyphs_in_sequence(sequence=ref_seq, color='darkorange')
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left'], visible=True)
        logo.style_xticks(rotation=90, fmt='%d', anchor=0)
        logo.ax.xaxis.set_ticks_position('none')
        logo.ax.xaxis.set_tick_params(pad=-1)
        #logo.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        logo, axis = plot_logo_simple(np.array(logo_df), axis, color_scheme, ref_seq=ref_seq, center_values=center_values)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.tick_params(axis='x', rotation=90)#, anchor=0)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        #axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axis.tick_params(bottom=False)
        axis.axhline(zorder=100, color='black', linewidth=0.5)

    axis.tick_params(axis="x", labelsize=fontsize)
    axis.tick_params(axis="y", labelsize=fontsize)

    axis.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
    axis.yaxis.offsetText.set_fontsize(fontsize)

    if center_values is True:
        centered_df = center_mean(np.array(logo_df))
        ymin = min(np.sum(np.where(centered_df<0,centered_df,0), axis=1))
        ymax = max(np.sum(np.where(centered_df>0,centered_df,0), axis=1))
    else:
        ymin = min(np.sum(np.where(np.array(logo_df)<0,np.array(logo_df),0), axis=1))
        ymax = max(np.sum(np.where(np.array(logo_df)>0,np.array(logo_df),0), axis=1))
    axis.set_ylim([ymin, ymax])

    if y_min_max is not None:
        if np.abs(y_min_max[0]) > np.abs(y_min_max[1]):
            y_max_abs = np.abs(y_min_max[0])
        else:
            y_max_abs = np.abs(y_min_max[1])
        plt.ylim(-1.*y_max_abs, y_max_abs)
    
    if map_length <= 100:
        axis.set_xticks(np.arange(0, map_length, 1))
    if map_length > 100:
        axis.set_xticks(np.arange(0, map_length-1, 5))
    if map_length > 1000:
        axis.set_xticks(np.arange(0, map_length-1, 10))

    plt.xlim(-0.5, map_length+.5)
    plt.tight_layout()
    return (logo, axis)



def plot_clusters_matches_sns(df, column='Entropy', sort=True, threshold=50, dpi=200, save_dir=None):
    """Function for visualizing all cluster positional statistics, including marginal distrubutions.

    Parameters
    ----------
    df : pandas.DataFrame, shape=(num_clusters x num_positions, 3)
        DataFrame corresponding to all mismatches (or matches) to reference sequence (or cluster consensus sequence).
        DataFrame is output by clusterer.py as 'compare_clusters.csv'
    column : str
        If 'Entropy', figure labels describe the Shannon entropy of characters at each position per cluster
        If 'Reference', figure labels describe the number of mismatches to reference sequence
        (recommended for local mutagenesis libraries).
        If 'Consensus', figure labels describe the number of matches to each cluster's respective consensus sequence
        (recommended for global mutagenesis libraries).
    sort : bool
        If True, the ordering of clusters will be sorted using the similarity of their positional elements.
    threshold : int in [0,100]
        Threshold value for whether to include positional elements in summation displayed in marginal plots.
        If 'reference' is True, all percent mismatches above the threshold will be summed.
        If 'reference' is False, all percent matches above the threshold will be summed,
        not including values at 100% to avoid obscuring the display due to presence of the conserved sequence pattern.
    dpi : int
        Controls the DPI used for rendering the figure.
    save_dir : str
        Directory for saving figures to file.
    """

    try:
        df = pd.read_csv(df)
    except:
        pass

    revels = df.pivot(columns='Position', index='Cluster', values=column)

    if sort is True: # reorder dataframe based on dendrogram sorting: 
        row_linkage = hierarchy.linkage(distance.pdist(revels), method='average')
        dendrogram = hierarchy.dendrogram(row_linkage, no_plot=True, color_threshold=-np.inf)
        reordered_ind = dendrogram['leaves']
        revels = df.pivot(columns='Position', index='Cluster', values=column)
        revels = revels.reindex(reordered_ind)

    nC = df['Cluster'].max() + 1
    nP = df['Position'].max() + 1

    # from https://stackoverflow.com/questions/65917235/how-to-create-a-heatmap-with-marginal-histograms-similar-to-a-jointplot
    joint = sns.jointplot(data=df, x='Position', y='Cluster', kind='hist', bins=(nC, nP))
    joint.ax_marg_y.cla()
    joint.ax_marg_x.cla()

    palette = sns.color_palette('rocket', n_colors=100)
    if column == 'Entropy':
        label = 'Shannon entropy (bits)'
    elif column == 'Reference':
        label = 'Mismatches to reference sequence'
        palette.reverse()
    elif column == 'Consensus':
        label = 'Matches to per-cluster consensus'

    heat = sns.heatmap(data=revels, ax=joint.ax_joint, cbar=False, cmap=palette,
                    cbar_kws={'format': '%.0f%%',
                                'label': label,
                                'location': 'bottom', #left
                                'orientation': 'horizontal'},
                                )

    # filter out all mismatches/matches based on threshold value (%):
    if column == 'Reference':
        df[column] = (df[column] >= threshold).astype(int) # binary mask
    elif column == 'Consensus':
        if threshold == 100:
            threshold_low = 99.9
        else:
            threshold_low = threshold
        df[column] = ((df[column] > threshold_low) & (df[column] < 100)).astype(int) # binary mask

    marg_color = 'dimgray'
    joint.ax_marg_y.barh(np.arange(0.5, nC), df.groupby(['Cluster'])[column].sum().to_numpy(), color=marg_color)
    joint.ax_marg_x.bar(np.arange(0.5, nP), df.groupby(['Position'])[column].sum().to_numpy(), color=marg_color)
    joint.ax_marg_x.yaxis.tick_right()

    if nP > 100:
        x_skip = 10
    else:
        x_skip = 1
    xtick_labels = []
    xtick_range = np.arange(0.5, nP, x_skip)
    for i in xtick_range:
        if int(i)%x_skip == 0:
            xtick_labels.append(str(int(i-0.5)))
    joint.ax_joint.set_xticks(xtick_range)
    joint.ax_joint.set_xticklabels(xtick_labels, rotation=0)

    if nC > 10:
        y_skip = 6
    else:
        y_skip = 1
    ytick_labels = []
    ytick_range = np.arange(0.5, nC, y_skip)
    y_idx = 0
    for i in ytick_range:
        if int(i)%y_skip == 0:
            if sort is True:
                ytick_labels.append(str(reordered_ind[y_idx]))
            else:
                ytick_labels.append(str(int(i-0.5)))
        y_idx += y_skip
    joint.ax_joint.set_yticks(ytick_range)
    joint.ax_joint.set_yticklabels(ytick_labels, rotation=0)

    joint.ax_joint.tick_params(axis='both', which='major', labelsize=6)
    joint.ax_marg_x.tick_params(axis='both', which='major', labelsize=6)
    joint.ax_marg_y.tick_params(axis='both', which='major', labelsize=6)
    joint.ax_marg_x.yaxis.set_major_locator(MaxNLocator(integer=True))
    joint.ax_marg_y.xaxis.set_major_locator(MaxNLocator(integer=True))

    # remove ticks between heatmap and histograms:
    joint.ax_marg_x.tick_params(axis='x', bottom=False, labelbottom=False)
    joint.ax_marg_y.tick_params(axis='y', left=False, labelleft=False)
    if 0: # remove ticks showing the heights of the histograms
        joint.ax_marg_x.tick_params(axis='y', left=False, labelleft=False)
        joint.ax_marg_y.tick_params(axis='x', bottom=False, labelbottom=False)
        joint.ax_marg_x.set_frame_on(False)
        joint.ax_marg_y.set_frame_on(False)

    # plot colorbar externally (hack):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[30,1])
    mg0 = SeabornFig2Grid(joint, fig, gs[0])
    ax2 = plt.subplot(gs[1])
    ax2.get_figure().colorbar(mpl.cm.ScalarMappable(norm=plt.Normalize(0, 100), cmap=mpl.colors.ListedColormap(list(palette))),
                cax=ax2, orientation='vertical', label=label,
                format='%.0f%%')

    # plot marginal legend externally:
    if column == 'Reference':
        if threshold == 100:
            threshold_operator = '='
        else:
            threshold_operator = '>'
        marginal_patch = mpatches.Patch(color=marg_color, label=r'Marginals: mismatches %s %s%s' % (threshold_operator, threshold, '%'))
    elif column == 'Consensus':
        marginal_patch = mpatches.Patch(color=marg_color, label=r'Marginals: %s%s < matches < 100%s' % (threshold_low, '%', '%'))
    elif column == 'Entropy': #ZULU
        print('TBD')
    fig.legend(handles=[marginal_patch], frameon=False, prop={'size': 8}, loc='upper left')

    plt.rcParams['savefig.dpi'] = dpi
    plt.tight_layout()
    if save_dir is not None:
        if column == 'Reference':
            fig.savefig(os.path.join(save_dir, 'figure_clusters_mismatches.png'))
        elif column == 'Consensus':
            fig.savefig(os.path.join(save_dir, 'figure_clusters_matches.png'))
        elif column == 'Entropy':
            fig.savefig(os.path.join(save_dir, 'figure_clusters_entropy.png'))
    plt.show()


def plot_clusters_matches_2d_gui(df, figure, sort=None, column='Entropy', delta=False, mut_rate=10, alphabet=['A','C','G','T'], sort_index=None):
    """Function for visualizing all cluster positional statistics. Does not include marginal distrubutions.

    Parameters
    ----------
    df : pandas.DataFrame, shape=(num_clusters x num_positions, 3)
        DataFrame corresponding to all mismatches (or matches) to reference sequence (or cluster consensus sequence).
        DataFrame is output by clusterer.py as 'mismatches_reference.csv' or 'matches_consensus.csv'.
    figure : Matplotlib.Figure
        Instantiated Figure from SEAM GUI.
    sort : {None, 'visual', 'predefined'}
        If 'visual', the ordering of clusters will be sorted using the similarity of their sequence-derived patterns.
        If 'predefined', the ordering of clusters will be sorted using a predefined list (e.g., the median DNN score of each cluster)
    column : str
        If 'Entropy', figure labels describe the Shannon entropy of characters at each position per cluster.
        If 'Reference', figure labels describe the number of mismatches to reference sequence
        (recommended for local mutagenesis libraries).
        If 'Consensus', figure labels describe the number of matches to each cluster's respective consensus sequence
        (recommended for global mutagenesis libraries).
    delta : bool
        If True and column='Entropy', display matrix using the change in entropy from the background expectation.
    mut_rate : float
        Mutation rate (as a percentage) used to initially generate the mutagenized sequences. Only required if 'delta' is True.
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.
    """
    sys.setrecursionlimit(100000) # fix: https://stackoverflow.com/questions/57401033/how-to-fixrecursionerror-maximum-recursion-depth-exceeded-while-getting-the-st

    ax = figure.add_subplot(111)

    nC = df['Cluster'].max() + 1
    nP = df['Position'].max() + 1
    revels = df.pivot(columns='Position', index='Cluster', values=column)

    if 0: # optional: set cells in MSM to be perfectly square
        ax.set_aspect('equal')

    if sort == 'visual': # reorder dataframe based on dendrogram
        row_linkage = hierarchy.linkage(distance.pdist(revels), method='average')
        dendrogram = hierarchy.dendrogram(row_linkage, no_plot=True, color_threshold=-np.inf)
        reordered_ind = dendrogram['leaves']
        revels = df.pivot(columns='Position', index='Cluster', values=column) # ZULU redundant?
        revels = revels.reindex(reordered_ind)
    elif sort == 'predefined':
        reordered_ind = sort_index
        revels = df.pivot(columns='Position', index='Cluster', values=column) # ZULU redundant
        revels = revels.reindex(reordered_ind)
        if nC != df['Cluster'].nunique():
            ax.set_aspect('equal')
            nC = df['Cluster'].nunique()
    elif sort is None:
        reordered_ind = np.arange(nC)

    if column == 'Entropy':
        palette = sns.color_palette('rocket', n_colors=100)
        if delta is False:
            label = 'Shannon entropy (bits)'
            palette.reverse()
        else:
            label = '$\Delta H$ (bits)'
            palette = sns.color_palette('vlag', n_colors=100)
    elif column == 'Reference':
        if 0:
            palette = sns.color_palette('rocket', n_colors=100)
        else:
            #palette = sns.color_palette('viridis', n_colors=100)
            palette = sns.color_palette('Blues', n_colors=100)
            palette.reverse()
        label = 'Mismatches to reference sequence'
        palette.reverse()
    elif column == 'Consensus':
        palette = sns.color_palette('rocket', n_colors=100)
        label = 'Matches to per-cluster consensus'

    if column == 'Entropy' and delta is True:
        from matplotlib import colors
        from scipy.stats import entropy
        r = mut_rate / 100
        p = np.array([1-r, r/float(len(alphabet)-1), r/float(len(alphabet)-1), r/float(len(alphabet)-1)])
        entropy_bg = entropy(p, base=2) # calculate Shannon entropy
        revels -= entropy_bg

        if 0: # remove noise based on threshold for cleaner visualization
            thresh = .10
            revels = revels.mask((revels < thresh) & (revels > -1.*thresh))

        divnorm = colors.TwoSlopeNorm(vmin=revels.min(numeric_only=True).min(), vcenter=0., vmax=revels.max(numeric_only=True).max())
        heatmap = ax.pcolormesh(revels, cmap='seismic', norm=divnorm)
    else:
        heatmap = ax.pcolormesh(revels, cmap=mpl.colors.ListedColormap(list(palette)))

    ax.set_xlabel('Position', fontsize=6)
    ax.set_ylabel('Cluster', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=4)
    ax.set_title('Click on a cell to view its sequence variation', fontsize=5)
    ax.invert_yaxis()

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

    if nC > 10:
        y_skip = 10
    else:
        y_skip = 1
    ytick_labels = []
    ytick_range = np.arange(0.5, nC, y_skip)
    y_idx = 0
    for i in ytick_range:
        if int(i)%y_skip == 0:
            if sort is True:
                ytick_labels.append(str(reordered_ind[y_idx]))
            else:
                ytick_labels.append(str(int(i-0.5)))
        y_idx += y_skip
    ax.set_yticks(ytick_range)
    ax.set_yticklabels(ytick_labels, rotation=0, minor=False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if column == 'Reference' or column == 'Consensus':
        cbar = figure.colorbar(heatmap, cax=cax, orientation='vertical', format='%.0f%%')
        heatmap.set_clim(0, 100)
    elif column == 'Entropy':
        cbar = figure.colorbar(heatmap, cax=cax, orientation='vertical')#, format='%.0f')
        if delta is False:
            heatmap.set_clim(0, 2)
    cbar.ax.set_ylabel(label, rotation=270, fontsize=6, labelpad=9)
    cbar.ax.tick_params(labelsize=6)

    return (ax, cax, reordered_ind, revels)


class SeabornFig2Grid():
    # https://stackoverflow.com/questions/69833665/seaborn-subplot-of-jointplots-doesnt-work
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # from https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def center_mean(x): # equivalent to center_values in Logomaker
    mu = np.expand_dims(np.mean(x, axis=1), axis=1) #position dependent
    z = (x - mu)
    return z


def plot_logo_simple(logo_df, axis, color_scheme, ref_seq=None, center_values=True):
    # Adapted from:
        # https://github.com/kundajelab/deeplift/blob/16ef5dd05c3e05e9e5c7ec04d1d8a24fad046d96/deeplift/visualization/viz_sequence.py
        # https://github.com/kundajelab/chrombpnet/blob/a5c231fdf231bb29e9ca53d42a4c6e196f7546e8/chrombpnet/evaluation/figure_notebooks/subsampling/viz_sequence.py

    def ic_scale(pwm, background):
        odds_ratio = ((pwm+0.001)/(1.004))/(background[None,:])
        ic = ((np.log((pwm+0.001)/(1.004))/np.log(2))*pwm -\
                (np.log(background)*background/np.log(2))[None,:])
        return pwm*(np.sum(ic,axis=1)[:,None])

    def plot_a(ax, base, left_edge, height, color):
        a_polygon_coords = [
            np.array([
            [0.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.2, 0.0],
            ]),
            np.array([
            [1.0, 0.0],
            [0.5, 1.0],
            [0.5, 0.8],
            [0.8, 0.0],
            ]),
            np.array([
            [0.225, 0.45],
            [0.775, 0.45],
            [0.85, 0.3],
            [0.15, 0.3],
            ])
        ]
        for polygon_coords in a_polygon_coords:
            ax.add_patch(mpl.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                    + np.array([left_edge,base])[None,:]),
                                                    facecolor=color, edgecolor=color))

    def plot_c(ax, base, left_edge, height, color):
        ax.add_patch(mpl.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                                facecolor=color, edgecolor=color))
        ax.add_patch(mpl.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                                facecolor='white', edgecolor='white'))
        ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                                facecolor='white', edgecolor='white', fill=True))

    def plot_g(ax, base, left_edge, height, color):
        ax.add_patch(mpl.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                                facecolor=color, edgecolor=color))
        ax.add_patch(mpl.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                                facecolor='white', edgecolor='white'))
        ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                                facecolor='white', edgecolor='white', fill=True))
        ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                                facecolor=color, edgecolor=color, fill=True))
        ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                                facecolor=color, edgecolor=color, fill=True))

    def plot_t(ax, base, left_edge, height, color):
        ax.add_patch(mpl.patches.Rectangle(xy=[left_edge+0.4, base],
                    width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
        ax.add_patch(mpl.patches.Rectangle(xy=[left_edge, base+0.8*height],
                    width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

    default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
    default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
    def plot_weights_given_ax(ax, array,
                    height_padding_factor,
                    length_padding,
                    subticks_frequency,
                    ref_index,
                    highlight,
                    colors=default_colors,
                    plot_funcs=default_plot_funcs,
                    ylabel="",
                    ylim=None):
        if len(array.shape)==3:
            array = np.squeeze(array)
        assert len(array.shape)==2, array.shape
        if (array.shape[0]==4 and array.shape[1] != 4):
            array = array.transpose(1,0)
        assert array.shape[1]==4
        max_pos_height = 0.0
        min_neg_height = 0.0
        heights_at_positions = []
        depths_at_positions = []
        # sort from smallest to highest magnitude
        for i in range(array.shape[0]):
            acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
            positive_height_so_far = 0.0
            negative_height_so_far = 0.0
            for letter in acgt_vals:
                plot_func = plot_funcs[letter[0]]
                if ref_index is not None and letter[0] == ref_index[i]:
                    color = 'darkorange'
                else:
                    color=colors[letter[0]]
                if (letter[1] > 0):
                    height_so_far = positive_height_so_far
                    positive_height_so_far += letter[1]
                else:
                    height_so_far = negative_height_so_far
                    negative_height_so_far += letter[1]
                plot_func(ax=ax, base=height_so_far, left_edge=i-0.5, height=letter[1], color=color)
            max_pos_height = max(max_pos_height, positive_height_so_far)
            min_neg_height = min(min_neg_height, negative_height_so_far)
            heights_at_positions.append(positive_height_so_far)
            depths_at_positions.append(negative_height_so_far)

        # now highlight any desired positions; the key of the highlight dict should be the color
        for color in highlight:
            for start_pos, end_pos in highlight[color]:
                assert start_pos >= 0.0 and end_pos <= array.shape[0]
                min_depth = np.min(depths_at_positions[start_pos:end_pos])
                max_height = np.max(heights_at_positions[start_pos:end_pos])
                ax.add_patch(
                    mpl.patches.Rectangle(xy=[start_pos,min_depth],
                        width=end_pos-start_pos,
                        height=max_height-min_depth,
                        edgecolor=color, fill=False))

        ax.set_xlim(-length_padding, array.shape[0]-length_padding)
        ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))

        # use user-specified y-axis limits
        if ylim is not None:
            min_neg_height, max_pos_height = ylim
            assert min_neg_height <= 0
            assert max_pos_height >= 0

        height_padding = max(abs(min_neg_height)*(height_padding_factor),
                            abs(max_pos_height)*(height_padding_factor))
        ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
        ax.set_ylabel(ylabel)
        ax.yaxis.label.set_fontsize(15)

    if color_scheme == 'classic':
        colors = {0: [0, .5, 0], 1: [0, 0, 1], 2: [1, .65, 0], 3: [1, 0, 0]} # Logomaker 'classic' mode
    elif color_scheme == 'dimgray':
        colors = {0: 'dimgray', 1: 'dimgray', 2: 'dimgray', 3: 'dimgray'}
    plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}

    if ref_seq is not None:
        ref_index = []
        for c in ref_seq:
            if c == 'A':
                ref_index.append(0)
            elif c == 'C':
                ref_index.append(1)
            elif c == 'G':
                ref_index.append(2)
            elif c == 'T':
                ref_index.append(3)
    else:
        ref_index = None

    if center_values is True:
        logo_df = center_mean(logo_df)

    logo = plot_weights_given_ax(ax=axis, array=logo_df,
                                height_padding_factor=0.2,
                                length_padding=.5,#1.0,
                                subticks_frequency=100.0,
                                colors=colors, plot_funcs=plot_funcs,
                                ref_index=ref_index,
                                highlight={}, ylabel="")
    return (logo, axis)


def mismatch2csv(comparison_df, clusters, sequences, reference, threshold=100, sort_index=None, alphabet=['A','C','G','T'], verbose=1, save_dir=None):
    # input sequences need to be pre-sliced to match cropped attribution region (if applicable)

    nC = comparison_df['Cluster'].max() + 1
    nP = comparison_df['Position'].max() + 1

    if sort_index is not None:
        mapping_dict = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(sort_index)}
        comparison_df['Cluster'] = comparison_df['Cluster'].map(mapping_dict)
        clusters['Cluster'] = clusters['Cluster'].map(mapping_dict)

    comparison_df = comparison_df.drop(comparison_df[comparison_df['Reference'] < threshold].index)

    column_names = ['Position', 'Cluster', 'Reference', 'Pct_mismatch']
    for char in alphabet:
        column_names.append(char)
    mismatch_df = pd.DataFrame(columns=column_names)
    mismatch_df['Position'] = comparison_df['Position']
    mismatch_df['Cluster'] = comparison_df['Cluster']
    mismatch_df['Pct_mismatch'] = comparison_df['Reference']
    mismatch_df = mismatch_df.fillna(0)
    mismatch_df.reset_index(drop=True, inplace=True)   
     
    for i in range(len(mismatch_df)):
        row = int(mismatch_df['Cluster'].iloc[i])
        col = int(mismatch_df['Position'].iloc[i])
        mismatch_df.loc[i, 'Reference'] = reference[col]
        k_idxs = np.array(clusters['Cluster'].loc[clusters['Cluster'] == row].index)
        seqs = sequences['Sequence']
        seqs_cluster = seqs[k_idxs]
        occ = seqs_cluster.str.slice(col, col+1)
        vc = occ.value_counts()
        vc = vc.sort_index()
        chars = list(vc.index)
        vals = list(vc.values)
        for j in range(len(chars)):
            mismatch_df.loc[i, chars[j]] = vals[j]

    mismatch_df = mismatch_df.sort_values('Position')
    mismatch_df['Sum'] = mismatch_df[list(mismatch_df.columns[4:])].sum(axis=1)

    if save_dir is not None:
        mismatch_df.to_csv(os.path.join(save_dir, 'mismatches_%spct.csv' % (int(threshold))), index=False)
    else:
        if verbose == 1:
            print(mismatch_df.to_string())

    # print some statistics for better understanding the distribution:
    num_hits = len(mismatch_df)
    pos_unique = np.unique(mismatch_df['Position'].values)
    pos_intersect = np.intersect1d(pos_unique, np.arange(0,nP))
    cluster_unique = np.unique(mismatch_df['Cluster'].values)
    cluster_intersect = np.intersect1d(cluster_unique, np.arange(0,nC))
    if verbose == 1:
        print('\tTotal number of hits (i.e., %s%s mismatches): %s of %s (%.2f%s)' % (threshold, '%', num_hits, nP*nC, 100*(num_hits/(nP*nC)), '%'))
        print('\tPercent of positions with at least one hit: %.2f%s' % (100*(len(pos_intersect)/nP),'%'))
        print('\tPercent of clusters with at least one hit: %.2f%s' % (100*(len(cluster_intersect)/nC),'%'))
        print('\tAverage occupancy of target clusters:', mismatch_df['Sum'].mean())

    return mismatch_df



if __name__ == "__main__":
    py_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(py_dir)

    #name = 'PIK3R3'#'CD40'#'NXF1'
    #mode = 'quantity'
    #mode = 'profile'

    #dir_name = 'examples/examples_clipnet/outputs_local_%s/heterozygous_pt01/%s_nfolds9' % (name, mode)
    dir_name = 'examples/examples_chrombpnet/outputs_local_PPIF_promoter_counts/mut_pt10/seed2_N100k_allFolds'
    #clusters_dir = '/clusters_umap_kmeans200'
    #clusters_dir = '/clusters_umap_dbscan'
    clusters_dir = '/clusters_hierarchical_maxclust200_sortMedian'
    #clusters_dir = '/clusters_hierarchical_cut0.0001'
    save_dir = os.path.join(parent_dir, dir_name + clusters_dir)
    comparison_df = pd.read_csv(os.path.join(parent_dir, dir_name + clusters_dir + '/compare_clusters.csv'))
    clusters = pd.read_csv(os.path.join(parent_dir, dir_name + clusters_dir + '/all_clusters.csv'))
    mave = pd.read_csv(os.path.join(parent_dir, dir_name + '/mave.csv'))
    reference = mave['Sequence'][0]
    mismatch_df = mismatch2csv(comparison_df, clusters, mave, reference, 
                               threshold=90,#100,
                               alphabet=['A','C','G','T','M','R','W','S','Y','K'], save_dir=save_dir)