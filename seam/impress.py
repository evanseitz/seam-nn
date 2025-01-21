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


def plot_pairwise_matrix(theta_lclc, view_window=None, alphabet=['A','C','G','T'], threshold=None, save_dir=None, cbar_title='Pairwise', gridlines=True):    
    """Function for visualizing pairwise matrix.

    Parameters
    ----------
    theta_lclc : numpy.ndarray
        Pairwise matrix parameters (shape: (L,C,L,C)).
    view_window : [int, int]
        Index of start and stop position along sequence to probe;
        i.e., [start, stop], where start < stop and both entries
        satisfy 0 <= int <= L.
    alphabet : list
        The alphabet used to determine the C characters in the logo such that
        each entry is a string; e.g., ['A','C','G','T'] for DNA.
    threshold : float
        Define threshold window centered around zero for removing potential noise
        from parameters for cleaner pairwise matrix visualization
    save_dir : str
        Directory for saving figures to file.

    Returns
    -------
    matplotlib.pyplot.Figure
    """
    if threshold is not None:
        temp = theta_lclc.flatten()
        temp[(temp >= -1.*threshold) & (temp <= threshold)] = 0
        theta_lclc = temp.reshape(theta_lclc.shape)

    if gridlines is True:
        show_seplines = True
        #sepline_kwargs = {'linestyle': '-',
        #                  'linewidth': .5,
        #                  'color':'gray'}
        sepline_kwargs = {'linestyle': '-',
                          'linewidth': .3,
                          'color':'lightgray'}
    else:
        show_seplines = False
        sepline_kwargs = {'linestyle': '-',
                          'linewidth': .5,
                          'color':'gray'}

    # plot maveen pairwise matrix
    fig, ax = plt.subplots(figsize=[10,5])
    ax, cb = heatmap_pairwise(values=theta_lclc,
                              alphabet=alphabet,
                              ax=ax,
                              gpmap_type='pairwise',
                              cmap_size='2%',
                              show_alphabet=False,
                              cmap='seismic',
                              cmap_pad=.1,
                              show_seplines=show_seplines,
                              sepline_kwargs = sepline_kwargs,
                              )           

    if view_window is not None:
        ax.xaxis.set_ticks(np.arange(0, view_window[1]-view_window[0], 2))
        ax.set_xticklabels(np.arange(view_window[0], view_window[1], 2))  
    cb.set_label(r'%s' % cbar_title,
                  labelpad=8, ha='center', va='center', rotation=-90)
    cb.outline.set_visible(False)
    cb.ax.tick_params(direction='in', size=20, color='white')
    ax.set_xlabel('Nucleotide position')

    if 1: # set up isometric colorbar
        theta_max = [abs(np.amin(theta_lclc)), abs(np.amax(theta_lclc))]
        #plt.cm.ScalarMappable.set_clim(cb, vmin=-1.*np.amax(theta_max), vmax=np.amax(theta_max))
        cb.mappable.set_clim(vmin=-1. * np.amax(theta_max), vmax=np.amax(theta_max))

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, '%s_matrix.pdf' % cbar_title.lower()), facecolor='w', dpi=600)
        plt.close()
    #else:
        #plt.show()
    return fig


def _get_45deg_mesh(mat):
    """Create X and Y grids rotated -45 degreees.
    Adapted from https://github.com/jbkinney/mavenn/blob/master/mavenn/src/visualization.py
    Original authors: Tareen, A., Kinney, J.
    """
    # Define rotation matrix
    theta = -np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Define unrotated coordinates on
    K = len(mat) + 1
    grid1d = np.arange(0, K) - .5
    X = np.tile(np.reshape(grid1d, [K, 1]), [1, K])
    Y = np.tile(np.reshape(grid1d, [1, K]), [K, 1])
    xy = np.array([X.ravel(), Y.ravel()])

    # Rotate coordinates
    xy_rot = R @ xy
    X_rot = xy_rot[0, :].reshape(K, K)
    Y_rot = xy_rot[1, :].reshape(K, K).T

    return X_rot, Y_rot


def heatmap_pairwise(values,
                     alphabet,
                     seq=None,
                     seq_kwargs=None,
                     ax=None,
                     gpmap_type="pairwise",
                     show_position=False,
                     position_size=None,
                     position_pad=1,
                     show_alphabet=True,
                     alphabet_size=None,
                     alphabet_pad=1,
                     show_seplines=True,
                     sepline_kwargs=None,
                     xlim_pad=.1,
                     ylim_pad=.1,
                     cbar=True,
                     cax=None,
                     clim=None,
                     clim_quantile=1,
                     ccenter=0,
                     cmap='coolwarm',
                     cmap_size="5%",
                     cmap_pad=0.1):
    """
    Adapted from https://github.com/jbkinney/mavenn/blob/master/mavenn/src/visualization.py
    Original authors: Tareen, A., Kinney, J.

    Draw a heatmap illustrating pairwise or neighbor values, e.g. representing
    model parameters, mutational effects, etc.

    Note: The resulting plot has aspect ratio of 1 and is scaled so that pixels
    have half-diagonal lengths given by ``half_pixel_diag = 1/(C*2)``, and
    blocks of characters have half-diagonal lengths given by
    ``half_block_diag = 1/2``. This is done so that the horizontal distance
    between positions (as indicated by x-ticks) is 1.

    Parameters
    ----------
    values: (np.array)
        An array, shape ``(L,C,L,C)``, containing pairwise or neighbor values.
        Note that only values at coordinates ``[l1, c1, l2, c2]`` with
        ``l2`` > ``l1`` will be plotted. NaN values will not be plotted.

    alphabet: (str, np.ndarray)
        Alphabet name ``'dna'``, ``'rna'``, or ``'protein'``, or 1D array
        containing characters in the alphabet.

    seq: (str, None)
        The sequence to show, if any, using dots plotted on top of the heatmap.
        Must have length ``L`` and be comprised of characters in ``alphabet``.

    seq_kwargs: (dict)
        Arguments to pass to ``Axes.scatter()`` when drawing dots to illustrate
        the characters in ``seq``.

    ax: (matplotlib.axes.Axes)
        The ``Axes`` object on which the heatmap will be drawn.
        If ``None``, one will be created. If specified, ``cbar=True``,
        and ``cax=None``, ``ax`` will be split in two to make room for a
        colorbar.

    gpmap_type: (str)
        Determines how many pairwise parameters are plotted.
        Must be ``'pairwise'`` or ``'neighbor'``. If ``'pairwise'``, a
        triangular heatmap will be plotted. If ``'neighbor'``, a heatmap
        resembling a string of diamonds will be plotted.

    show_position: (bool)
        Whether to annotate the heatmap with position labels.

    position_size: (float)
        Font size to use for position labels. Must be >= 0.

    position_pad: (float)
        Additional padding, in units of ``half_pixel_diag``, used to space
        the position labels further from the heatmap.

    show_alphabet: (bool)
        Whether to annotate the heatmap with character labels.

    alphabet_size: (float)
        Font size to use for alphabet. Must be >= 0.

    alphabet_pad: (float)
        Additional padding, in units of ``half_pixel_diag``, used to space
        the alphabet labels from the heatmap.

    show_seplines: (bool)
        Whether to draw lines separating character blocks for different
        position pairs.

    sepline_kwargs: (dict)
        Keywords to pass to ``Axes.plot()`` when drawing seplines.

    xlim_pad: (float)
        Additional padding to add (in absolute units) both left and right of
        the heatmap.

    ylim_pad: (float)
        Additional padding to add (in absolute units) both above and below the
        heatmap.

    cbar: (bool)
        Whether to draw a colorbar next to the heatmap.

    cax: (matplotlib.axes.Axes, None)
        The ``Axes`` object on which the colorbar will be drawn, if requested.
        If ``None``, one will be created by splitting ``ax`` in two according
        to ``cmap_size`` and ``cmap_pad``.

    clim: (list, None)
        List of the form ``[cmin, cmax]``, specifying the maximum ``cmax``
        and minimum ``cmin`` values spanned by the colormap. Overrides
        ``clim_quantile``.

    clim_quantile: (float)
        Must be a float in the range [0,1]. ``clim`` will be automatically
        chosen to include this central quantile of values.

    ccenter: (float)
        Value at which to position the center of a diverging
        colormap. Setting ``ccenter=0`` often makes sense.

    cmap: (str, matplotlib.colors.Colormap)
        Colormap to use.

    cmap_size: (str)
        Fraction of ``ax`` width to be used for the colorbar. For formatting
        requirements, see the documentation for
        ``mpl_toolkits.axes_grid1.make_axes_locatable()``.

    cmap_pad: (float)
        Space between colorbar and the shrunken heatmap ``Axes``. For formatting
        requirements, see the documentation for
        ``mpl_toolkits.axes_grid1.make_axes_locatable()``.

    Returns
    -------
    ax: (matplotlib.axes.Axes)
        ``Axes`` object containing the heatmap.

    cb: (matplotlib.colorbar.Colorbar, None)
        Colorbar object linked to ``ax``, or ``None`` if no colorbar was drawn.
    """

    L, C, L2, C2 = values.shape

    values = values.copy()

    ls = np.arange(L).astype(int)
    l1_grid = np.tile(np.reshape(ls, (L, 1, 1, 1)),
                      (1, C, L, C))
    l2_grid = np.tile(np.reshape(ls, (1, 1, L, 1)),
                      (L, C, 1, C))

    # If user specifies gpmap_type="neighbor", remove non-neighbor entries
    if gpmap_type == "neighbor":
        nan_ix = ~(l2_grid - l1_grid == 1)

    elif gpmap_type == "pairwise":
        nan_ix = ~(l2_grid - l1_grid >= 1)

    # Set values at invalid positions to nan
    values[nan_ix] = np.nan

    # Reshape values into a matrix
    mat = values.reshape((L*C, L*C))
    mat = mat[:-C, :]
    mat = mat[:, C:]
    K = (L - 1) * C

    # Verify that mat is the right size
    assert mat.shape == (K, K), \
        f'mat.shape={mat.shape}; expected{(K,K)}. Should never happen.'

    # Get indices of finite elements of mat
    ix = np.isfinite(mat)

    # Set color lims to central 95% quantile
    if clim is None:
        clim = np.quantile(mat[ix], q=[(1 - clim_quantile) / 2,
                                    1 - (1 - clim_quantile) / 2])

    # Create axis if none already exists
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Needed to center colormap at zero
    if ccenter is not None:

        # Reset ccenter if is not compatible with clim
        if (clim[0] > ccenter) or (clim[1] < ccenter):
            ccenter = 0.5 * (clim[0] + clim[1])

        norm = TwoSlopeNorm(vmin=clim[0], vcenter=ccenter, vmax=clim[1])

    else:
        norm = Normalize(vmin=clim[0], vmax=clim[1])

    # Get rotated mesh
    X_rot, Y_rot = _get_45deg_mesh(mat)

    # Normalize
    half_pixel_diag = 1 / (2*C)
    pixel_side = 1 / (C * np.sqrt(2))
    X_rot = X_rot * pixel_side + half_pixel_diag
    Y_rot = Y_rot * pixel_side


    # Set parameters that depend on gpmap_type
    ysep_min = -0.5 - .001 * half_pixel_diag
    xlim = [-xlim_pad, L - 1 + xlim_pad]
    if gpmap_type == "pairwise":
        ysep_max = L / 2 + .001 * half_pixel_diag
        ylim = [-0.5 - ylim_pad, (L - 1) / 2 + ylim_pad]
    else:
        ysep_max = 0.5 + .001 * half_pixel_diag
        ylim = [-0.5 - ylim_pad, 0.5 + ylim_pad]

    # Not sure why I have to do this
    Y_rot = -Y_rot

    # Draw rotated heatmap
    im = ax.pcolormesh(X_rot,
                       Y_rot,
                       mat,
                       cmap=cmap,
                       norm=norm)

    # Remove spines
    for loc, spine in ax.spines.items():
        spine.set_visible(False)

    # Set sepline kwargs
    if show_seplines:
        if sepline_kwargs is None:
            sepline_kwargs = {'color': 'gray',
                              'linestyle': '-',
                              'linewidth': .5}

        # Draw white lines to separate position pairs
        for n in range(0, K+1, C):

            # TODO: Change extent so these are the right length
            x = X_rot[n, :]
            y = Y_rot[n, :]
            ks = (y >= ysep_min) & (y <= ysep_max)
            ax.plot(x[ks], y[ks], **sepline_kwargs)

            x = X_rot[:, n]
            y = Y_rot[:, n]
            ks = (y >= ysep_min) & (y <= ysep_max)
            ax.plot(x[ks], y[ks], **sepline_kwargs)

    if 1: # Plot an outline around the triangular boundary
        boundary_kwargs = {'linestyle': '-',
                          'linewidth': .7,
                          'color':'k'}

        # Manually draw the left edge of the triangle
        top_x = X_rot[0, :]
        top_y = Y_rot[0, :]
        ax.plot(top_x, top_y, **boundary_kwargs)

        # Manually draw the rigth edge of the triangle
        right_x = [X_rot[0, -1], X_rot[-1, -1]]
        right_y = [Y_rot[0, -1], Y_rot[-1, -1]]
        ax.plot(right_x, right_y, **boundary_kwargs)

        # Draw the zigzag bottom edge (manually tracing the bottom row cells)
        bottom_x = []
        bottom_y = []
        for i in range(len(X_rot) - 1):
            # Append bottom-left corner of the current cell
            bottom_x.append(X_rot[i + 1, i])
            bottom_y.append(Y_rot[i + 1, i])
            # Append bottom-right corner of the current cell
            bottom_x.append(X_rot[i + 1, i + 1])
            bottom_y.append(Y_rot[i + 1, i + 1])
        ax.plot(bottom_x, bottom_y, **boundary_kwargs)

        # Fill in remaining segment
        last_x = [top_x[0], bottom_x[0]]
        last_y = [top_y[0], bottom_y[0]]
        ax.plot(last_x, last_y, **boundary_kwargs)


    # Set lims
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set aspect
    ax.set_aspect("equal")

    # Remove yticks
    ax.set_yticks([])

    # Set xticks
    xticks = np.arange(L).astype(int)
    ax.set_xticks(xticks)

    # If drawing characters
    if show_alphabet:

        # Draw c1 alphabet
        for i, c in enumerate(alphabet):
            x1 = 0.5 * half_pixel_diag \
                 + i * half_pixel_diag \
                 - alphabet_pad * half_pixel_diag
            y1 = - 0.5 * half_pixel_diag \
                 - i * half_pixel_diag \
                 - alphabet_pad * half_pixel_diag
            ax.text(x1, y1, c, va='center',
                    ha='center', rotation=-45, fontsize=alphabet_size)

        # Draw c2 alphabet
        for i, c in enumerate(alphabet):
            x2 = 0.5 + 0.5 * half_pixel_diag \
                 + i * half_pixel_diag \
                 + alphabet_pad * half_pixel_diag
            y2 = - 0.5 + 0.5 * half_pixel_diag \
                 + i * half_pixel_diag \
                 - alphabet_pad * half_pixel_diag
            ax.text(x2, y2, c, va='center',
                    ha='center', rotation=45, fontsize=alphabet_size)

    # Display positions if requested (only if model is pairwise)
    l1_positions = np.arange(0, L-1)
    l2_positions = np.arange(1, L)
    half_block_diag = C * half_pixel_diag
    if show_position and gpmap_type == "pairwise":

        # Draw l2 positions
        for i, l2 in enumerate(l2_positions):
            x2 = 0.5 * half_block_diag \
                 + i * half_block_diag \
                 - position_pad * half_pixel_diag
            y2 = 0.5 * half_block_diag \
                 + i * half_block_diag \
                 + position_pad * half_pixel_diag
            ax.text(x2, y2, f'{l2:d}', va='center',
                    ha='center', rotation=45, fontsize=position_size)

        # Draw l1 positions
        for i, l1 in enumerate(l1_positions):
            x1 = (L - 0.5) * half_block_diag \
                 + i * half_block_diag \
                 + position_pad * half_pixel_diag
            y1 = (L - 1.5) * half_block_diag \
                 - i * half_block_diag \
                 + position_pad * half_pixel_diag
            ax.text(x1, y1, f'{l1:d}', va='center',
                    ha='center', rotation=-45, fontsize=position_size)

    elif show_position and gpmap_type == "neighbor":

        # Draw l2 positions
        for i, l2 in enumerate(l2_positions):
            x2 = 0.5 * half_block_diag \
                 + 2 * i * half_block_diag \
                 - position_pad * half_pixel_diag
            y2 = 0.5 * half_block_diag \
                 + position_pad * half_pixel_diag
            ax.text(x2, y2, f'{l2:d}', va='center',
                    ha='center', rotation=45, fontsize=position_size)

        # Draw l1 positions
        for i, l1 in enumerate(l1_positions):
            x1 = 1.5 * half_block_diag \
                 + 2* i * half_block_diag \
                 + position_pad * half_pixel_diag
            y1 = + 0.5 * half_block_diag \
                 + position_pad * half_pixel_diag
            ax.text(x1, y1, f'{l1:d}', va='center',
                    ha='center', rotation=-45, fontsize=position_size)

    # Mark wt sequence
    if seq:
        # Set seq_kwargs if not set in constructor
        if seq_kwargs is None:
            seq_kwargs = {'marker': '.', 'color': 'k', 's': 2}

        # Iterate over pairs of positions
        for l1 in range(L):
            for l2 in range(l1+1, L):

                # Break out of loop if gmap_type is "neighbor" and l2 > l1+1
                if (l2-l1 > 1) and gpmap_type == "neighbor":
                    continue

                # Iterate over pairs of characters
                for i1, c1 in enumerate(alphabet):
                    for i2, c2 in enumerate(alphabet):

                        # If there is a match to the wt sequence,
                        if seq[l1] == c1 and seq[l2] == c2:

                            # Compute coordinates of point
                            x = half_pixel_diag + \
                                (i1 + i2) * half_pixel_diag + \
                                (l1 + l2 - 1) * half_block_diag
                            y = (i2 - i1) * half_pixel_diag + \
                                (l2 - l1 - 1) * half_block_diag

                            # Plot point
                            ax.scatter(x, y, **seq_kwargs)


    # Create colorbar if requested, make one
    if cbar:
        if cax is None:
            cax = make_axes_locatable(ax).new_horizontal(size=cmap_size,
                                                         pad=cmap_pad)
            fig.add_axes(cax)
        cb = plt.colorbar(im, cax=cax)

        # Otherwise, return None for cb
    else:
        cb = None

    return ax, cb



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