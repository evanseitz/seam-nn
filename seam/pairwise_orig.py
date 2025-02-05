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