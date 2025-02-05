"""
Utility functions for SEAM-NN package.
Core functionality for sequence processing, data handling, and computation.
"""
import sys, os
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Tuple
from scipy.stats import entropy
from scipy.spatial import distance
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.patches as patches

# Warning Management
def suppress_warnings() -> None:
    """Suppress common warnings for cleaner output."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# Device Management
def get_device(gpu: bool = False) -> Optional[str]:
    """Get appropriate compute device."""
    if not gpu:
        return None
    try:
        import tensorflow as tf
        return '/GPU:0' if tf.test.is_built_with_cuda() else '/CPU:0'
    except ImportError:
        return None

# Sequence Processing
def arr2pd(x: np.ndarray, alphabet: List[str] = ['A','C','G','T']) -> pd.DataFrame:
    """Convert array to pandas DataFrame with proper column headings."""
    labels = {i: x[:,idx] for idx, i in enumerate(alphabet)}
    return pd.DataFrame.from_dict(labels, orient='index').T

def oh2seq(one_hot: np.ndarray, 
           alphabet: List[str] = ['A','C','G','T'], 
           encoding: int = 1) -> str:
    """Convert one-hot encoding to sequence."""
    if encoding == 1:
        seq = []
        for i in range(np.shape(one_hot)[0]):
            for j in range(len(alphabet)):
                if one_hot[i][j] == 1:
                    seq.append(alphabet[j])
        return ''.join(seq)
    
    elif encoding == 2:
        encoding_map = {
            tuple(np.array([2,0,0,0])): 'A',
            tuple(np.array([0,2,0,0])): 'C',
            tuple(np.array([0,0,2,0])): 'G',
            tuple(np.array([0,0,0,2])): 'T',
            tuple(np.array([0,0,0,0])): 'N',
            tuple(np.array([1,1,0,0])): 'M',
            tuple(np.array([1,0,1,0])): 'R',
            tuple(np.array([1,0,0,1])): 'W',
            tuple(np.array([0,1,1,0])): 'S',
            tuple(np.array([0,1,0,1])): 'Y',
            tuple(np.array([0,0,1,1])): 'K',
        }
        return ''.join(encoding_map.get(tuple(row), 'N') for row in one_hot)

def seq2oh(seq: str, 
           alphabet: List[str] = ['A','C','G','T'], 
           encoding: int = 1) -> np.ndarray:
    """Convert sequence to one-hot encoding."""
    if encoding == 1:
        L = len(seq)
        one_hot = np.zeros(shape=(L,len(alphabet)), dtype=np.float32)
        for idx, i in enumerate(seq):
            for jdx, j in enumerate(alphabet):
                if i == j:
                    one_hot[idx,jdx] = 1
        return one_hot
    
    elif encoding == 2:
        encoding_map = {
            "A": np.array([2,0,0,0]),
            "C": np.array([0,2,0,0]),
            "G": np.array([0,0,2,0]),
            "T": np.array([0,0,0,2]),
            "N": np.array([0,0,0,0]),
            "M": np.array([1,1,0,0]),
            "R": np.array([1,0,1,0]),
            "W": np.array([1,0,0,1]),
            "S": np.array([0,1,1,0]),
            "Y": np.array([0,1,0,1]),
            "K": np.array([0,0,1,1]),
        }
        return np.array([encoding_map.get(s.upper(), encoding_map["N"]) 
                        for s in seq])

# Helper Functions
def calculate_background_entropy(mut_rate: float, alphabet_size: int) -> float:
    """Calculate background entropy given mutation rate."""
    p = np.array([1-mut_rate] + [mut_rate/(alphabet_size-1)] * (alphabet_size-1))
    return entropy(p, base=2)

# File Management
def safe_file_path(directory: str, filename: str, extension: str) -> str:
    """Generate safe file path, creating directories if needed."""
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{filename}.{extension}")

def _get_45deg_mesh(mat):
    """Create X and Y grids rotated -45 degrees."""
    theta = -np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    K = len(mat) + 1
    grid1d = np.arange(0, K) - .5
    X = np.tile(np.reshape(grid1d, [K, 1]), [1, K])
    Y = np.tile(np.reshape(grid1d, [1, K]), [K, 1])
    xy = np.array([X.ravel(), Y.ravel()])

    xy_rot = R @ xy
    X_rot = xy_rot[0, :].reshape(K, K)
    Y_rot = xy_rot[1, :].reshape(K, K).T

    return X_rot, Y_rot

def plot_pairwise_matrix(theta_lclc, view_window=None, alphabet=['A','C','G','T'], 
                        threshold=None, save_dir=None, cbar_title='Pairwise', 
                        gridlines=True):
    """Plot pairwise matrix visualization.
    Adapted from https://github.com/jbkinney/mavenn/blob/master/mavenn/src/visualization.py
    Original authors: Tareen, A. and Kinney, J.
    """
    if threshold is not None:
        temp = theta_lclc.flatten()
        temp[(temp >= -1.*threshold) & (temp <= threshold)] = 0
        theta_lclc = temp.reshape(theta_lclc.shape)

    # Set up gridlines
    if gridlines:
        show_seplines = True
        sepline_kwargs = {'linestyle': '-',
                         'linewidth': .3,
                         'color':'lightgray'}
    else:
        show_seplines = False
        sepline_kwargs = {'linestyle': '-',
                         'linewidth': .5,
                         'color':'gray'}

    # Create figure
    fig, ax = plt.subplots(figsize=[10,5])

    # Get matrix dimensions
    L = theta_lclc.shape[0]
    C = theta_lclc.shape[1]
    
    # Create position grids
    ls = np.arange(L)
    cs = np.arange(C)
    l1_grid = np.tile(np.reshape(ls, (L, 1, 1, 1)), (1, C, L, C))
    c1_grid = np.tile(np.reshape(cs, (1, C, 1, 1)), (L, 1, L, C))
    l2_grid = np.tile(np.reshape(ls, (1, 1, L, 1)), (L, C, 1, C))

    # Set up pairwise matrix
    nan_ix = ~(l2_grid - l1_grid >= 1)
    values = theta_lclc.copy()
    values[nan_ix] = np.nan

    # Reshape into matrix
    mat = values.reshape((L*C, L*C))
    mat = mat[:-C, :]
    mat = mat[:, C:]
    K = (L - 1) * C

    # Get finite elements
    ix = np.isfinite(mat)
    
    # Set color limits
    clim = [np.min(mat[ix]), np.max(mat[ix])]
    ccenter = 0
    
    # Set up normalization
    if ccenter is not None:
        if (clim[0] > ccenter) or (clim[1] < ccenter):
            ccenter = 0.5 * (clim[0] + clim[1])
        norm = TwoSlopeNorm(vmin=clim[0], vcenter=ccenter, vmax=clim[1])
    else:
        norm = Normalize(vmin=clim[0], vmax=clim[1])

    # Get rotated mesh
    X_rot, Y_rot = _get_45deg_mesh(mat)
    
    # Normalize coordinates
    half_pixel_diag = 1 / (2*C)
    pixel_side = 1 / (C * np.sqrt(2))
    X_rot = X_rot * pixel_side + half_pixel_diag
    Y_rot = Y_rot * pixel_side
    Y_rot = -Y_rot

    # Set up plot limits
    xlim_pad = 0.1
    ylim_pad = 0.1
    xlim = [-xlim_pad, L - 1 + xlim_pad]
    ylim = [-0.5 - ylim_pad, (L - 1) / 2 + ylim_pad]

    # Create heatmap
    im = ax.pcolormesh(X_rot, Y_rot, mat, cmap='seismic', norm=norm)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add gridlines if requested
    if show_seplines:
        ysep_min = -0.5 - .001 * half_pixel_diag
        ysep_max = L / 2 + .001 * half_pixel_diag
        
        for n in range(0, K+1, C):
            x = X_rot[n, :]
            y = Y_rot[n, :]
            ks = (y >= ysep_min) & (y <= ysep_max)
            ax.plot(x[ks], y[ks], **sepline_kwargs)

            x = X_rot[:, n]
            y = Y_rot[:, n]
            ks = (y >= ysep_min) & (y <= ysep_max)
            ax.plot(x[ks], y[ks], **sepline_kwargs)

    # Add triangle boundary
    boundary_kwargs = {'linestyle': '-', 'linewidth': .7, 'color':'k'}
    
    # Top edge
    top_x = X_rot[0, :]
    top_y = Y_rot[0, :]
    ax.plot(top_x, top_y, **boundary_kwargs)
    
    # Right edge
    right_x = [X_rot[0, -1], X_rot[-1, -1]]
    right_y = [Y_rot[0, -1], Y_rot[-1, -1]]
    ax.plot(right_x, right_y, **boundary_kwargs)
    
    # Bottom edge
    bottom_x = []
    bottom_y = []
    for i in range(len(X_rot) - 1):
        bottom_x.extend([X_rot[i + 1, i], X_rot[i + 1, i + 1]])
        bottom_y.extend([Y_rot[i + 1, i], Y_rot[i + 1, i + 1]])
    ax.plot(bottom_x, bottom_y, **boundary_kwargs)
    
    # Left edge completion
    last_x = [top_x[0], bottom_x[0]]
    last_y = [top_y[0], bottom_y[0]]
    ax.plot(last_x, last_y, **boundary_kwargs)

    # Set plot properties
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_xticks(np.arange(L).astype(int))
    ax.set_xlabel('Nucleotide position')

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(cbar_title, labelpad=8, rotation=-90)
    cb.outline.set_visible(False)
    cb.ax.tick_params(direction='in', size=20, color='white')

    # Set symmetric colorbar limits
    theta_max = max(abs(np.min(theta_lclc)), abs(np.max(theta_lclc)))
    cb.mappable.set_clim(vmin=-theta_max, vmax=theta_max)

    plt.tight_layout()
    
    if save_dir is not None:
        plt.savefig(f"{save_dir}/{cbar_title.lower()}_matrix.pdf", 
                   facecolor='w', dpi=600)
        plt.close()
        
    return fig