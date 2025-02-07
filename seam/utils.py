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