import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance

class Compiler:
    """
    Compiler: A utility for compiling sequence analysis data into a standardized format
    
    This implementation processes sequence data and associated metrics into a 
    pandas DataFrame containing:
    - DNN predictions
    - Hamming distances (if reference sequence provided in x_ref)
    - Global Importance Analysis (GIA) scores (if background predictions provided in y_bg)
    - Sequence strings
    
    Requirements:
    - numpy
    - pandas
    - scipy
    - squid (for one-hot to sequence conversion)
    
    Example usage:
        # Initialize compiler
        compiler = Compiler(
            x=sequences,          # One-hot sequences (N, L, A)
            y=predictions,        # DNN predictions (N, 1)
            x_ref=ref_sequence,   # Optional reference sequence (1, L, A)
            y_bg=bg_predictions,  # Optional background predictions (N, 1)
            alphabet=['A','C','G','T']
        )
        
        # Compile data
        mave = compiler.compile()
        
        # Save to CSV
        mave.to_csv('mave.csv', index=False)
    """
    
    def __init__(self, x, y, x_ref=None, y_bg=None, alphabet=None, gpu=False):
        """Initialize the Compiler.
        
        Args:
            x: One-hot sequences of shape (N, L, A)
            y: DNN predictions of shape (N, 1)
            x_ref: Optional reference sequence of shape (1, L, A)
            y_bg: Optional background predictions of shape (N, 1)
            alphabet: List of characters for sequence conversion (e.g., ['A','C','G','T'])
            gpu: Whether to use GPU-accelerated sequence conversion (default: False)
        """
        self.x = x
        self.y = y
        self.x_ref = x_ref
        self.y_bg = y_bg
        self.alphabet = alphabet or ['A','C','G','T']
        self.gpu = gpu
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input shapes and types."""
        if len(self.x.shape) != 3:
            raise ValueError("x must be 3D array of shape (N, L, A)")
        if len(self.y.shape) != 2:
            self.y = self.y.reshape(-1, 1)
        if self.x_ref is not None and len(self.x_ref.shape) != 3:
            raise ValueError("x_ref must be 3D array of shape (1, L, A)")
        if self.y_bg is not None and len(self.y_bg.shape) != 2:
            self.y_bg = self.y_bg.reshape(-1, 1)
        
        # Check that only one analysis type is requested
        if self.x_ref is not None and self.y_bg is not None:
            raise ValueError("Cannot provide both x_ref and y_bg. Choose either Hamming distance analysis (x_ref) or GIA analysis (y_bg) for a local or global library, respectively.")
            
    def _oh2seq_cpu(self, one_hot):
        """Convert one-hot encoding to sequence using CPU."""
        seq_index = np.argmax(one_hot, axis=1)
        alphabet_dict = dict(enumerate(self.alphabet))
        seq = [alphabet_dict[i] for i in seq_index]
        return ''.join(seq)
    
    def _oh2seq_gpu(self, x):
        """Convert batch of one-hot encodings to sequences using GPU."""
        # Get indices of 1s for all sequences at once
        seq_indices = np.argmax(x, axis=-1)
        
        # Create mapping of indices to alphabet characters
        num2alpha = dict(enumerate(self.alphabet))
        
        # Vectorize the conversion
        seq_chars = np.vectorize(num2alpha.get)(seq_indices)
        
        # Convert to list of sequences
        sequences = [''.join(seq) for seq in seq_chars]
        return sequences
    
    def _compute_hamming_cpu(self, seq1, seq2):
        """Compute Hamming distance between two sequences using CPU."""
        return int(distance.hamming(list(seq1), list(seq2)) * len(seq1))
    
    def _compute_hamming_gpu(self, ref_seq, sequences):
        """Compute Hamming distances for all sequences at once using GPU."""
        # Use scipy.spatial.distance.hamming for consistency with CPU version
        distances = []
        ref_list = list(ref_seq)
        for seq in sequences:
            d = distance.hamming(ref_list, list(seq)) * len(ref_seq)
            distances.append(int(d))
        return np.array(distances)
    
    def _compute_gia(self, y_pred, y_bg):
        """Compute GIA score."""
        return y_pred - y_bg
    
    def compile(self):
        """Compile data into pandas DataFrame."""
        print("Compiling data...")
        
        # Initialize DataFrame
        N = len(self.x)
        df = pd.DataFrame()
        
        # Convert sequences and add DNN predictions
        print("Converting sequences...")
        if self.gpu:
            sequences = self._oh2seq_gpu(self.x)
        else:
            sequences = []
            for i in tqdm(range(N), desc='Processing'):
                seq = self._oh2seq_cpu(self.x[i])
                sequences.append(seq)
        
        df['DNN'] = self.y.flatten()
        df['Sequence'] = sequences
        
        # Compute Hamming distances if reference provided
        if self.x_ref is not None:
            print("Computing Hamming distances...")
            ref_seq = self._oh2seq_cpu(self.x_ref[0]) if not self.gpu else self._oh2seq_gpu(self.x_ref)[0]
            
            if self.gpu:
                df['Hamming'] = self._compute_hamming_gpu(ref_seq, sequences)
            else:
                df['Hamming'] = [self._compute_hamming_cpu(ref_seq, seq) for seq in tqdm(sequences, desc='Hamming')]
        
        # Compute GIA scores if background provided
        if self.y_bg is not None:
            print("Computing GIA scores...")
            df['GIA'] = self._compute_gia(self.y, self.y_bg).flatten()
        
        # Reorder columns
        cols = ['DNN']
        if 'Hamming' in df.columns:
            cols.append('Hamming')
        if 'GIA' in df.columns:
            cols.append('GIA')
        cols.append('Sequence')
        
        return df[cols]
    

# TODO:
# - SEAM utils calls
# - Task-specific predictions in each column
# - Move hamming distance calculation to SEAM utils?
# - save output to self.mave
# - docstring check (readthedocs compatibility)