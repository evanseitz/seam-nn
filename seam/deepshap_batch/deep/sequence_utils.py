import numpy as np
import tensorflow as tf
from typing import Union, Tuple, Optional

def get_dinucleotide_frequencies(sequence: np.ndarray) -> np.ndarray:
    """Calculate dinucleotide frequencies from one-hot encoded sequence.
    
    Parameters
    ----------
    sequence : np.ndarray
        One-hot encoded sequence with shape (length, 4)
        
    Returns
    -------
    np.ndarray
        Dinucleotide frequency matrix with shape (4, 4)
    """
    # Convert one-hot to base indices
    base_indices = np.argmax(sequence, axis=1)
    
    # Calculate frequencies
    frequencies = np.zeros((4, 4))
    for i in range(len(base_indices) - 1):
        frequencies[base_indices[i], base_indices[i + 1]] += 1
    
    # Normalize
    frequencies = frequencies / np.sum(frequencies)
    return frequencies

def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0.
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens

def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]

def dinuc_shuffle(seq, rng=None):
    """
    Creates shuffles of the given sequence, preserving dinucleotide frequencies.
    
    Parameters
    ----------
    seq : np.ndarray
        L x D one-hot encoded sequence
    rng : np.random.RandomState, optional
        Random number generator
        
    Returns
    -------
    np.ndarray
        Shuffled sequence
    """
    if rng is None:
        rng = np.random.RandomState()
        
    # Convert to tokens
    tokens = one_hot_to_tokens(seq)
    
    # Get unique tokens
    chars, tokens = np.unique(tokens, return_inverse=True)
    
    # Get next indices for each token
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)
    
    # Shuffle next indices
    for t in range(len(chars)):
        inds = np.arange(len(shuf_next_inds[t]))
        if len(inds) > 1:
            inds[:-1] = rng.permutation(len(inds) - 1)
        shuf_next_inds[t] = shuf_next_inds[t][inds]
    
    # Build result
    counters = [0] * len(chars)
    ind = 0
    result = np.empty_like(tokens)
    result[0] = tokens[ind]
    
    for j in range(1, len(tokens)):
        t = tokens[ind]
        ind = shuf_next_inds[t][counters[t]]
        counters[t] += 1
        result[j] = tokens[ind]
    
    return tokens_to_one_hot(chars[result], seq.shape[1])

def batch_dinuc_shuffle(
    sequence: Union[np.ndarray, tf.Tensor],
    num_shuffles: int,
    seed: Optional[int] = 1234
) -> tf.Tensor:
    """Generate multiple dinucleotide-preserved shuffles efficiently.
    
    Parameters
    ----------
    sequence : array-like
        One-hot encoded sequence with shape (length, 4)
    num_shuffles : int
        Number of shuffled sequences to generate
    seed : int, optional
        Random seed for reproducibility (defaults to 1234 to match original)
        
    Returns
    -------
    tf.Tensor
        Batch of shuffled sequences with shape (num_shuffles, length, 4)
    """
    # Convert to numpy if needed
    if isinstance(sequence, tf.Tensor):
        sequence = sequence.numpy()
    
    # Ensure sequence is 2D
    if len(sequence.shape) == 3:
        sequence = sequence[0]  # Take first sequence if batched
    
    # Create RandomState with seed (matching original implementation)
    rng = np.random.RandomState(seed)
    
    # Generate shuffles using the same RNG
    shuffled_sequences = [dinuc_shuffle(sequence, rng=rng) for _ in range(num_shuffles)]
    
    # Stack and convert to tensor, ensuring shape (num_shuffles, length, 4)
    shuffled_tensor = tf.convert_to_tensor(
        np.stack(shuffled_sequences, axis=0),
        dtype=tf.float32
    )
    
    return shuffled_tensor  # Now returns shape (num_shuffles, length, 4) 