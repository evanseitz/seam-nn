"""
Shared utility functions for deep learning explainers.
"""

import tensorflow as tf
try:    
    import torch
except ImportError:
    torch = None

def standard_combine_mult_and_diffref(mult, orig_inp, bg_data):
    """Standard DeepLIFT combination function that works with PyTorch, TensorFlow, and NumPy."""
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        
    if has_torch and isinstance(mult, torch.Tensor):
        # PyTorch implementation
        if isinstance(mult, list):
            return [(m*(o - b)).mean(0) for m, o, b in zip(mult, orig_inp, bg_data)]
        return (mult*(orig_inp - bg_data)).mean(0)
    else:
        # TensorFlow/NumPy implementation
        if isinstance(mult, list):
            return [tf.reduce_mean(m*(o - b), axis=0) for m, o, b in zip(mult, orig_inp, bg_data)]
        return tf.reduce_mean(mult*(orig_inp - bg_data), axis=0)
