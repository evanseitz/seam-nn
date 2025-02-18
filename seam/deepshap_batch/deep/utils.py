"""
Shared utility functions for deep learning explainers.
"""

import tensorflow as tf
import torch

def standard_combine_mult_and_diffref(mult, orig_inp, bg_data):
    """Standard DeepLIFT combination function that works with both PyTorch and TensorFlow."""
    if isinstance(mult, torch.Tensor):
        # PyTorch implementation
        if isinstance(mult, list):
            return [(m*(o - b)).mean(0) for m, o, b in zip(mult, orig_inp, bg_data)]
        return (mult*(orig_inp - bg_data)).mean(0)
    else:
        # TensorFlow implementation
        if isinstance(mult, list):
            return [tf.reduce_mean(m*(o - b), axis=0) for m, o, b in zip(mult, orig_inp, bg_data)]
        return tf.reduce_mean(mult*(orig_inp - bg_data), axis=0)
