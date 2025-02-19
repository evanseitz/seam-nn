"""
Shared utility functions for deep learning explainers.
"""
try:    
    import tensorflow as tf
except ImportError:
    pass
try:
    import torch
except ImportError:
    pass

def standard_combine_mult_and_diffref(mult, orig_inp, bg_data):
    """Standard DeepLIFT combination function that works with both PyTorch and TensorFlow."""
    print("\n=== Debug utils.py ===")
    # Handle both numpy arrays and tf tensors
    if hasattr(mult[0], 'numpy'):
        print(f"mult[0] first few values: {mult[0].numpy().flatten()[:5]}")
        print(f"orig_inp[0] first few values: {orig_inp[0].numpy().flatten()[:5]}")
        print(f"bg_data[0] first few values: {bg_data[0].numpy().flatten()[:5]}")
    else:
        print(f"mult[0] first few values: {mult[0].flatten()[:5]}")
        print(f"orig_inp[0] first few values: {orig_inp[0].flatten()[:5]}")
        print(f"bg_data[0] first few values: {bg_data[0].flatten()[:5]}")
    
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        
    if has_torch and isinstance(mult[0], torch.Tensor):
        return [(m*(o - b)).mean(0) for m, o, b in zip(mult, orig_inp, bg_data)]
    else:
        if hasattr(mult[0], 'numpy'):  # TensorFlow tensor
            print("Using TensorFlow implementation")
            return [tf.reduce_mean(m*(o - b), axis=0) for m, o, b in zip(mult, orig_inp, bg_data)]
        else:  # Numpy array
            print("Using NumPy implementation")
            return [(m*(o - b)).mean(0) for m, o, b in zip(mult, orig_inp, bg_data)]