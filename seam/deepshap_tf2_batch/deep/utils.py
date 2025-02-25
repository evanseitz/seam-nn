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
    #print("HERE!")
    """Standard DeepLIFT combination function that works with PyTorch, TensorFlow, and NumPy."""
    
    #print("\n=== Debug utils.py ===")
    #print(f"mult[0] first few values: {mult[0].numpy().flatten()[:5]}")
    #print(f"orig_inp[0] first few values: {orig_inp[0].numpy().flatten()[:5]}")
    #print(f"bg_data[0] first few values: {bg_data[0].numpy().flatten()[:5]}")
    
    import numpy as np
    
    # Try importing torch and tensorflow, but don't fail if they're not available
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        
    try:
        import tensorflow as tf
        has_tf = True
    except ImportError:
        has_tf = False
    
    # Check if mult is a list of arrays/tensors
    if not isinstance(mult, list):
        mult = [mult]
    if not isinstance(orig_inp, list):
        orig_inp = [orig_inp]
    if not isinstance(bg_data, list):
        bg_data = [bg_data]
        
    # Determine the type and use appropriate implementation
    if has_torch and isinstance(mult[0], torch.Tensor):
        print('Using PyTorch implementation')
        return [(m*(o - b)).mean(0) for m, o, b in zip(mult, orig_inp, bg_data)]
    elif has_tf and isinstance(mult[0], tf.Tensor):
        print('Using TensorFlow implementation')
        return [tf.reduce_mean(m*(o - b), axis=0) for m, o, b in zip(mult, orig_inp, bg_data)]
    else:
        print('Using NumPy implementation')

        if 1: # correct
            assert len(orig_inp) == 1, "Only single input supported for numpy implementation"
            projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
            assert len(orig_inp[0].shape) == 2, "Input must be 2D for numpy implementation"
            
            for i in range(orig_inp[0].shape[-1]):
                # 1. Create a one-hot vector for each feature position
                # For a 4-feature input, if i=1, this creates [0,1,0,0] for each sample
                hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")
                hypothetical_input[:, i] = 1.0
                
                # 2. Compute difference between this one-hot input and all background samples
                # The [None,:,:] adds a dimension to match bg_data shape (n_background, n_samples, n_features)
                hypothetical_difference_from_reference = (hypothetical_input[None,:,:] - bg_data[0])
                
                # 3. Multiply differences by gradients (mult[0])
                # mult[0] shape is (n_background, n_samples, n_features)
                # This gives contribution of each feature to each output
                hypothetical_contribs = hypothetical_difference_from_reference * mult[0]
                
                # 4. Sum across features to get total contribution for this one-hot position
                # Result shape is (n_background, n_samples)
                projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs, axis=-1)
            
            # 5. Average across background samples to get final attribution
            # Returns shape (n_samples, n_features)
            return [np.mean(projected_hypothetical_contribs, axis=0)]

        else: # incorrect
            return [(m*(o - b)).mean(0) for m, o, b in zip(mult, orig_inp, bg_data)]