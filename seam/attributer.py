import os
import sys
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from typing import Union, Optional

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

class Attributer:
    """
    Attributer: A unified interface for computing attribution maps in TensorFlow 2.x

    This implementation is optimized for TensorFlow 2.x and provides
    GPU-accelerated implementations of common attribution methods:
    - Saliency Maps
    - SmoothGrad
    - Integrated Gradients
    - DeepSHAP (via SHAP package, requires TensorFlow setup before initialization - see below)
    - ISM (In-Silico Mutagenesis)

    Requirements:
    - tensorflow
    - numpy
    - tqdm
    - shap (for DeepSHAP only)

    Key Features:
    - Batch processing for saliency, smoothgrad, integrated gradients, and ISM
    - DeepSHAP processes sequences one at a time (no batch mode)
    - GPU-optimized implementations for saliency, smoothgrad, and integrated gradients
    - Consistent interface across methods
    - Support for multi-head models
    - Memory-efficient processing of large datasets
    - Flexible sequence windowing for long sequences

    Example usage:
        # Basic usage with output reduction function
        attributer = Attributer(
            model, 
            method='saliency',
            task_index=0,  # Select first output head
            compress_fun=tf.math.reduce_mean,  # Reduce selected output to scalar
            pred_fun=None  # Not used for gradient-based methods
        )

        # Example with ChromBPNet compression functions
        attributer = Attributer(
            model,
            method='deepshap',
            task_index=0,  # Select first output head
            compress_fun=Attributer.bpnet_profile_deepshap,  # ChromBPNet profile compression with stop_gradient
            pred_fun=None
        )

        # Example with ISM (forward-pass method)
        attributer = Attributer(
            model,
            method='ism',
            task_index=0,  # Select first output head
            compress_fun=tf.math.reduce_mean,  # Reduce selected output to scalar
            pred_fun=model.predict_on_batch  # Optional: use predict_on_batch for ISM
        )

        # Computing attributions for a specific window while maintaining full context
        attributions = attributer.compute(
            x=input_sequences,          # Shape: (N, window_size, A)
            x_ref=reference_sequence,   # Shape: (1, full_length, A)
            save_window=[100, 200],     # Compute attributions for positions 100-200
            batch_size=128
        )

        # Method-specific parameters
        attributions = attributer.compute(
            x=input_sequences,
            num_steps=20,          # for intgrad
            num_samples=20,        # for smoothgrad
            multiply_by_inputs=False  # for intgrad
            log2fc=False  # for ism
        )

    Note: For optimal performance, ensure TensorFlow is configured to use GPU acceleration.
    
    DeepSHAP Requirements:
    DeepSHAP requires specific TensorFlow setup that must be done BEFORE creating the Attributer
    (because DeepSHAP was designed for earlier TensorFlow versions):
    1. Disable TensorFlow eager execution: tf.compat.v1.disable_eager_execution()
    2. Disable TensorFlow v2 behavior: tf.compat.v1.disable_v2_behavior() 
    3. Load/reload the model from file after disabling eager execution
    4. Rebuild the model graph by passing a dummy input through it
    5. Configure SHAP op handlers for TensorFlow compatibility
    
    Example setup sequence:
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.disable_v2_behavior()
        import shap
        shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        _ = model(tf.keras.Input(shape=model.input_shape[1:]))
        # Now create Attributer with the prepared model
    """
    
    SUPPORTED_METHODS = {'saliency', 'smoothgrad', 'intgrad', 'deepshap', 'ism'}

    # Define default batch sizes for each method
    DEFAULT_BATCH_SIZES = {
        'saliency': 128,
        'intgrad': 128,
        'smoothgrad': 64,
        'ism': 32
        # Note: DeepSHAP does not use batch processing - it processes sequences one at a time
    }
    
    def __init__(self, model, method='saliency', task_index=None,
                batch_size=None, num_shuffles=20, compress_fun=tf.math.reduce_mean, 
                pred_fun=None, gpu=True):
        """Initialize the Attributer.
        
        Args:
            model: TensorFlow model to explain
            method: Attribution method (default: 'saliency')
            task_index: Index of output head to explain (optional)
            batch_size: Batch size for computing attributions (optional, defaults to method-specific size)
            num_shuffles: Number of shuffles for DeepSHAP background (default: 20, matches ChromBPNet)
            compress_fun: Function to compress model output to scalar (default: tf.math.reduce_mean)
            pred_fun: Function to use for model predictions in forward-pass methods like ISM.
                     Not used for gradient-based methods (saliency, smoothgrad, intgrad).
                     Default: model.__call__
            gpu: Whether to use GPU-optimized implementation (default: True)

        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
            
        self.model = model
        self.method = method
        self.task_index = task_index
        self.compress_fun = compress_fun
        self.pred_fun = pred_fun or model.__call__
        self.gpu = gpu
        self.num_shuffles = num_shuffles

        # Set batch size based on method if not specified
        # Note: DeepSHAP does not use batch processing
        if method == 'deepshap':
            self.batch_size = None  # DeepSHAP doesn't use batching
            self._check_deepshap_dependencies()
        else:
            self.batch_size = batch_size or self.DEFAULT_BATCH_SIZES[method]

    @tf.function
    def _saliency_map(self, X):
        """Compute saliency maps."""
        if not tf.is_tensor(X):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        else:
            X = tf.cast(X, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X)
            pred = self.model(X)
            if self.task_index is not None:
                pred = pred[self.task_index]
            
            outputs = self.compress_fun(pred)
        return tape.gradient(outputs, X)

    def saliency(self, X, batch_size=None):
        """Compute saliency maps in batches."""
        return self._function_batch(X, self._saliency_map, 
                                  batch_size or self.batch_size)

    def smoothgrad(self, X, num_samples=20, mean=0.0, stddev=0.1, gpu=True, **kwargs):
        """Compute SmoothGrad attribution maps.
        
        Args:
            X: Input tensor of shape (batch_size, L, A)
            num_samples: Number of noisy samples
            mean: Mean of noise
            stddev: Standard deviation of noise
            gpu: Whether to use GPU-optimized implementation
            **kwargs: Additional arguments (ignored)
        
        Returns:
            numpy.ndarray: Attribution maps of shape (batch_size, L, A)
        """
        if gpu:
            return self._smoothgrad_gpu(X, num_samples, mean, stddev)
        else:
            return self._smoothgrad_cpu(X, num_samples, mean, stddev)

    def _smoothgrad_cpu(self, X, num_samples=20, mean=0.0, stddev=0.1):
        """CPU implementation of SmoothGrad."""
        scores = []
        for x in X:
            x = np.expand_dims(x, axis=0)  # (1, L, A)
            x = tf.cast(x, dtype=tf.float32)
            x_noisy = tf.tile(x, (num_samples,1,1)) + tf.random.normal((num_samples,x.shape[1],x.shape[2]), mean, stddev)
            grad = self._saliency_map(x_noisy)
            scores.append(tf.reduce_mean(grad, axis=0))
        return np.stack(scores, axis=0)

    def _smoothgrad_gpu(self, X, num_samples=20, mean=0.0, stddev=0.1):
        """GPU-optimized implementation with parallel noise generation."""
        X = tf.cast(X, dtype=tf.float32)
        batch_size = tf.shape(X)[0]
        
        # Expand X to (batch_size, 1, L, A) for broadcasting
        X_expanded = tf.expand_dims(X, axis=1)
        
        # Tile along samples dimension to (batch_size, num_samples, L, A)
        X_tiled = tf.tile(X_expanded, [1, num_samples, 1, 1])
        
        # Generate noise (batch_size, num_samples, L, A)
        noise = tf.random.normal(tf.shape(X_tiled), mean, stddev)
        
        # Add noise
        X_noisy = X_tiled + noise
        
        # Reshape to (batch_size * num_samples, L, A) for gradient computation
        X_reshaped = tf.reshape(X_noisy, [-1, tf.shape(X)[1], tf.shape(X)[2]])
        
        # Compute gradients
        grads = self._saliency_map(X_reshaped)
        
        # Reshape back to (batch_size, num_samples, L, A)
        grads = tf.reshape(grads, [batch_size, num_samples, tf.shape(X)[1], tf.shape(X)[2]])
        
        # Average over samples
        return tf.reduce_mean(grads, axis=1)
    
    def intgrad(self, X, baseline_type='zeros', num_steps=20, gpu=True, multiply_by_inputs=False, seed=None):
        """Compute Integrated Gradients attribution maps.
        
        Parameters
        ----------
        X : array-like
            Input sequences
        baseline_type : str
            Type of baseline to use:
            - 'zeros': Zero baseline
            - 'random_shuffle': Random shuffle of input sequence
            - 'dinuc_shuffle': Dinucleotide-preserved shuffle of input sequence (default)
        num_steps : int
            Number of steps for integration
        gpu : bool
            Whether to use GPU-optimized implementation
        multiply_by_inputs : bool
            Whether to multiply gradients by inputs
        seed : int, optional
            Random seed for reproducibility in shuffling methods
            
        Returns
        -------
        array-like
            Attribution maps
        """
        if gpu:
            return self._intgrad_gpu(X, baseline_type, num_steps, multiply_by_inputs, seed=seed)
        else:
            return self._intgrad_cpu(X, baseline_type, num_steps, multiply_by_inputs, seed=seed)
    
    def _integrated_grad(self, x, baseline, num_steps, multiply_by_inputs=False):
        """Compute Integrated Gradients for a single input."""
        alphas = tf.linspace(0.0, 1.0, num_steps+1)
        alphas = alphas[:, tf.newaxis, tf.newaxis]
        path_inputs = baseline + alphas * (x - baseline)
        grads = self._saliency_map(path_inputs)
        
        # Riemann trapezoidal approximation
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0, keepdims=True)  # Keep batch dim: (1, L, A)
        
        if multiply_by_inputs:
            return avg_grads * (x - baseline)
        return avg_grads
    
    def _intgrad_cpu(self, X, baseline_type='zeros', num_steps=20, multiply_by_inputs=False, seed=None):
        """CPU-optimized implementation using loop-based computation."""
        scores = []
        #for i, x in enumerate(tqdm(X, desc="Computing IntGrad")):
        for i, x in enumerate(X):
            x = np.expand_dims(x, axis=0)  # Add batch dimension: (1, L, A)

            if seed is not None:
                np.random.seed(seed)
            
            # Explicitly handle each baseline type
            if baseline_type == 'zeros':
                baseline = np.zeros_like(x)
            elif baseline_type == 'random_shuffle':
                baseline = self._random_shuffle(x)
            elif baseline_type == 'dinuc_shuffle':
                if i == 0:  # Only compute all shuffles once at the start
                    try:
                        from deeplift.dinuc_shuffle import dinuc_shuffle
                        baselines = np.array([dinuc_shuffle(X[i]) for i in range(X.shape[0])])
                    except ImportError:
                        raise ImportError("dinuc_shuffle baseline requires deeplift package. Install with: pip install deeplift")
                baseline = np.expand_dims(baselines[i], axis=0)
            else:
                raise ValueError("baseline_type must be one of: 'zeros', 'random_shuffle', 'dinuc_shuffle'")
            
            score = self._integrated_grad(x, baseline, num_steps, multiply_by_inputs)
            scores.append(score[0])  # Remove batch dimension before appending
        return np.stack(scores, axis=0)  # Stack to get (N, L, A)

    @tf.function
    def _intgrad_gpu(self, X, baseline_type='zeros', num_steps=20, multiply_by_inputs=False, seed=None):
        """GPU-optimized implementation using vectorized computation."""
        # Ensure input is float32
        X = tf.cast(X, tf.float32)
        
        if seed is not None:
            tf.random.set_seed(seed)
        
        # Explicitly handle each baseline type
        if baseline_type == 'zeros':
            baseline = tf.zeros_like(X, dtype=tf.float32)
        elif baseline_type == 'random_shuffle':
            baseline = tf.map_fn(self._random_shuffle, X)
        elif baseline_type == 'dinuc_shuffle':
            try:
                from deeplift.dinuc_shuffle import dinuc_shuffle
                baselines = np.array([dinuc_shuffle(X[i].numpy()) for i in range(X.shape[0])])
                baseline = tf.convert_to_tensor(baselines, dtype=tf.float32)
            except ImportError:
                raise ImportError("dinuc_shuffle baseline requires deeplift package. Install with: pip install deeplift")
        else:
            raise ValueError("baseline_type must be one of: 'zeros', 'random_shuffle', 'dinuc_shuffle'")
        
        # Compute path inputs for all samples at once
        alphas = tf.linspace(0.0, 1.0, num_steps+1)
        alphas = tf.cast(alphas[:, tf.newaxis, tf.newaxis, tf.newaxis], tf.float32)
        
        # Expand dimensions for broadcasting
        X = X[tf.newaxis, ...]         # shape: (1, batch, L, A)
        baseline = baseline[tf.newaxis, ...]  # shape: (1, batch, L, A)
        
        path_inputs = baseline + alphas * (X - baseline)  # shape: (steps, batch, L, A)
        
        # Reshape to (steps*batch, L, A) for efficient gradient computation
        batch_size = tf.shape(X)[1]
        path_inputs_reshape = tf.reshape(path_inputs, (-1, tf.shape(X)[2], tf.shape(X)[3]))
        
        grads = self._saliency_map(path_inputs_reshape)
        grads = tf.reshape(grads, (num_steps+1, batch_size, -1, tf.shape(X)[3]))
        
        # Riemann trapezoidal approximation
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.reduce_mean(grads, axis=0)
        
        if multiply_by_inputs:
            return avg_grads * (X[0] - baseline[0])
        return avg_grads
    
    @tf.function
    def _intgrad_gpu_matches_cpu(self, X, baseline_type='zeros', num_steps=20, multiply_by_inputs=False, seed=None):
        """GPU-optimized implementation using individual sample processing for mathematical equivalence."""
        # Ensure input is float32
        X = tf.cast(X, tf.float32)
        
        if seed is not None:
            tf.random.set_seed(seed)
        
        # Explicitly handle each baseline type
        if baseline_type == 'zeros':
            baseline = tf.zeros_like(X, dtype=tf.float32)
        elif baseline_type == 'random_shuffle':
            baseline = tf.map_fn(self._random_shuffle, X)
        elif baseline_type == 'dinuc_shuffle':
            try:
                from deeplift.dinuc_shuffle import dinuc_shuffle
                baselines = np.array([dinuc_shuffle(X[i].numpy()) for i in range(X.shape[0])])
                baseline = tf.convert_to_tensor(baselines, dtype=tf.float32)
            except ImportError:
                raise ImportError("dinuc_shuffle baseline requires deeplift package. Install with: pip install deeplift")
        else:
            raise ValueError("baseline_type must be one of: 'zeros', 'random_shuffle', 'dinuc_shuffle'")
        
        # Process each sample individually for mathematical equivalence with CPU version
        batch_size = tf.shape(X)[0]
        
        # Use tf.map_fn to process each sample individually
        def process_single_sample(args):
            x, sample_baseline = args
            
            # Add batch dimension back for processing
            x = tf.expand_dims(x, axis=0)  # Shape: (1, L, A)
            sample_baseline = tf.expand_dims(sample_baseline, axis=0)  # Shape: (1, L, A)
            
            # Compute path inputs for single sample
            alphas = tf.linspace(0.0, 1.0, num_steps+1)
            alphas = tf.cast(alphas[:, tf.newaxis, tf.newaxis], tf.float32)  # Shape: (num_steps+1, 1, 1)
            
            path_inputs = sample_baseline + alphas * (x - sample_baseline)  # Shape: (num_steps+1, 1, L, A)
            
            # Compute gradients for single sample (same as CPU version)
            grads = self._saliency_map(path_inputs)  # Shape: (num_steps+1, 1, L, A)
            
            # Riemann trapezoidal approximation
            grads = (grads[:-1] + grads[1:]) / 2.0  # Shape: (num_steps, 1, L, A)
            avg_grads = tf.reduce_mean(grads, axis=0, keepdims=True)  # Shape: (1, 1, L, A)
            
            if multiply_by_inputs:
                score = avg_grads * (x - sample_baseline)
            else:
                score = avg_grads
            
            return score[0]  # Remove batch dimension: Shape: (L, A)
        
        # Process all samples using map_fn
        scores = tf.map_fn(
            process_single_sample,
            (X, baseline),
            fn_output_signature=tf.float32
        )
        
        return scores  # Shape: (batch, L, A)
    
    def ism(self, X, log2fc=False, gpu=True, snv_window=None):
        """Compute In-Silico Mutagenesis attribution maps.
        
        Args:
            X: Input tensor of shape (batch_size, L, A)
            log2fc: Whether to compute log2 fold change instead of difference
            gpu: Whether to attempt GPU-optimized implementation
            snv_window: Optional [start, end] positions to compute variants for.
                       If None, compute for all positions.
        
        Returns:
            numpy.ndarray: Attribution maps of shape (batch_size, L, A)
        """
        try:
            if gpu:
                try:
                    return self._ism_gpu(X, log2fc, snv_window)
                except Exception as e:
                    print(f"GPU implementation failed with error: {str(e)}")
                    print("Falling back to CPU")
        except:
            print("GPU implementation failed, falling back to CPU")
        return self._ism_cpu(X, log2fc, snv_window)
    

    def _ism_cpu(self, X, log2fc=False, snv_window=None):
        """CPU implementation of ISM.
        
        Args:
            X: Input sequences
            log2fc: Whether to compute log2 fold change
            snv_window: Optional [start, end] positions to compute variants for
        """
        X = X.astype(np.float32)  # Ensure float32
        scores = []
        
        # Handle SNV window if provided
        N, L, A = X.shape
        if snv_window is not None:
            start, end = snv_window
            if start < 0 or end > L or start >= end:
                raise ValueError(f"Invalid snv_window [{start}, {end}]. Must be within [0, {L}] and start < end")
            window_length = end - start
        else:
            start, end = 0, L
            window_length = L
        
        # Pre-allocate mutation array for reuse
        mut_seq = np.zeros_like(X[0:1], dtype=np.float32)
        
        for x in X:
            x = x[np.newaxis]
            # Create score matrix for just the window
            score_matrix = np.zeros((window_length, A), dtype=np.float32)
            
            # Get wild-type predictions
            wt_output = self.pred_fun(tf.constant(x, dtype=tf.float32))
            if self.task_index is not None:
                wt_output = wt_output[self.task_index]

            wt_pred = float(self.compress_fun(wt_output))
            
            # Store all mutation predictions first
            mut_preds = np.empty((window_length * (A-1)), dtype=np.float32)
            mut_locs = []
            
            # Reuse pre-allocated array
            mut_seq[:] = x
            
            idx = 0
            for pos in range(start, end):
                for b in range(1, A):
                    mut_seq[0, pos] = np.roll(x[0, pos], b)
                    new_base = np.where(mut_seq[0, pos] == 1)[0][0]
                    
                    mut_output = self.pred_fun(tf.constant(mut_seq))
                    if self.task_index is not None:
                        mut_output = mut_output[self.task_index]

                    mut_preds[idx] = float(self.compress_fun(mut_output))
                    
                    mut_locs.append((pos, new_base))
                    idx += 1
                    
                    # Restore original sequence for next position
                    mut_seq[0, pos] = x[0, pos]
            
            # Apply log2fc calculation after collecting all predictions
            if log2fc:
                pred_min = min(min(mut_preds), wt_pred)
                offset = abs(pred_min) + 1
                wt_pred_adj = wt_pred + offset
                
                for (pos, new_base), mut_pred in zip(mut_locs, mut_preds):
                    if mut_pred != wt_pred:
                        score_matrix[pos-start, new_base] = np.log2(mut_pred + offset) - np.log2(wt_pred_adj)
                    else:
                        score_matrix[pos-start, new_base] = 0.
            else:
                for (pos, new_base), mut_pred in zip(mut_locs, mut_preds):
                    score_matrix[pos-start, new_base] = mut_pred - wt_pred
                        
            scores.append(score_matrix)
        
        scores = np.stack(scores, axis=0)
        
        # If using SNV window, pad with zeros
        if snv_window is not None:
            full_scores = np.zeros((N, L, A), dtype=np.float32)
            full_scores[:, start:end, :] = scores
            scores = full_scores
            
        return scores
    
    def _ism_gpu(self, X, log2fc=False, snv_window=None):
        """GPU-accelerated implementation of In-Silico Mutagenesis.
        
        This implementation achieves significant speedup over the CPU version by:
        1. Generating all possible mutations for each input sequence in parallel
        2. Identifying and processing only unique mutations to avoid redundant predictions
        3. Using batched inference on GPU
        4. Efficiently restoring the full mutation set using index mapping
        
        Args:
            X: Input sequences of shape (N, L, A) where:
               N = number of sequences
               L = sequence length
               A = alphabet size (typically 4 for DNA/RNA)
            log2fc: If True, compute log2 fold change instead of simple difference
            snv_window: Optional [start, end] positions to specifiy contiguous region over which variants are computed.
                       If None, compute for all positions.
            pred_fun: Function to use for model predictions (default: self.model)
        
        Returns:
            numpy.ndarray: Attribution maps of shape (N, L, A) containing the effect
            of each possible mutation at each position.
        """
        # Convert input to tensor if needed
        if not isinstance(X, tf.Tensor):
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        N, L, A = X.shape
        
        # Handle SNV window if provided
        if snv_window is not None:
            start, end = snv_window
            if start < 0 or end > L or start >= end:
                raise ValueError(f"Invalid snv_window [{start}, {end}]. Must be within [0, {L}] and start < end")
            window_length = end - start
            mutations_per_seq = window_length * A
        else:
            start, end = 0, L
            window_length = L
            mutations_per_seq = L * A

        # Generate all possible single-nucleotide mutations for each sequence
        # First, tile each input sequence
        X_tiled = tf.repeat(X, mutations_per_seq, axis=0)  # Shape: (N*L*A, L, A)

        # Create position indices for mutations
        if snv_window is not None:
            pos_indices = tf.repeat(tf.range(start, end), A)  # Shape: (window_length*A,)
        else:
            pos_indices = tf.repeat(tf.range(L), A)  # Shape: (L*A,)
        pos_indices = tf.tile(pos_indices, [N])  # Shape: (N*window_length*A,)

        # Create base indices for mutations
        base_indices = tf.tile(tf.range(A), [window_length])  # Shape: (window_length*A,)
        base_indices = tf.tile(base_indices, [N])  # Shape: (N*window_length*A,)

        # Create update indices for scatter_nd
        update_indices = tf.stack([tf.range(N * mutations_per_seq), pos_indices], axis=1)

        # Create one-hot vectors for mutations
        mutation_vectors = tf.one_hot(base_indices, A)  # Shape: (N*window_length*A, A)

        # Apply mutations using single scatter_nd operation
        all_mutations = tf.tensor_scatter_nd_update(
            X_tiled,
            update_indices,
            mutation_vectors
        )

        # Get wild-type predictions
        wt_preds = self.pred_fun(X)

        if self.task_index is not None:
            wt_preds = wt_preds[self.task_index]
        wt_preds = tf.cast(self.compress_fun(wt_preds), tf.float32)  # Apply compression to get scalar values
                            
        # Find unique mutations
        flattened_mutations = tf.reshape(all_mutations, [-1, L * A])
        string_mutations = tf.strings.reduce_join(tf.strings.as_string(flattened_mutations), axis=1)
        unique_mutations, restore_indices = tf.unique(string_mutations)[0:2]
        num_unique = tf.shape(unique_mutations)[0]
        
        # Get indices of first occurrences
        # Create a boolean mask for each unique value
        matches = tf.equal(restore_indices[:, tf.newaxis], tf.range(num_unique))
        # Find the first True value for each unique mutation
        matches = tf.cast(matches, tf.int64)  # Cast to int64 before argmax
        unique_indices = tf.argmax(matches, axis=0)
        unique_mutations = tf.gather(all_mutations, unique_indices)

        # Run inference in batches on unique mutations
        # Pre-allocate a tensor for all predictions
        mut_preds = tf.zeros((num_unique,), dtype=tf.float32)
        
        # Process unique mutations in batches
        for i in range(0, num_unique, self.batch_size):
            end_idx = min(i + self.batch_size, num_unique)
            batch = unique_mutations[i:end_idx]
            batch_preds = self.pred_fun(batch)
            
            if self.task_index is not None:
                batch_preds = batch_preds[self.task_index]

            # Apply compression function to get scalar values
            batch_preds = tf.cast(self.compress_fun(batch_preds), tf.float32)
            
            # Update the pre-allocated tensor
            mut_preds = tf.tensor_scatter_nd_update(
                mut_preds,
                tf.reshape(tf.range(i, end_idx), [-1, 1]),  # Shape: [batch_size, 1]
                tf.reshape(batch_preds, [-1])  # Shape: [batch_size]
            )
        
        # Stack predictions and map back to full mutation set
        mut_preds = tf.concat(mut_preds, axis=0)  # Shape: (num_unique,)
        
        # Create mapping from mutations back to their original sequences
        sequence_indices = tf.repeat(tf.range(N), mutations_per_seq)  # Shape: (N * mutations_per_seq,)
        
        # Ensure mut_preds has correct shape for gathering
        mut_preds = tf.reshape(mut_preds, [-1])  # Ensure 1D tensor # TODO: is the generalizable?
        
        # Restore predictions to full mutation set and get corresponding wild-types
        full_mut_preds = tf.gather(mut_preds, restore_indices)  # Shape: (N * L * A,)            
        wt_preds_for_mutations = tf.gather(wt_preds, sequence_indices)  # Shape: (N * L * A,)
        # Handle both squeezed and unsqueezed wild-type predictions
        if wt_preds_for_mutations.shape.ndims > 1:
            wt_preds_for_mutations = tf.squeeze(wt_preds_for_mutations)  # Remove extra dimension if present

        if log2fc:
            # Reshape predictions to group by input sequence
            mut_preds_per_seq = tf.reshape(full_mut_preds, [N, mutations_per_seq])  # Shape: (N, L*A)
            wt_preds_expanded = tf.repeat(wt_preds, mutations_per_seq)  # Shape: (N * L * A,)
            wt_preds_per_seq = tf.reshape(wt_preds_expanded, [N, mutations_per_seq])  # Shape: (N, L*A)
            
            # Calculate offset per sequence
            all_preds_per_seq = tf.concat([mut_preds_per_seq, wt_preds_per_seq[:, :1]], axis=1)  # Shape: (N, L*A + 1)
            pred_mins = tf.reduce_min(all_preds_per_seq, axis=1, keepdims=True)  # Shape: (N, 1)
            offsets = tf.abs(pred_mins) + 1.0  # Shape: (N, 1)
            
            # Apply log2fc calculation per sequence
            log2_mut = tf.math.log(mut_preds_per_seq + offsets) / tf.math.log(2.0)
            log2_wt = tf.math.log(wt_preds_per_seq + offsets) / tf.math.log(2.0)
            differences = tf.reshape(log2_mut - log2_wt, [-1])  # Back to shape: (N * L * A,)
        else:
            differences = full_mut_preds - wt_preds_for_mutations  # Shape: (N * L * A,)

        # Reshape to final attribution maps
        attribution_maps = tf.reshape(differences, [N, window_length, A])

        # Create and apply wild-type mask
        X_expanded = tf.repeat(X, mutations_per_seq, axis=0)
        X_expanded = tf.reshape(X_expanded, [N, mutations_per_seq, L, A])
        wt_mask = tf.reduce_all(tf.equal(tf.reshape(all_mutations, [N, mutations_per_seq, L, A]), X_expanded), axis=[2, 3])
        wt_mask = tf.reshape(wt_mask, [N, window_length, A])

        # Zero out positions where mutation matches wild-type
        attribution_maps = tf.where(wt_mask, tf.zeros_like(attribution_maps), attribution_maps)

        # If using SNV window, pad with zeros
        if snv_window is not None:
            full_attribution_maps = tf.zeros([N, L, A], dtype=tf.float32)
            full_attribution_maps = tf.tensor_scatter_nd_update(
                full_attribution_maps,
                tf.stack([tf.repeat(tf.range(N), window_length * A),
                         tf.tile(tf.repeat(tf.range(start, end), A), [N]),
                         tf.tile(tf.tile(tf.range(A), [window_length]), [N])], axis=1),
                tf.reshape(attribution_maps, [-1])
            )
            attribution_maps = full_attribution_maps

        return attribution_maps.numpy()
    

    def _function_batch(self, X, func, batch_size, **kwargs):
        """Run computation in batches."""
        dataset = tf.data.Dataset.from_tensor_slices(X)
        outputs = []
        for x in dataset.batch(batch_size):
            outputs.append(func(x, **kwargs))
        return np.concatenate(outputs, axis=0)

    @staticmethod
    def _random_shuffle(x):
        """Randomly shuffle sequence using appropriate backend."""
        if isinstance(x, tf.Tensor):
            # GPU case - use TensorFlow ops
            seq_len = tf.shape(x)[1]
            shuffle = tf.random.shuffle(tf.range(seq_len))
            return tf.gather(x, shuffle, axis=1)
        else:
            # CPU case - use NumPy ops
            shuffle = np.random.permutation(x.shape[1])
            return x[:, shuffle, :]

    @staticmethod
    def _generate_background_data(x, num_shuffles):
        """Generate background data for DeepSHAP."""
        seq = x[0]
        shuffled = np.array([
            Attributer._random_shuffle(seq)
            for _ in range(num_shuffles)
        ])
        return [shuffled]
    
    def compute(self, x, x_ref=None, batch_size=128, save_window=None, **kwargs):
        """Compute attribution maps.
        
        Args:
            x: One-hot sequences (shape: (N, L, A))
            x_ref: One-hot reference sequence (shape: (1, L, A)) for windowed analysis.
                Not used for DeepSHAP background data, which is handled during initialization.
            batch_size: Number of attribution maps per batch (ignored for DeepSHAP)
            save_window: Window [start, stop] for computing attributions. If provided along with x_ref,
                        the input sequences will be padded with the reference sequence outside this window.
                        This allows computing attributions for a subset of positions while maintaining
                        the full sequence context.
            **kwargs: Additional arguments for specific attribution methods
                - gpu: Whether to use GPU implementation (default: True)
                - log2FC (bool): Whether to compute log2 fold change (for ISM)
                - num_steps: Steps for integrated gradients (default: 50)
                - num_samples: Samples for smoothgrad (default: 50)
                - mean, stddev: Parameters for smoothgrad noise
                - multiply_by_inputs: Whether to multiply gradients by inputs (default: False)
                - baseline_type: Background type for intgrad and deepshap ('zeros', 'random_shuffle', 'dinuc_shuffle')
                - background: Background sequences for DeepSHAP (shape: (N, L, A)) - overrides baseline_type
                - snv_window: Window [start, end] for ISM to compute variants (default: None)
        
        Returns:
            numpy.ndarray: Attribution maps (shape: (N, L, A))
        """
        # Ensure model is in evaluation mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
            
        if x_ref is not None:
            x_ref = x_ref.astype('uint8')
            if x_ref.ndim == 2:
                x_ref = x_ref[np.newaxis, :]

        # DeepSHAP processes all sequences at once (no batching)
        if self.method == 'deepshap':
            attribution_values = self._process_batch(x, x_ref, save_window, batch_size, **kwargs)
            self.attributions = attribution_values
            return attribution_values

        # For other methods, use batch processing
        N, L, A = x.shape
        num_batches = int(np.floor(N/batch_size))
        attribution_values = []

        # Process full batches
        for i in tqdm(range(num_batches), desc="Attribution"):
            x_batch = x[i*batch_size:(i+1)*batch_size]
            batch_values = self._process_batch(x_batch, x_ref, save_window, batch_size, **kwargs)
            attribution_values.append(batch_values)

        # Process remaining samples
        if num_batches*batch_size < N:
            x_batch = x[num_batches*batch_size:]
            batch_values = self._process_batch(x_batch, x_ref, save_window, batch_size, **kwargs)
            attribution_values.append(batch_values)

        attribution_values = np.vstack(attribution_values)
        self.attributions = attribution_values
        return attribution_values

    def _process_batch(self, x_batch, x_ref=None, save_window=None, batch_size=128, **kwargs):
        """Process a single batch of inputs."""
        if save_window is not None and x_ref is not None:
            x_batch = self._apply_save_window(x_batch, x_ref, save_window)

        if self.method == 'deepshap':
            # NOTE: DeepSHAP processes sequences one at a time, not in batches
            # The TFDeepExplainer loops through each sequence individually
            # Initialize explainer if not already done
            if not hasattr(self, 'explainer'):
                # Get background type from kwargs (default to 'dinuc_shuffle' for DeepSHAP)
                baseline_type = kwargs.get('baseline_type', 'dinuc_shuffle')
                
                # Generate background data based on baseline_type
                if baseline_type == 'dinuc_shuffle':
                    background = self._shuffle_several_times
                elif baseline_type == 'random_shuffle':
                    background = self._random_shuffle
                elif baseline_type == 'zeros':
                    # For zeros baseline, create a function that returns zero sequences
                    def zeros_background(s):
                        if len(s) == 2:
                            return [np.zeros_like(s[0]), s[1]]
                        else:
                            return [np.zeros_like(s[0])]
                    background = zeros_background
                else:
                    raise ValueError(f"Unsupported baseline_type '{baseline_type}' for DeepSHAP. "
                                   f"Supported types: 'dinuc_shuffle', 'random_shuffle', 'zeros'")
                
                # Allow override with custom background function
                custom_background = kwargs.get('background', None)
                if custom_background is not None:
                    background = custom_background
                
                # DeepSHAP expects a list for background
                if not isinstance(background, list):
                    background = [background]
                
                # For DeepSHAP, use the provided compress_fun to create target output
                target_output = self.compress_fun(self.model)
                
                # Always use TFDeepExplainer with combine_mult_and_diffref for ChromBPNet compatibility
                self.explainer = shap.explainers.deep.TFDeepExplainer(
                    (self.model.input, target_output),
                    background,
                    combine_mult_and_diffref=self._combine_mult_and_diffref
                )
            
            batch_values = self.explainer.shap_values(x_batch, progress_message=100)
        elif self.method == 'saliency':
            batch_values = self.saliency(x_batch, batch_size=batch_size)
        elif self.method == 'smoothgrad':
            gpu = kwargs.get('gpu', self.gpu)
            batch_values = self.smoothgrad(
                x_batch,
                num_samples=kwargs.get('num_samples', 20),
                mean=kwargs.get('mean', 0.0),
                stddev=kwargs.get('stddev', 0.1),
                gpu=gpu
            )
        elif self.method == 'intgrad':
            gpu = kwargs.get('gpu', self.gpu)  # Use instance default if not specified
            multiply_by_inputs = kwargs.get('multiply_by_inputs', False)
            baseline_type = kwargs.get('baseline_type', 'zeros')  # Get from kwargs with default
            batch_values = self.intgrad(
                x_batch, 
                baseline_type=baseline_type,
                num_steps=kwargs.get('num_steps', 20),
                gpu=gpu,
                multiply_by_inputs=multiply_by_inputs,
                seed=kwargs.get('seed', None)
            )
        elif self.method == 'ism':
            gpu = kwargs.get('gpu', self.gpu)
            log2fc = kwargs.get('log2fc', False)
            snv_window = kwargs.get('snv_window', None)
            batch_values = self.ism(x_batch, gpu=gpu, log2fc=log2fc, snv_window=snv_window)

        return batch_values

    def _apply_save_window(self, x_batch, x_ref, save_window):
        """Apply save window to batch using reference sequence.
        
        This function pads the input sequences with the reference sequence outside
        the specified window, allowing attribution computation on a subset of positions
        while maintaining the full sequence context.
        
        Args:
            x_batch: Input sequences of shape (batch_size, L, A)
            x_ref: Reference sequence of shape (1, L, A)
            save_window: [start, stop] positions defining the window
        
        Returns:
            Padded sequences of shape (batch_size, L, A)
        """
        start, stop = save_window
        
        # Validate window boundaries
        if start < 0 or stop > x_ref.shape[1] or start >= stop:
            raise ValueError(f"Invalid save_window [{start}, {stop}]. Must be within [0, {x_ref.shape[1]}] and start < stop")
        
        # Validate shapes
        if x_batch.shape[1] != (stop - start) or x_batch.shape[2] != x_ref.shape[2]:
            raise ValueError(f"Input shape {x_batch.shape} incompatible with window size {stop-start} and reference shape {x_ref.shape}")
        
        x_ref_start = np.broadcast_to(
            x_ref[:, :start, :],
            (x_batch.shape[0], start, x_ref.shape[2])
        )
        x_ref_stop = np.broadcast_to(
            x_ref[:, stop:, :],
            (x_batch.shape[0], x_ref.shape[1]-stop, x_ref.shape[2])
        )
        return np.concatenate([x_ref_start, x_batch, x_ref_stop], axis=1)


    def show_params(self, method=None):
        """Show available parameters for attribution methods.
        
        Args:
            method: Specific method to show params for. If None, shows all methods.
        """
        params = {
            'saliency': {
                'gpu': 'bool, Whether to use GPU acceleration (default: True)',
                'batch_size': 'int, Batch size for processing (default: 128)',
                'compress_fun': ('callable, Function to compress model output to scalar (default: tf.math.reduce_mean). '
                        'Required if model output is not already a scalar.')
            },
            'smoothgrad': {
                'gpu': 'bool, Whether to use GPU acceleration (default: True)',
                'num_samples': 'int, Number of noise samples (default: 50)',
                'mean': 'float, Mean of noise distribution (default: 0.0)',
                'stddev': 'float, Standard deviation of noise (default: 0.1)',
                'batch_size': 'int, Batch size for processing (default: 64)',
                'compress_fun': ('callable, Function to compress model output to scalar (default: tf.math.reduce_mean). '
                        'Required if model output is not already a scalar.')
            },
            'intgrad': {
                'gpu': 'bool, Whether to use GPU acceleration (default: True)',
                'num_steps': 'int, Number of integration steps (default: 50)',
                'multiply_by_inputs': 'bool, Whether to multiply gradients by inputs (default: False)',
                'baseline_type': ('str, Type of baseline to use (default: zeros). Options:\n'
                               '    - zeros: Zero baseline\n'
                               '    - random_shuffle: Random shuffle of input sequence\n'
                               '    - dinuc_shuffle: Dinucleotide-preserved shuffle'),
                'seed': 'int, Random seed for reproducibility in shuffling methods (optional)',
                'batch_size': 'int, Batch size for processing (default: 128)',
                'compress_fun': ('callable, Function to compress model output to scalar (default: tf.math.reduce_mean). '
                        'Required if model output is not already a scalar.')
            },
            'deepshap': {
                'background': ('array, Background sequences for DeepSHAP (optional). Shape: (N, L, A). '
                            'If not provided, will generate shuffled backgrounds using num_shuffles.')
            },
            'ism': {
                'gpu': 'bool, Whether to use GPU acceleration (default: True)',
                'log2FC': 'bool, Whether to compute log2 fold change (default: False)',
                'batch_size': 'int, Batch size for processing (default: 128)',
                'compress_fun': ('callable, Function to compress model output to scalar (default: tf.math.reduce_mean). '
                        'Required if model output is not already a scalar.')
            }
        }
        
        common_params = {
            'x_ref': ('array, Reference sequence for comparison (optional). Shape: (1, L, A). '
                    'Used for padding in windowed analysis when save_window is specified. '
                    'Not used for DeepSHAP background.'),
            'save_window': ('list, Window [start, end] to compute attributions (optional). '
                        'When provided with x_ref, allows computing attributions for a subset of positions '
                        'while maintaining full sequence context. Input x should contain only the windowed region '
                        'with shape (N, end-start, A), and x_ref provides the full-length context with '
                        'shape (1, L, A). Example: [100, 200] computes attributions for positions 100-200.')
        }
        
        if method is not None:
            if method not in self.SUPPORTED_METHODS:
                print(f"Method '{method}' not supported. Available methods: {self.SUPPORTED_METHODS}")
                return
            
            print(f"\nParameters for {method}:")
            print("\nRequired:")
            print("x: array, Input sequences to compute attributions for")
            print("\nOptional:")
            for param, desc in params[method].items():
                print(f"{param}: {desc}")
            print("\nCommon Optional:")
            for param, desc in common_params.items():
                print(f"{param}: {desc}")
        else:
            for method in self.SUPPORTED_METHODS:
                print(f"\nParameters for {method}:")
                print("\nRequired:")
                print("x: array, Input sequences to compute attributions for")
                print("\nOptional:")
                for param, desc in params[method].items():
                    print(f"{param}: {desc}")
                print("\nCommon Optional:")
                for param, desc in common_params.items():
                    print(f"{param}: {desc}")
                print("\n" + "-"*50)

    @staticmethod
    def _check_deepshap_dependencies():
        """Check if all dependencies for ChromBPNet DeepSHAP are available.
        
        Returns:
            shap module if all dependencies are available
            
        Raises:
            ImportError: If any required dependency is missing
        """
        if not HAS_SHAP:
            error_msg = "Missing required dependency for ChromBPNet DeepSHAP: kundajelab-shap\n\n"
            error_msg += "Install missing package with:\n"
            error_msg += "  pip install kundajelab-shap==1\n"
            error_msg += "Note: ChromBPNet DeepSHAP requires the Kundaje SHAP package."
            
            raise ImportError(error_msg)
        
        return shap

    @staticmethod
    def _combine_mult_and_diffref(mult, orig_inp, bg_data):
        """Combine multipliers and difference from reference for DeepSHAP.
        
        This function implements the ChromBPNet-specific multiplier combination
        that projects hypothetical contributions onto the actual sequence.
        
        Args:
            mult: Multipliers from DeepSHAP
            orig_inp: Original input sequence
            bg_data: Background data
            
        Returns:
            list: Combined attribution scores
        """
        to_return = []
        
        for l in [0]:
            projected_hypothetical_contribs = \
                np.zeros_like(bg_data[l]).astype("float")
            assert len(orig_inp[l].shape)==2
            
            # At each position in the input sequence, we iterate over the
            # one-hot encoding possibilities (eg: for genomic sequence, 
            # this is ACGT i.e. 1000, 0100, 0010 and 0001) and compute the
            # hypothetical difference-from-reference in each case. We then 
            # multiply the hypothetical differences-from-reference with 
            # the multipliers to get the hypothetical contributions. For 
            # each of the one-hot encoding possibilities, the hypothetical
            # contributions are then summed across the ACGT axis to 
            # estimate the total hypothetical contribution of each 
            # position. This per-position hypothetical contribution is then
            # assigned ("projected") onto whichever base was present in the
            # hypothetical sequence. The reason this is a fast estimate of
            # what the importance scores *would* look like if different 
            # bases were present in the underlying sequence is that the
            # multipliers are computed once using the original sequence, 
            # and are not computed again for each hypothetical sequence.
            for i in range(orig_inp[l].shape[-1]):
                hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
                hypothetical_input[:, i] = 1.0
                hypothetical_difference_from_reference = \
                    (hypothetical_input[None, :, :] - bg_data[l])
                hypothetical_contribs = hypothetical_difference_from_reference * \
                                        mult[l]
                projected_hypothetical_contribs[:, :, i] = \
                    np.sum(hypothetical_contribs, axis=-1) 
                
            to_return.append(np.mean(projected_hypothetical_contribs,axis=0))

        if len(orig_inp)>1:
            to_return.append(np.zeros_like(orig_inp[1]))
        
        return to_return

    @staticmethod
    def _shuffle_several_times(s, numshuffles=20):
        """Generate multiple dinucleotide shuffles for DeepSHAP background.
        
        This implements the exact same approach as the ChromBPNet repository.
        
        Args:
            s: Input sequence
            numshuffles: Number of shuffles to generate (default: 20, matches ChromBPNet)
            
        Returns:
            list: List of shuffled sequences
        """
        try:
            # Try to use the deeplift dinuc_shuffle function (same as ChromBPNet)
            from deeplift.dinuc_shuffle import dinuc_shuffle
            
            if len(s)==2:
                return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)]),
                        np.array([s[1] for i in range(numshuffles)])]
            else:
                return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)])]
        except ImportError:
            # Fallback to random shuffle if deeplift is not available
            print("WARNING: deeplift package not found. Dinucleotide shuffling unavailable.")
            print("Falling back to random shuffling for DeepSHAP background generation.")
            print("For optimal results with genomic sequences, install deeplift: pip install deeplift")
            
            if len(s)==2:
                return [np.array([Attributer._random_shuffle(s[0]) for i in range(numshuffles)]),
                        np.array([s[1] for i in range(numshuffles)])]
            else:
                return [np.array([Attributer._random_shuffle(s[0]) for i in range(numshuffles)])]

    @staticmethod
    def bpnet_profile(x):
        """ChromBPNet profile compression function.
        
        This function implements the ChromBPNet profile task compression.
        For DeepSHAP, x should be the model. For other methods, x should be the output tensor.
        
        Args:
            x: Model output tensor (profile logits) or model (for DeepSHAP)
            
        Returns:
            tf.Tensor: Weighted sum of mean-normalized logits
        """
        # Check if x is a model (has outputs attribute) or a tensor
        if hasattr(x, 'outputs'):
            # x is a model - create target output tensor for DeepSHAP
            meannormed_logits = (x.outputs[0] - 
                               tf.reduce_mean(x.outputs[0], axis=1, keepdims=True))
            
            # 'stop_gradient' will prevent importance from being propagated
            # through this operation; we do this because we just want to treat
            # the post-softmax probabilities as 'weights' on the different 
            # logits, without having the network explain how the probabilities
            # themselves were derived. Could be worth contrasting explanations
            # derived with and without stop_gradient enabled
            stopgrad_meannormed_logits = tf.stop_gradient(meannormed_logits)
            softmax_out = tf.nn.softmax(stopgrad_meannormed_logits, axis=1)
        else:
            # x is the output tensor - compress it (existing behavior)
            meannormed_logits = x - tf.reduce_mean(x, axis=1, keepdims=True)
            
            # Apply softmax to meannormed logits (without stop_gradient for non-DeepSHAP methods)
            softmax_out = tf.nn.softmax(meannormed_logits, axis=1)
        
        # Weight the logits according to softmax probabilities and sum
        weightedsum = tf.reduce_sum(softmax_out * meannormed_logits, axis=1)
        return weightedsum

    @staticmethod
    def bpnet_counts(x):
        """ChromBPNet counts compression function.
        
        This function implements the ChromBPNet counts task compression.
        For DeepSHAP, x should be the model. For other methods, x should be the output tensor.
        
        Args:
            x: Model output tensor (counts logits) or model (for DeepSHAP)
            
        Returns:
            tf.Tensor: For DeepSHAP: sum of counts across output dimension
                     For other methods: tensor as-is (no reduction)
        """
        # Check if x is a model (has outputs attribute) or a tensor
        if hasattr(x, 'outputs'):
            # x is a model - create target output tensor for DeepSHAP
            # For DeepSHAP, we need to reduce to scalar to avoid list output
            return tf.reduce_sum(x.outputs[1], axis=-1)
        else:
            # x is the output tensor - return as-is for other methods (no reduction)
            return x