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

def _check_shap_available():
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP is required for this functionality. "
            "Install it with: pip install shap"
        )
    return shap

class Attributer:
    """
    Attributer: A unified interface for computing attribution maps in TensorFlow 2.x

    This implementation is optimized for TensorFlow 2.x (tested on 2.17.1) and provides
    GPU-accelerated implementations of common attribution methods:
    - Saliency Maps
    - SmoothGrad
    - Integrated Gradients
    - DeepSHAP (via SHAP package)
    - ISM (In-Silico Mutagenesis)

    Requirements:
    - tensorflow >= 2.10.0
    - numpy
    - tqdm
    - shap (for DeepSHAP only)

    Key Features:
    - Batch processing for all methods
    - GPU-optimized implementations for saliency, smoothgrad, and integrated gradients
    - Consistent interface across methods
    - Support for multi-head models
    - Memory-efficient processing of large datasets
    - Flexible sequence windowing for long sequences

    Performance Notes:
        Benchmarks on 10,000 sequences (249bp) using NVIDIA A100-SXM4-40GB:
        - Saliency: ~2.4s
        - IntGrad (GPU): ~4.9s (50 steps)
        - SmoothGrad (GPU): ~5.8s (50 samples)
        - ISM (GPU): ~44.6s
        
        GPU acceleration provides significant speedup for IntGrad and SmoothGrad.
        Batch processing is optimized for saliency, intgrad, and smoothgrad methods.
        
        Hardware used for benchmarks:
        - GPU: NVIDIA A100-SXM4-40GB
        - Compute Capability: 8.0
        - TensorFlow: 2.17.1

    Example usage:
        # Basic usage with output reduction function
        attributer = Attributer(
            model, 
            method='saliency',
            task_index=0,
            func=lambda x: tf.reduce_mean(x[:, :, 1])  # Example: mean of second output channel
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
            num_steps=50,          # for intgrad
            num_samples=50,        # for smoothgrad
            multiply_by_inputs=False  # for intgrad
            log2fc=False  # for ism
        )

    Note: For optimal performance, ensure TensorFlow is configured to use GPU acceleration.
    """
    
    SUPPORTED_METHODS = {'saliency', 'smoothgrad', 'intgrad', 'deepshap', 'ism'}

    # Define default batch sizes for each method
    DEFAULT_BATCH_SIZES = {
        'saliency': 128,
        'intgrad': 128,
        'smoothgrad': 64,
        'deepshap': 1,    # not optimized for batch mode
        'ism': 32         # not optimized for batch mode
    }
    
    def __init__(self, model, method='saliency', task_index=None, out_layer=-1, 
                batch_size=None, num_shuffles=100, func=tf.math.reduce_mean, gpu=True):
        """Initialize the Attributer.
        
        Args:
            model: TensorFlow model to explain
            method: Attribution method (default: 'saliency')
            task_index: Index of output head to explain (optional)
            out_layer: Output layer index for DeepSHAP
            batch_size: Batch size for computing attributions (optional, defaults to method-specific size)
            num_shuffles: Number of shuffles for DeepSHAP background
            func: Function to apply to model output (default: tf.math.reduce_mean)
            gpu: Whether to use GPU-optimized implementation (default: True)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
            
        self.model = model
        self.method = method
        self.task_index = task_index
        self.func = func
        self.out_layer = out_layer
        self.gpu = gpu
        self.num_shuffles = num_shuffles

        # Set batch size based on method if not specified
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZES[method]

        if self.batch_size > 1 and method == 'deepshap':  # removed ISM from this check
            print(f"Warning: {method} is not optimized for batch mode. Using batch_size=1")
            self.batch_size = 1

        if method == 'shap':
            self.shap = _check_shap_available()

    @tf.function
    def _saliency_map(self, X):
        """Compute saliency maps."""
        if not tf.is_tensor(X):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
        else:
            X = tf.cast(X, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X)
            if self.task_index is not None:
                outputs = self.model(X)[self.task_index]
            else:
                outputs = self.func(self.model(X))
        return tape.gradient(outputs, X)

    def saliency(self, X, batch_size=None):
        """Compute saliency maps in batches."""
        return self._function_batch(X, self._saliency_map, 
                                  batch_size or self.batch_size)

    def smoothgrad(self, X, num_samples=50, mean=0.0, stddev=0.1, gpu=True, **kwargs):
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

    def _smoothgrad_cpu(self, X, num_samples=50, mean=0.0, stddev=0.1):
        """CPU implementation of SmoothGrad."""
        scores = []
        for x in X:
            x = np.expand_dims(x, axis=0)  # (1, L, A)
            x = tf.cast(x, dtype=tf.float32)
            x_noisy = tf.tile(x, (num_samples,1,1)) + tf.random.normal((num_samples,x.shape[1],x.shape[2]), mean, stddev)
            grad = self._saliency_map(x_noisy)
            scores.append(tf.reduce_mean(grad, axis=0))
        return np.stack(scores, axis=0)

    @tf.function(jit_compile=True)
    def _smoothgrad_gpu(self, X, num_samples=50, mean=0.0, stddev=0.1):
        """GPU-optimized implementation with parallel noise generation."""
        X = tf.cast(X, dtype=tf.float32)
        
        # Generate all noise samples in parallel
        noise_shape = [tf.shape(X)[0] * num_samples, tf.shape(X)[1], tf.shape(X)[2]]
        X_noisy = tf.tile(X, [num_samples, 1, 1]) + tf.random.normal(noise_shape, mean, stddev)
        
        # Single gradient call on all samples
        grads = self._saliency_map(X_noisy)
        
        # Reshape and reduce
        grads = tf.reshape(grads, [-1, num_samples, tf.shape(X)[1], tf.shape(X)[2]])
        return tf.reduce_mean(grads, axis=1)
    
    def intgrad(self, X, baseline_type='zeros', num_steps=25, gpu=True, multiply_by_inputs=False, seed=None):
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
    
    def _intgrad_cpu(self, X, baseline_type='zeros', num_steps=25, multiply_by_inputs=False, seed=None):
        """CPU-optimized implementation using loop-based computation."""
        scores = []
        for i, x in enumerate(X):
            x = np.expand_dims(x, axis=0)  # Add batch dimension: (1, L, A)
            
            # Explicitly handle each baseline type
            if baseline_type == 'zeros':
                baseline = np.zeros_like(x)
            elif baseline_type == 'random_shuffle':
                # Set random seed for reproducibility if provided
                if seed is not None:
                    np.random.seed(seed)
                baseline = self._random_shuffle(x)
            elif baseline_type == 'dinuc_shuffle':
                if i == 0:  # Only compute all shuffles once at the start
                    baselines = self._batch_dinuc_shuffle(X, num_shuffles=1, seed=seed)
                baseline = np.expand_dims(baselines[i], axis=0)
            else:
                raise ValueError("baseline_type must be one of: 'zeros', 'random_shuffle', 'dinuc_shuffle'")
            
            score = self._integrated_grad(x, baseline, num_steps, multiply_by_inputs)
            scores.append(score[0])  # Remove batch dimension before appending
        return np.stack(scores, axis=0)  # Stack to get (N, L, A)

    @tf.function(jit_compile=True)
    def _intgrad_gpu(self, X, baseline_type='zeros', num_steps=25, multiply_by_inputs=False, seed=None):
        """GPU-optimized implementation using vectorized computation."""
        # Ensure input is float32
        X = tf.cast(X, tf.float32)
        
        # Explicitly handle each baseline type
        if baseline_type == 'zeros':
            baseline = tf.zeros_like(X, dtype=tf.float32)
        elif baseline_type == 'random_shuffle':
            # Set random seed for reproducibility if provided
            if seed is not None:
                tf.random.set_seed(seed)
            baseline = tf.cast(tf.vectorized_map(self._random_shuffle, X), tf.float32)
        elif baseline_type == 'dinuc_shuffle':
            # Pre-compute all shuffles at once for efficiency
            baseline = tf.cast(self._batch_dinuc_shuffle(X, num_shuffles=1, seed=seed), tf.float32)
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
    
    def ism(self, X, log2fc=False, gpu=True):
        """Compute In-Silico Mutagenesis attribution maps.
        
        Args:
            X: Input tensor of shape (batch_size, L, A)
            log2fc: Whether to compute log2 fold change instead of difference
            gpu: Whether to attempt GPU-optimized implementation
        
        Returns:
            numpy.ndarray: Attribution maps of shape (batch_size, L, A)
        """
        try:
            if gpu:
                try:
                    return self._ism_gpu(X, log2fc)
                except Exception as e:
                    print(f"GPU implementation failed with error: {str(e)}")
                    print("Falling back to CPU")
        except:
            print("GPU implementation failed, falling back to CPU")
        return self._ism_cpu(X, log2fc)

    def _ism_cpu(self, X, log2fc=False):
        """CPU implementation of ISM."""
        X = X.astype(np.float32)  # Ensure float32
        scores = []
        
        # Pre-allocate mutation array for reuse
        mut_seq = np.zeros_like(X[0:1], dtype=np.float32)
        
        for x in X:
            x = x[np.newaxis]  # Faster than expand_dims
            score_matrix = np.zeros_like(x[0], dtype=np.float32)
            
            # Get wild-type prediction
            wt_pred = self.func(self.model(tf.constant(x, dtype=tf.float32)))
            
            # Store all mutation predictions first
            mut_preds = np.empty((x.shape[1] * (x.shape[2]-1)), dtype=np.float32)
            mut_locs = []
            
            # Reuse pre-allocated array
            mut_seq[:] = x
            
            idx = 0
            for pos in range(x.shape[1]):
                for b in range(1, x.shape[2]):
                    mut_seq[0, pos] = np.roll(x[0, pos], b)
                    new_base = np.where(mut_seq[0, pos] == 1)[0][0]
                    
                    mut_preds[idx] = float(self.func(self.model(tf.constant(mut_seq))))
                    mut_locs.append((pos, new_base))
                    idx += 1
                
                # Restore original sequence for next position
                mut_seq[0, pos] = x[0, pos]
            
            # Apply log2fc calculation after collecting all predictions
            if log2fc:
                pred_min = min(min(mut_preds), float(wt_pred))
                offset = abs(pred_min) + 1
                wt_pred_adj = wt_pred + offset
                
                for (pos, new_base), mut_pred in zip(mut_locs, mut_preds):
                    if mut_pred != wt_pred:
                        score_matrix[pos, new_base] = np.log2(mut_pred + offset) - np.log2(wt_pred_adj)
                    else:
                        score_matrix[pos, new_base] = 0.
            else:
                for (pos, new_base), mut_pred in zip(mut_locs, mut_preds):
                    score_matrix[pos, new_base] = mut_pred - wt_pred
                        
            scores.append(score_matrix)
        return np.stack(scores, axis=0)

    @tf.function(jit_compile=True)
    def _ism_gpu(self, X, log2fc=False):
        """GPU-optimized implementation of ISM using vectorized operations."""
        X = tf.cast(X, tf.float32)
        batch_size, L, A = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
        
        # Get wild-type predictions
        wt_preds = self.func(self.model(X))
        
        # Initialize output tensor
        scores = tf.zeros((batch_size, L, A), dtype=tf.float32)
        
        if log2fc:
            # Store all mutation predictions to find global minimum
            all_preds = [wt_preds]
        
        # For each position
        for pos in tf.range(L, dtype=tf.int64):
            # Get reference bases for this position
            ref_bases = tf.cast(tf.argmax(X[:, pos], axis=-1), tf.int64)
            
            # For each base (using full range and masking instead of range(1,A))
            for alt_base in tf.range(A, dtype=tf.int64):
                # Only proceed if this is not the reference base
                should_mutate = tf.not_equal(ref_bases, alt_base)
                
                if tf.reduce_any(should_mutate):
                    # Create mutated sequences
                    X_mut = tf.identity(X)
                    
                    # Create mutation mask
                    mut_mask = tf.one_hot(alt_base, A, dtype=tf.float32)
                    mut_mask = tf.tile(mut_mask[None, None, :], [batch_size, 1, 1])
                    
                    # Apply mutations where needed
                    X_mut = tf.tensor_scatter_nd_update(
                        X_mut,
                        tf.stack([
                            tf.cast(tf.range(batch_size), tf.int64),
                            tf.fill([batch_size], pos)
                        ], axis=1),
                        tf.where(
                            should_mutate[:, None],
                            tf.tile(mut_mask[:, 0, :], [1, 1]),
                            X[:, pos]
                        )
                    )
                    
                    # Get predictions
                    mut_preds = self.func(self.model(X_mut))
                    
                    if log2fc:
                        # Store predictions for finding global minimum
                        all_preds.append(mut_preds)
                        
                        # Store mutation locations and predictions for later
                        scores = tf.tensor_scatter_nd_update(
                            scores,
                            tf.stack([
                                tf.cast(tf.range(batch_size), tf.int64),
                                tf.fill([batch_size], pos),
                                tf.fill([batch_size], alt_base)
                            ], axis=1),
                            tf.where(
                                should_mutate,
                                mut_preds,  # Store raw predictions temporarily
                                tf.zeros_like(mut_preds)
                            )
                        )
                    else:
                        # For non-log2fc, just compute differences directly
                        scores = tf.tensor_scatter_nd_update(
                            scores,
                            tf.stack([
                                tf.cast(tf.range(batch_size), tf.int64),
                                tf.fill([batch_size], pos),
                                tf.fill([batch_size], alt_base)
                            ], axis=1),
                            tf.where(
                                should_mutate,
                                mut_preds - wt_preds,
                                tf.zeros_like(mut_preds)
                            )
                        )
        
        if log2fc:
            # Find global minimum across all predictions
            all_preds = tf.stack(all_preds, axis=1)  # [batch_size, num_preds]
            pred_min = tf.reduce_min(all_preds, axis=1)  # [batch_size]
            offset = tf.abs(pred_min) + 1
            
            # Add offset to wild-type predictions
            wt_preds_adj = wt_preds + offset[:, tf.newaxis]
            
            # Process each position and base
            for pos in tf.range(L, dtype=tf.int64):
                for alt_base in tf.range(A, dtype=tf.int64):
                    mut_preds = scores[:, pos, alt_base]  # Get stored raw predictions
                    should_mutate = tf.not_equal(mut_preds, 0.0)  # Where we stored predictions
                    
                    if tf.reduce_any(should_mutate):
                        # Add offset and compute log2fc
                        mut_preds_adj = mut_preds + offset[:, tf.newaxis]
                        delta = tf.where(
                            should_mutate,
                            tf.math.log(mut_preds_adj) - tf.math.log(wt_preds_adj),
                            tf.zeros_like(mut_preds)
                        )
                        
                        # Update scores with log2fc values
                        scores = tf.tensor_scatter_nd_update(
                            scores,
                            tf.stack([
                                tf.cast(tf.range(batch_size), tf.int64),
                                tf.fill([batch_size], pos),
                                tf.fill([batch_size], alt_base)
                            ], axis=1),
                            delta
                        )
        
        return scores

    def _function_batch(self, X, func, batch_size, **kwargs):
        """Run computation in batches."""
        dataset = tf.data.Dataset.from_tensor_slices(X)
        outputs = []
        for x in dataset.batch(batch_size):
            outputs.append(func(x, **kwargs))
        return np.concatenate(outputs, axis=0)
    
    def _set_baseline(self, x, baseline_type, seed=None):
        """Set baseline for Integrated Gradients.
        
        Parameters
        ----------
        x : array-like
            Input sequence to generate baseline for
        baseline_type : str
            Type of baseline to use:
            - 'zeros': Zero baseline
            - 'random_shuffle': Random shuffle of input sequence
            - 'dinuc_shuffle': Dinucleotide-preserved shuffle of input sequence (default)
        num_backgrounds : int
            Number of background sequences to generate for shuffling methods
        seed : int, optional
            Random seed for reproducibility in shuffling methods
        
        Returns
        -------
        array-like
            Baseline sequence(s)
        """
        if baseline_type == 'zeros':
            return np.zeros_like(x)
        elif baseline_type == 'random_shuffle':
            return self._random_shuffle(x, seed=seed)
        elif baseline_type == 'dinuc_shuffle':
            return self._batch_dinuc_shuffle(x, num_shuffles=1, seed=seed)[0]
        else:
            raise ValueError("baseline_type must be one of: 'dinuc_shuffle', 'random_shuffle', 'zeros'")

    @staticmethod
    def _one_hot_to_tokens(one_hot):
        """Convert an L x D one-hot encoding into an L-vector of integers.
        
        Parameters
        ----------
        one_hot : np.ndarray
            One-hot encoded sequence with shape (length, D)
            
        Returns
        -------
        np.ndarray
            Vector of integers representing sequence
        """
        tokens = np.tile(one_hot.shape[1], one_hot.shape[0])
        seq_inds, dim_inds = np.where(one_hot)
        tokens[seq_inds] = dim_inds
        return tokens

    @staticmethod
    def _tokens_to_one_hot(tokens, one_hot_dim):
        """Convert an L-vector of integers to an L x D one-hot encoding.
        
        Parameters
        ----------
        tokens : np.ndarray
            Vector of integers representing sequence
        one_hot_dim : int
            Dimension of one-hot encoding (usually 4 for DNA)
            
        Returns
        -------
        np.ndarray
            One-hot encoded sequence
        """
        identity = np.identity(one_hot_dim + 1)[:, :-1]
        return identity[tokens]

    @staticmethod
    def _dinuc_shuffle(seq, rng=None):
        """Create shuffle of sequence preserving dinucleotide frequencies.
        
        Parameters
        ----------
        seq : np.ndarray
            L x D one-hot encoded sequence
        rng : np.random.RandomState, optional
            Random number generator
            
        Returns
        -------
        np.ndarray
            Shuffled sequence preserving dinucleotide frequencies
        """
        if rng is None:
            rng = np.random.RandomState()
            
        # Convert to tokens
        tokens = Attributer._one_hot_to_tokens(seq)
        
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
        
        return Attributer._tokens_to_one_hot(chars[result], seq.shape[1])

    @staticmethod
    def _batch_dinuc_shuffle(
        sequence: Union[np.ndarray, tf.Tensor],
        num_shuffles: int = 1,
        seed: Optional[int] = None
    ) -> Union[np.ndarray, tf.Tensor]:
        """Generate multiple dinucleotide-preserved shuffles efficiently.
        
        Parameters
        ----------
        sequence : Union[np.ndarray, tf.Tensor]
            One-hot encoded sequence(s) with shape (length, 4), (1, length, 4),
            or (batch_size, length, 4)
        num_shuffles : int
            Number of shuffled sequences to generate per input sequence
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        Union[np.ndarray, tf.Tensor]
            Batch of shuffled sequences. If input is batched, shape will be
            (batch_size, length, 4). If num_shuffles > 1, shape will be
            (num_shuffles, length, 4) for single input or 
            (batch_size, num_shuffles, length, 4) for batched input.
        """
        # Convert to numpy if needed
        was_tensor = isinstance(sequence, tf.Tensor)
        if was_tensor:
            sequence = sequence.numpy()
        
        # Handle different input shapes
        if len(sequence.shape) == 2:  # (length, 4)
            sequence = sequence[np.newaxis, ...]  # Add batch dimension
            single_sequence = True
        elif len(sequence.shape) == 3:  # (batch_size, length, 4) or (1, length, 4)
            single_sequence = sequence.shape[0] == 1
        else:
            raise ValueError("Input sequence must have shape (length, 4), (1, length, 4), or (batch_size, length, 4)")
        
        # Create RandomState with seed
        rng = np.random.RandomState(seed)
        
        if single_sequence:
            # Generate shuffles for single sequence
            shuffled_sequences = [
                Attributer._dinuc_shuffle(sequence[0], rng=rng) 
                for _ in range(num_shuffles)
            ]
            shuffled = np.stack(shuffled_sequences, axis=0)
        else:
            # Generate one shuffle per sequence in batch
            if num_shuffles > 1:
                # Generate multiple shuffles per sequence
                shuffled_sequences = [[
                    Attributer._dinuc_shuffle(seq, rng=rng)
                    for _ in range(num_shuffles)
                ] for seq in sequence]
                shuffled = np.stack([np.stack(seq_shuffles, axis=0) 
                                   for seq_shuffles in shuffled_sequences], axis=0)
            else:
                # Generate one shuffle per sequence
                shuffled_sequences = [
                    Attributer._dinuc_shuffle(seq, rng=rng)
                    for seq in sequence
                ]
                shuffled = np.stack(shuffled_sequences, axis=0)
        
        # Convert back to tensor if input was tensor
        if was_tensor:
            shuffled = tf.convert_to_tensor(shuffled, dtype=tf.float32)
            
        return shuffled

    @staticmethod
    def _random_shuffle(x):
        """Randomly shuffle sequence."""
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
        """Compute attribution maps in batch mode.
        
        Args:
            x: One-hot sequences (shape: (N, L, A))
            x_ref: One-hot reference sequence (shape: (1, L, A)) for windowed analysis.
                Not used for DeepSHAP background data, which is handled during initialization.
            batch_size: Number of attribution maps per batch
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
                - background: Background sequences for DeepSHAP (shape: (N, L, A))
        
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
            # Initialize explainer if not already done
            if not hasattr(self, 'explainer'):
                shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
                background = kwargs.get('background', None)
                if background is None:
                    background = self._generate_background_data(x_batch, self.num_shuffles)
                else:
                    background = [background]  # DeepSHAP expects a list
                
                self.explainer = shap.DeepExplainer(
                    (self.model.layers[0].input, self.model.layers[self.out_layer].output),
                    data=background
                )
            batch_values = self.explainer.shap_values(x_batch)[0]
        elif self.method == 'saliency':
            batch_values = self.saliency(x_batch, batch_size=batch_size)
        elif self.method == 'smoothgrad':
            gpu = kwargs.get('gpu', self.gpu)
            batch_values = self.smoothgrad(
                x_batch,
                num_samples=kwargs.get('num_samples', 50),
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
                num_steps=kwargs.get('num_steps', 50),
                gpu=gpu,
                multiply_by_inputs=multiply_by_inputs,
                seed=kwargs.get('seed', None)
            )
        elif self.method == 'ism':
            gpu = kwargs.get('gpu', self.gpu)
            log2fc = kwargs.get('log2fc', False)
            batch_values = self.ism(x_batch, gpu=gpu, log2fc=log2fc)

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
                'func': ('callable, Function to reduce model output to scalar (default: tf.math.reduce_mean). '
                        'Required if model output is not already a scalar.')
            },
            'smoothgrad': {
                'gpu': 'bool, Whether to use GPU acceleration (default: True)',
                'num_samples': 'int, Number of noise samples (default: 50)',
                'mean': 'float, Mean of noise distribution (default: 0.0)',
                'stddev': 'float, Standard deviation of noise (default: 0.1)',
                'batch_size': 'int, Batch size for processing (default: 64)',
                'func': ('callable, Function to reduce model output to scalar (default: tf.math.reduce_mean). '
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
                'func': ('callable, Function to reduce model output to scalar (default: tf.math.reduce_mean). '
                        'Required if model output is not already a scalar.')
            },
            'deepshap': {
                'batch_size': 'int, Batch size for processing (default: 1)',
                'background': ('array, Background sequences for DeepSHAP (optional). Shape: (N, L, A). '
                            'If not provided, will generate shuffled backgrounds using num_shuffles.')
            },
            'ism': {
                'gpu': 'bool, Whether to use GPU acceleration (default: True)',
                'log2FC': 'bool, Whether to compute log2 fold change (default: False)',
                'batch_size': 'int, Batch size for processing (default: 128)',
                'func': ('callable, Function to reduce model output to scalar (default: tf.math.reduce_mean). '
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


# Convenience function
def compute_attributions(model, x, x_ref=None, method='saliency', func=tf.math.reduce_mean, **kwargs):
    """Compute attribution maps for a given model and input.
    
    Args:
        model: TensorFlow model to explain
        x: Input sequences to compute attributions for
        x_ref: Reference sequence for windowed analysis (optional)
        method: Attribution method (default: 'saliency')
        func: Function to reduce model output to scalar (default: tf.math.reduce_mean)
        **kwargs: Additional method-specific arguments
    """
    attributer = Attributer(model, method=method, func=func)
    return attributer.compute(x, x_ref=x_ref, **kwargs)