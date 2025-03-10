import numpy as np
import warnings
from core.explainer import Explainer
from .utils import standard_combine_mult_and_diffref
import tensorflow as tf
from .deep_tf2_utils import (
    op_handlers,
    nonlinearity_1d
)

# Add global variables for gradient debugging
last_layer_gradients = None
last_op_type = None
last_layer_name = None

class CustomGradientLayer(tf.keras.layers.Layer):
    def __init__(self, op_type, op_name, op, handler, explainer):
        super().__init__(name=f'custom_grad_{op_name}')
        self.layer = op
        print(f"\nCustomGradientLayer init for {op_name}")
        print(f"Layer input shape: {self.layer.input_shape}")
        print(f"Layer output shape: {self.layer.output_shape}")
        self.explainer = explainer
        self.handler = handler
        self.op_type = op_type
        self.internal_ops = self._trace_internal_ops()

    def call(self, inputs):
        print(f"\nCustomGradientLayer call")
        print(f"Inputs shape: {inputs.shape}")
        
        # Watch the inputs tensor
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self.layer(inputs)
            outputs = tf.identity(outputs)  # Ensure outputs remain connected
        
        print(f"Outputs shape: {outputs.shape}")
        
        # Return outputs and gradient function, passing layer name
        return custom_gradient_function(outputs, inputs, self.internal_ops)
    
    def _trace_internal_ops(self):
        """Trace internal operations of the layer."""
        # Handle input shape more carefully
        if isinstance(self.layer.input_shape, list):
            input_shape = tuple([1] + list(self.layer.input_shape[0][1:]))
        else:
            input_shape = tuple([1] + list(self.layer.input_shape[1:]))
            
        # Create sample input
        sample_input = tf.random.normal(input_shape)
        
        @tf.autograph.experimental.do_not_convert
        def trace_ops(inputs):
            with tf.GradientTape() as tape:
                outputs = self.layer(inputs)
            return outputs
            
        # Convert to tf.function and get concrete function
        traced_fn = tf.function(trace_ops)
        concrete_fn = traced_fn.get_concrete_function(sample_input)
        
        # Print ALL operations, including fused ones
        print("\nALL operations in layer:")
        for op in concrete_fn.graph.get_operations():
            print(f"- {op.type}: {op.name}")
            if hasattr(op, 'get_attr'):
                try:
                    fused_ops = op.get_attr('fused_ops')
                    print(f"  Fused ops: {fused_ops}")
                except:
                    pass
        
        # Filter operations but keep all BatchNorm and Identity ops
        ops = [
            op for op in concrete_fn.graph.get_operations() 
            if (op.type not in ['Placeholder', 'Const', 'NoOp', 'ReadVariableOp'] or
                'batch_normalization' in op.name or
                op.type == 'Identity')  # Keep Identity ops
            and any(isinstance(output, tf.Tensor) for output in op.outputs)
        ]
        
        # Store operation types as strings
        op_types = [str(op.type) for op in ops]  # Convert to strings
        
        if self.explainer.verbose:
            print(f"\nTraced operations for {self.layer.name}:")
            for op_type in op_types:
                print(f"- {op_type}")
        
        return op_types


@tf.custom_gradient
def custom_gradient_function(outputs, inputs, op_types):
    """Standalone function to apply custom gradients"""
    layer_name = tf.get_current_name_scope().split('/')[-1]
    print(f"\n\nNew CustomGradientLayer for ops: {op_types} in {layer_name}")

    def grad_fn(upstream, variables=None):
        global last_layer_gradients, last_op_type, last_layer_name
        
        print("\nGradient transition details:")
        print(f"Operation types: {op_types}")
        
        # Check layer transition if we have previous gradients
        #if last_layer_gradients is not None:
            #_check_gradient_change(last_layer_gradients, upstream, f"Layer transition to {layer_name}")
        
        # Create modified_grads
        modified_grads = upstream
        last_layer_gradients = modified_grads
        last_layer_name = layer_name
        
        num_var_grads = 1
        for op_idx, op_type in enumerate(op_types):
            num_var_grads += 1
            print(f"\n\tChecking operation {op_idx}: {op_type}")
                
            if op_type in op_handlers:
                last_op_type = op_type
                handler = op_handlers[op_type]

                if handler.__name__ == 'handler' and handler.__qualname__.startswith('nonlinearity_1d.<locals>'):
                    print(f"\n\t🟢 Using custom handler for {op_type}")

                    #input_ind = handler.__closure__[0].cell_contents

                    n_backgrounds = len(inputs) // 2  # Half of inputs are backgrounds
                    xin0 = tf.stack(inputs[:n_backgrounds])  # First half -> shape (n_backgrounds, 256)
                    rin0 = tf.stack(inputs[n_backgrounds:])  # Second half -> shape (n_backgrounds, 256)

                    # Create a more complete mock op object
                    class MockOp:
                        def __init__(self, op_type):
                            self.type = op_type
                            self.inputs = [None]
                            self.outputs = [None]
                            self.name = f"mock_{op_type}"

                    mock_op = MockOp(op_type)
                    modified_grads = handler(None, mock_op, modified_grads, xin0=xin0, rin0=rin0)[0]
                    
                    # After modifying gradients:
                    #_check_gradient_change(upstream, modified_grads, op_type)
                else:
                    print("\tUsing passthrough handler")

                # Debug gradients after operation
                print(f"\tFinal gradients after {op_type} (first 5):")
                tf.print(modified_grads[:5])

        print("\n\tReturning gradients:")
        print(f"\tModified gradients shape: {modified_grads.shape}")
        print("\tFinal modified gradients (first 5):")
        tf.print(modified_grads[:5])

        # Different return values based on operation type
        var_grads = [None] * num_var_grads
        return (modified_grads,) + tuple(var_grads)

    # Return both the forward pass output and the gradient function
    return outputs, grad_fn


def _check_gradient_change(upstream_grads, modified_grads, op_name, tolerance=1e-6):
    """Check if gradients have changed more than tolerance."""
    # Skip check if shapes don't match
    if upstream_grads.shape != modified_grads.shape:
        print(f"Skipping gradient check for {op_name} due to shape mismatch:")
        print(f"Upstream shape: {upstream_grads.shape}")
        print(f"Modified shape: {modified_grads.shape}")
        return
        
    # Convert to numpy for consistent comparison
    up = upstream_grads.numpy()
    mod = modified_grads.numpy()
    
    # Calculate max absolute difference
    diff = np.abs(up - mod)
    max_diff = np.max(diff)
    
    # Only report if difference exceeds tolerance
    if max_diff > tolerance:
        print(f"\n🔴 Gradient change detected at {op_name}:")
        # Show first few values
        #print(f"First 5 values:")
        #print(f"From: {np.array2string(up[0, :5], precision=16)}")
        #print(f"To:   {np.array2string(mod[0, :5], precision=16)}")
        
        # Show where maximum difference occurs
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Max difference: {max_diff} at index {max_idx}")
        print(f"Values at max diff:")
        print(f"From: {up[max_idx]}")
        print(f"To:   {mod[max_idx]}")
        
        # Show statistics about the differences
        nonzero_diff = diff[diff > tolerance]
        if len(nonzero_diff) > 0:
            total_elements = up.size
            percent_changed = (len(nonzero_diff) / total_elements) * 100
            print(f"{percent_changed:.1f}% of values changed ({len(nonzero_diff)} / {total_elements})")
            print(f"Mean difference where changed: {np.mean(nonzero_diff)}")


@tf.autograph.experimental.do_not_convert
class TF2DeepExplainer(Explainer):
    """DeepExplainer for modern TF2 that computes SHAP values by comparing outputs 
    with background reference values."""
    
    def __init__(self, model, background, 
                 combine_mult_and_diffref=standard_combine_mult_and_diffref, 
                 output_idx=None, batch_size=512, verbose=False):
        """Initialize DeepExplainer.
        
        Args:
            model: TF2 model to explain
            background: Background reference values to compare against
            combine_mult_and_diffref: Function to combine multiplicative and difference reference values
            output_idx: Optional index to explain specific output
            batch_size: Batch size for processing
            verbose: Whether to print debug information
        """
        # Store parameters first
        self.model = model
        self.background = self._validate_background(background)
        self.combine_mult_and_diffref = combine_mult_and_diffref
        self.output_idx = output_idx
        self.batch_size = batch_size
        self.verbose = verbose

        # Initialize base class
        super().__init__()
        
        # Validate model output shape and store expected values
        self._validate_model_output()
        self._store_expected_value()
        
        # Initialize operation handlers and tracking for different operation types
        self.op_handlers = op_handlers.copy()
        
        # Now build the model with custom gradients
        self.model_custom = self._build_custom_gradient_model(use_custom_gradients=True)

    def _validate_background(self, background):
        """Validate and process background data."""
        if isinstance(background, list):
            background = background[0]
        if len(background.shape) == 1:
            background = np.expand_dims(background, 0)
        if background.shape[0] > 5000:
            warnings.warn("Over 5k background samples provided. Consider using smaller random sample for better performance.")
        return background

    def _validate_model_output(self):
        """Validate model output shape and type."""
        dummy_input = tf.zeros([1] + list(self.model.input_shape[1:]))
        output = self.model(dummy_input)
        
        if isinstance(output, list):
            raise ValueError("Model output must be a single tensor!")
        if len(output.shape) >= 3:
            raise ValueError("Model output must be a vector or single value!")
        
        # Store number of outputs and whether it's a multi-output model
        self.noutputs = output.shape[1]
        self.multi_output = self.noutputs > 1

    def _store_expected_value(self):
        """Compute and store expected value from background data."""
        background_output = self.model(self.background)
        self.expected_value = tf.reduce_mean(background_output, axis=0)
    
    def _build_custom_gradient_model(self, use_custom_gradients=True):
        """Build a model with custom gradient computation."""
        inputs = tf.keras.Input(shape=self.model.input_shape[1:])
        x = inputs

        print("\nLayer connections:")
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.layers.InputLayer):
                print(f"\nSkipping input layer: {layer.name}")
                continue
            
            if isinstance(layer, tf.keras.layers.Dropout):
                print(f"\nSkipping dropout layer: {layer.name}")
                continue  # Skip Dropout layers entirely
            
            print(f"\nLayer {i}: {layer.name} ({layer.__class__.__name__})")
            print(f"Input shape: {layer.input_shape}")
            print(f"Output shape: {layer.output_shape}")
            
            if use_custom_gradients:
                x = CustomGradientLayer(
                    op_type=layer.name,
                    op_name=layer.name,
                    op=layer,
                    handler=None,
                    explainer=self
                )(x)
            else:
                x = layer(x)

        return tf.keras.Model(inputs=inputs, outputs=x, name='model_custom')

    def _forward_with_custom_grads(self, inputs):
        """Forward pass using custom gradients."""
        x = inputs
        for layer in self.model.layers:
            if layer.name in self.custom_ops:
                x = self.custom_ops[layer.name](x)
            else:
                x = layer(x)
        return x
    
    def test_custom_grad(self, X):

        # Print layer connections after BatchNorm
        for layer in self.model_custom.layers:
            print(f"Layer: {layer.name}, Input: {layer.input.shape}, Output: {layer.output.shape}")

        """Test custom gradient computation.
        
        Note: Current implementation follows TF1's sample-by-sample approach for compatibility,
        but this is suboptimal. Future optimization could process all samples in batch.
        """
        if not isinstance(X, list):
            X = [X]
        
        print("\nTesting custom gradients...")
        print("Input shapes:", [x.shape for x in X])
        print("Background shapes:", [b.shape for b in self.background])
        
        # Create model output ranks
        # For each sample, create array of output indices
        # E.g., if using a single-output model (one class at a time) with one sample in X:
        # - noutputs = 1
        # - ranks = [[0]] (one [0] for each sample)
        model_output_ranks = np.tile(np.arange(self.noutputs)[None,:], (X[0].shape[0],1))
        
        # Process one sample at a time (following TF1's approach)
        # TODO: Potential optimization - process all samples in batch
        all_grads = []
        for i in range(model_output_ranks.shape[1]): # for each output
            for j in range(X[0].shape[0]): # for each sample
                print(f"\nProcessing sample {j}")
                
                # Get feature index for current sample (matching TF1)
                feature_ind = model_output_ranks[j,i]

                # For each sample, create copies matching the number of background samples
                # This replicates TF1's approach where each sample is compared against all backgrounds
                tiled_X = [np.tile(x[j:j+1], (len(self.background),) + tuple([1 for k in range(len(x.shape)-1)])) 
                        for x in X]
                
                # Create joint input for this sample:
                # Stack backgrounds into single array to match TF1's shape
                stacked_background = tf.stack(self.background)  # Shape: (3, 249, 4)
                joint_input = [tf.convert_to_tensor(
                    np.concatenate([tiled_X[l], stacked_background], 0),  # Will be (6, 249, 4)
                    dtype=tf.float32) 
                    for l in range(len(X))]
                
                print(f'joint_input shape: {[j.shape for j in joint_input]}')
                
                # Compute gradients for this sample using the feature index
                with tf.keras.backend.learning_phase_scope(0):  # 0 = inference mode #TODO: is this needed?
                    with tf.GradientTape() as tape:
                        tape.watch(joint_input)
                        predictions = self.model_custom(joint_input)
                        target_output = predictions[:,feature_ind] if self.multi_output else predictions
                        print(f'joint_input shape: {[j.shape for j in joint_input]}')
                        print(f"Target output shape: {target_output.shape}")

                sample_grads = tape.gradient(target_output, joint_input)
                all_grads.append(sample_grads)
                
                print(f"Sample {j} gradients first 10:", sample_grads[0].numpy().flatten()[:10])
        
        # TODO: Future optimization opportunities:
        # 1. Process all samples in one batch instead of loop
        # 2. Parallelize sample-background comparisons
        # 3. Reduce memory by avoiding sample replication
        
        return all_grads



'''
TODO:
- More general/robust solution for finding the output head:
    -   # Note: BiasAdd is typically the last computation before activation
        output_ops = [op for op in self.compute_ops 
                      if op.type == "BiasAdd" and output_name in op.name]
- /opt/anaconda3/envs/deepstarr2/lib/python3.11/contextlib.py:137: UserWarning: `tf.keras.backend.learning_phase_scope` is deprecated and will be removed after 2020-10-11. To update it, simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
  return next(self.gen)

- background should not be initialized in the constructor, but rather be passed in as an argument to the explain method
'''

'''
IMPLEMENTATION NOTES AND ANALYSIS
===============================

Mission Statement
---------------
This implementation (deep_tf2.py and deep_tf2_utils.py) aims to port DeepSHAP (Deep Learning SHAP) from TensorFlow 1.x (session-based) 
to TensorFlow 2.x (eager execution). DeepSHAP computes SHAP values by comparing outputs with background reference values, requiring
custom gradient computations for non-linear operations. The key challenge is maintaining the same gradient behavior as the TF1
implementation (deep_tf1.py and deep_tf1_utils.py) while working within a modern TF2 framework (deep_tf2.py and deep_tf2_utils.py).

About DeepLIFT
---------------
DeepLIFT's fundamental goal is not to modify gradients directly but to compute attributions based on reference activations rather
than standard gradients. This means:
- Forward pass remains unchanged: We evaluate activations as usual.
- Attribution propagation replaces standard gradient flow: Instead of backpropagating standard gradients, DeepLIFT defines attribution scores by comparing activation changes relative to a reference input.

Key constraints:
- It needs access to both the forward activations and the reference activations at every layer.
- It does not rely on standard backpropagation but instead defines a custom propagation rule, where attributions are computed using quotient derivatives (change in activation over change in input).
- It must support non-linear activations correctly (e.g., for ReLU, handling the case when an activation is zero).
- If the goal is just tracking operations, TensorFlow's gradient tape or function tracing might be useful, but if we want to compute DeepLIFT-style attributions, we'd need to ensure that each layer retains both activations and reference activations, then applies custom backpropagation rules instead of standard gradients.
'''


"""
CRITICAL TODO: Investigate data structure difference between TF1 and TF2 implementations

In TF1's nonlinearity_1d_handler:
- xin0 shape is (3,256)
- Each row contains [sample_values | reference_values]
- The same concatenated row is duplicated 3 times (one per background)

In TF2's nonlinearity_1d_handler:
- xin0 shape is (128,) - just the sample values
- rin0 shape is (128,) - just the reference values
- These are kept as separate tensors

Both implementations contain the same information but structured differently:
TF1: [[xin0|rin0],    TF2: xin0: [sample_values]
      [xin0|rin0],         rin0: [reference_values]
      [xin0|rin0]]

This structural difference needs to be investigated to ensure:
1. Gradient computations are equivalent
2. We're not doing unnecessary duplicate computations in TF1
3. The split vs concatenated approach doesn't affect final attributions
"""