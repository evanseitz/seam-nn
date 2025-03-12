import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
np.set_printoptions(suppress=True, precision=8)


"""
Implementation of custom gradient computation for neural networks, allowing modification
of gradients during backpropagation (e.g., for DeepLIFT-style gradient attribution).
"""

# Make op_handlers global
#global op_handlers
#global original_grads_dict
# Define handlers for custom model
op_handlers = {}

class CustomGradientLayer(tf.keras.layers.Layer):
    """Wraps a layer to enable custom gradient computation during backpropagation.
    Traces operations in the wrapped layer and applies custom gradient handlers."""

    def __init__(self, layer):
        super().__init__(name=f'custom_grad_{layer.name}')
        self.layer = layer
        self.internal_ops = self._trace_internal_ops()
        
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self.layer(inputs)
        return custom_gradient_function(outputs, inputs, self.internal_ops)
    
    def _trace_internal_ops(self):
        """Identifies operations in the layer that might need custom gradient handling.
        Returns a list of operation types (e.g., 'Relu', 'MatMul') in the layer."""
        input_shape = tuple([1] + list(self.layer.input_shape[1:]))
        sample_input = tf.random.normal(input_shape)
        
        @tf.autograph.experimental.do_not_convert
        def traced_fn(x):
            return self.layer(x)
        
        concrete_fn = tf.function(traced_fn).get_concrete_function(sample_input)
        
        # Keep more operations, similar to the old script
        ops = [op for op in concrete_fn.graph.get_operations() 
               if (op.type not in ['Placeholder', 'Const', 'NoOp', 'ReadVariableOp'] or
                   'batch_normalization' in op.name or
                   op.type == 'Identity')
               and any(isinstance(output, tf.Tensor) for output in op.outputs)]
        return [str(op.type) for op in ops]

@tf.custom_gradient
def custom_gradient_function(outputs, inputs, op_types):
    """Custom gradient function that modifies gradients based on operation type.
    For operations with registered handlers (e.g., ReLU), computes modified gradients
    using both the input sample and background reference values."""
    def grad_fn(*args):
        upstream = args[0]
        modified_grads = upstream
        
        print(f"\n=== Layer: {tf.get_current_name_scope()} ===")
        print(f"Operations: {op_types}")
        print(f"Input gradients (first 5): {upstream.numpy().flatten()[:5]}")
        
        # Count required gradients: one for input plus one per operation
        # This ensures TensorFlow receives the expected number of gradients
        
        # For operations with handlers, split inputs into samples and backgrounds
        # and compute modified gradients (e.g., (Δy/Δx) instead of dy/dx)
        
        # Count gradients like the old script
        num_var_grads = 1  # One for input
        for op_idx, op_type in enumerate(op_types):
            num_var_grads += 1  # One for each operation
        
        for op_type in op_types:
            if op_type in op_handlers:
                handler = op_handlers[op_type]
                if handler.__name__ == 'handler' and handler.__qualname__.startswith('nonlinearity_1d.<locals>'):
                    print(f"\n  🔄 ReLU modification")
                    n_backgrounds = len(inputs) // 2
                    xin0 = tf.stack(inputs[:n_backgrounds])
                    rin0 = tf.stack(inputs[n_backgrounds:])
                    mock_op = type('MockOp', (), {'type': op_type, 'inputs': [None], 'outputs': [None], 'name': f"mock_{op_type}"})()
                    modified_grads = handler(None, mock_op, modified_grads, xin0=xin0, rin0=rin0)[0]
                    print(f"  Modified gradients (first 5): {modified_grads.numpy().flatten()[:5]}")
                else:
                    print("\tUsing passthrough handler")

        # Return gradients
        var_grads = [None] * (num_var_grads)
        return (modified_grads,) + tuple(var_grads)

    return outputs, grad_fn

def build_custom_gradient_model(model):
    # Register handlers globally
    print(f"Handlers: {op_handlers}")
    
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            x = CustomGradientLayer(layer)(x)  # Remove op_handlers from here
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def create_simple_model():
    """Simplified version of DeepSTARR with just essential components"""
    # Set initializers with fixed seeds
    kernel_init = tf.keras.initializers.GlorotUniform(seed=42)
    bias_init = tf.keras.initializers.Zeros()

    inputs = kl.Input(shape=(249, 4))
    
    # First conv block
    x = kl.Conv1D(32, kernel_size=7, padding='same', 
                  kernel_initializer=kernel_init, 
                  bias_initializer=bias_init)(inputs)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    
    # Dense block
    x = kl.Flatten()(x)
    x = kl.Dense(64, 
                 kernel_initializer=kernel_init, 
                 bias_initializer=bias_init)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    
    # Output heads
    dev = kl.Dense(1, name='Dense_Dev',
                   kernel_initializer=kernel_init, 
                   bias_initializer=bias_init)(x)
    hk = kl.Dense(1, name='Dense_Hk',
                  kernel_initializer=kernel_init, 
                  bias_initializer=bias_init)(x)
    
    model = Model(inputs=inputs, outputs=[dev, hk])
    model.compile(optimizer=Adam(0.001), loss=['mse', 'mse'])
    return model

# Define a dictionary of custom gradient functions at the top of the file
CUSTOM_GRAD_REGISTRY = {
    "Relu": lambda grads, inputs: tf.raw_ops.ReluGrad(gradients=grads, features=inputs),
    "Elu": lambda grads, inputs: tf.raw_ops.EluGrad(gradients=grads, features=inputs),
    # Users can add their own definitions here
}

'''def get_original_grad_fn(op_type, grads, inputs):
    """Get original gradient either from custom registry or TF registry"""
    # First check our custom registry
    if op_type in CUSTOM_GRAD_REGISTRY:
        return CUSTOM_GRAD_REGISTRY[op_type](grads, inputs)
    
    # Fall back to TF registry
    tf_grad_fn = tf.gradient_function(op_type)
    if tf_grad_fn is not None:
        return tf_grad_fn(op, grads)
        
    # If all else fails, return identity
    print(f"Warning: No gradient function found for {op_type}, using identity")
    return grads'''

# In TF1's DeepLIFT, they store the original gradient functions from TensorFlow's registry
# and use these complete gradient functions which receive both the operation and incoming gradients.
# This is more comprehensive than just using raw ops like ReluGrad, as the original functions
# may handle internal state or edge cases differently.
#
# TF1 approach:
# 1. Store original gradient functions: self.orig_grads[op.type] = registry[op.type]["type"]
# 2. Use them later: orig_grads = self.orig_grads[op.type](op, grads[0])

from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import gen_nn_ops

def get_original_grad_fn(op_type, grads, inputs):
    """Get original gradient either from custom registry or TF registry"""
    # For ReLU, use the same gradient function that TF uses internally
    if op_type == "Relu":
        # This is the actual gradient function TF uses for ReLU
        return gen_nn_ops.relu_grad(grads, inputs)
    
    # Fall back to raw ops if needed
    print(f"Warning: No gradient function found for {op_type}, using identity")
    return grads

def nonlinearity_1d(input_ind):
    def handler(explainer, op, *grads, **kwargs):
        return nonlinearity_1d_handler(input_ind, op, *grads, **kwargs)
    return handler

def nonlinearity_1d_handler(input_ind, op, *grads, xin0=None, rin0=None):
    """Handles gradient computation for nonlinear operations using DeepLIFT rules.
    Instead of standard gradients, computes (Δy/Δx) where Δ represents finite differences
    between sample and reference inputs/outputs."""
    
    combined_inputs = tf.concat([xin0, rin0], axis=0)
    orig_grads = get_original_grad_fn(op.type, grads[0], combined_inputs)


    if 0:  # Switch for vanilla ReLU behavior (debugging only)
        # Concatenate inputs and backgrounds to match gradient batch size
        active_mask = tf.cast(combined_inputs > 0, dtype=grads[0].dtype)
        modified_grads = grads[0] * active_mask
        out = [None for _ in range(len(op.inputs))]
        out[input_ind] = modified_grads

        print(f"orig_grads: {orig_grads}")

        return out
    else:
        xout = getattr(tf.raw_ops, op.type)(features=xin0)
        rout = getattr(tf.raw_ops, op.type)(features=rin0)
        
        # Calculate input differences
        delta_in0 = xin0 - rin0
        dup0 = [2] + [1 for i in delta_in0.shape[1:]]

        result = tf.where(
            tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
            orig_grads[input_ind] if len(op.inputs) > 1 else orig_grads,
            grads[0] * tf.tile((xout - rout) / delta_in0, dup0)
        )
        
        out = [None for _ in range(len(op.inputs))]
        out[input_ind] = result

        return out
    
if 1:  # Switch for custom nonlinearity handling
    op_handlers["Relu"] = nonlinearity_1d(0)
    print("Registered handlers:", list(op_handlers.keys()))

if __name__ == "__main__":
    # Example usage showing:
    # 1. Model creation with fixed random seeds for reproducibility
    # 2. Custom gradient model construction
    # 3. Comparison of original vs custom gradients
    # When handlers are disabled (if 0:), gradients should match exactly
    # When enabled (if 1:), ReLU gradients use DeepLIFT-style computation
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)

    # Create original model
    original_model = create_simple_model()
    
    # Select which output head to explain (0 = dev, 1 = hk)
    class_idx = 0
    output_layer = original_model.outputs[class_idx]
    model_output_idx = tf.keras.Model(inputs=original_model.input, outputs=output_layer)

    # Test with dummy data (with fixed seed)
    batch_size = 1
    n_backgrounds = 3
    x = tf.random.normal((batch_size, 249, 4), seed=42)
    x_with_backgrounds = tf.concat([
        tf.tile(x, [1, 1, 1]),
        tf.tile(tf.random.normal((n_backgrounds, 249, 4), seed=43), [1, 1, 1])
    ], axis=0)

    custom_model = build_custom_gradient_model(model_output_idx)
    
    # Compare gradients between original and custom model
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_with_backgrounds)
        orig_pred = model_output_idx(x_with_backgrounds)
        custom_pred = custom_model(x_with_backgrounds)
        orig_loss = tf.reduce_mean(orig_pred)
        custom_loss = tf.reduce_mean(custom_pred)
    
    orig_grads = tape.gradient(orig_loss, x_with_backgrounds)
    custom_grads = tape.gradient(custom_loss, x_with_backgrounds)
    
    print("Original prediction shape:", orig_pred.shape)
    print("Custom prediction shape:", custom_pred.shape)
    print("Original gradient shape:", orig_grads.shape)
    print("Custom gradient shape:", custom_grads.shape)
    
    print("\nFirst 10 original gradients:", orig_grads.numpy().flatten()[:10])
    print("First 10 custom gradients:", custom_grads.numpy().flatten()[:10]) 