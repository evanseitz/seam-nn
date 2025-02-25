import numpy as np
import warnings
from core.explainer import Explainer
from .utils import standard_combine_mult_and_diffref
import tensorflow as tf
from .deep_tf2_utils import (
    op_handlers,
    backward_walk_ops_tf2,
    forward_walk_ops_tf2,
    break_dependence,
    get_model_graph,
    passthrough,
    nonlinearity_1d
)

class CustomGradientLayer(tf.keras.layers.Layer):
    def __init__(self, op_type, op_name, op, handler, explainer):
        super().__init__(name=f'custom_grad_{op_name}')
        self.layer = op
        self.explainer = explainer
        self.handler = handler
        self.op_type = op_type
        
        # Trace internal operations
        self.internal_ops = self._trace_internal_ops()
        
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
        
        # Filter out Placeholder operations
        ops = [op for op in concrete_fn.graph.get_operations() if op.type != 'Placeholder']
        
        if self.explainer.verbose:
            print(f"\nTraced operations for {self.layer.name}:")
            for op in ops:
                print(f"- {op.type}: {op.name}")
        return ops

    @tf.custom_gradient
    def call(self, inputs):
        # Forward pass: use the layer's forward pass
        outputs = self.layer(inputs)
        
        def grad(upstream, variables=None):
            # Look at each operation in this layer
            for op in self.internal_ops:
                if op.type in self.explainer.op_handlers:
                    handler = self.explainer.op_handlers[op.type]
                    if handler == break_dependence:
                        continue
                        
                    if handler == passthrough:
                        # Use original gradients for passthrough
                        with tf.GradientTape() as tape:
                            tape.watch(inputs)
                            layer_output = self.layer(inputs)
                        input_grads = tape.gradient(layer_output, inputs, output_gradients=upstream)
                    else:
                        # Check if this is a nonlinearity_1d handler by looking at its name
                        if handler.__name__ == 'handler' and handler.__qualname__.startswith('nonlinearity_1d.<locals>'):
                            if self.explainer.verbose:
                                print(f"Handling nonlinearity_1d for: {op.type}")
                            
                            # Split input into x and reference
                            xin0, rin0 = tf.split(inputs, 2)
                            
                            # Call the actual handler from deep_tf2_utils
                            input_grads = handler(self.explainer, op, upstream, xin0=xin0, rin0=rin0)[0]
                        else:
                            if self.explainer.verbose:
                                print(f"Need to implement non-passthrough handler for: {op.type}")
                            input_grads = upstream  # Temporary fallback
            
            if variables is not None:
                var_grads = [None] * len(variables)
                return input_grads, var_grads
            return input_grads

        return outputs, grad

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
        
        # Setup operation tracking between inputs and outputs
        self.between_ops = self._setup_op_tracking()
        
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
        
        self.multi_output = len(output.shape) > 1

    def _store_expected_value(self):
        """Compute and store expected value from background data."""
        background_output = self.model(self.background)
        self.expected_value = tf.reduce_mean(background_output, axis=0)

    def _setup_op_tracking(self):
        """Setup operation tracking between inputs and outputs.
        
        This method:
        1. Gets the raw computation graph from the model
        2. Filters out training-phase operations while keeping core computation
        3. Identifies input and output operations
        4. Performs bidirectional graph traversal to find all operations between input and output
        
        Returns:
            List of operations that form the computation path between inputs and outputs
        """
        # Get all operations and setup computation graph
        all_ops = get_model_graph(self.model)  # Get raw computation graph
                
        # Filter out training-phase operations but keep core computation
        # Note: In TF2, dropout and batch norm are handled differently than TF1:
        # - Dropout uses Identity ops during inference instead of Switch/Merge
        # - BatchNorm's moving averages are handled by the layer implementation
        training_phase_ops = {
            "Switch", "Merge",  # Control flow ops
            "AssignMovingAvg", "AssignMovingAvgDefault",  # BatchNorm moving stats
            "ReadVariableOp", "ResourceGather",  # Variable ops
            "Const",  # Constants
            "NoOp"  # No operations
        }
        
        self.compute_ops = [op for op in all_ops if op.type not in training_phase_ops]
        
        # Find input operation (used as starting point for graph traversal)
        self.input_op = next(op for op in all_ops 
                           if op.type == "Placeholder" and op.name == "x")
        
        # Add input op to compute ops if not already there
        if self.input_op not in self.compute_ops:
            self.compute_ops.append(self.input_op)

        # Find output op by looking at the model's output layer
        output_layer = self.model.layers[-1]
        output_name = output_layer.name
        
        # Find the BiasAdd op that corresponds to this output layer
        # Note: BiasAdd is typically the last computation before activation
        output_ops = [op for op in self.compute_ops 
                      if op.type == "BiasAdd" and output_name in op.name]
        
        if len(output_ops) != 1:
            # If we can't find it by name, try finding the last BiasAdd in compute_ops
            output_ops = [op for op in self.compute_ops if op.type == "BiasAdd"]
            if not output_ops:
                raise ValueError("Could not find output operation")
            self.output_op = output_ops[-1]  # Take the last BiasAdd
        else:
            self.output_op = output_ops[0]
        
        # Find all operations between inputs and outputs using bidirectional walk
        dependence_breakers = [k for k in op_handlers if op_handlers[k] == break_dependence] # Identify operations that should break dependencies
        
        # Do backward walk from output
        back_ops = backward_walk_ops_tf2(
            start_ops=[self.output_op],
            compute_ops=self.compute_ops,
            dependence_breakers=dependence_breakers
        )
        
        # Then do forward walk from input
        forward_ops = forward_walk_ops_tf2(
            start_ops=[self.input_op],
            compute_ops=self.compute_ops,
            dependence_breakers=dependence_breakers
        )
        
        # Get intersection of forward and backward reachable ops
        between_ops = [op for op in forward_ops if op in back_ops]        
        between_ops = [op for op in between_ops if op.type != 'Placeholder']

        if self.verbose:
            print(f"\nAll ops: {[op.type for op in all_ops]}")
            print(f"\nCompute ops: {[op.type for op in self.compute_ops]}")
            print(f"\nInput op: {self.input_op.type}: {self.input_op.name}")
            print(f"\nOutput op: {self.output_op.type}: {self.output_op.name}")
            print(f"\nDependence breakers: {dependence_breakers}")
            print(f"\nBackward ops: {[op.type for op in back_ops]}")
            print(f"\nForward ops: {[op.type for op in forward_ops]}")
        print(f"\nBetween ops: {[op.type for op in between_ops]}")

        return between_ops
    
    def _build_custom_gradient_model(self, use_custom_gradients=True):
        """Build a model with custom gradient computation."""
        # Create new input layer
        inputs = tf.keras.Input(shape=self.model.input_shape[1:])
        x = inputs

        # Apply each layer in sequence
        for layer in self.model.layers:
            if use_custom_gradients:
                # Wrap the entire layer computation
                x = CustomGradientLayer(
                    op_type=layer.name,
                    op_name=layer.name,
                    op=layer,
                    handler=None,  # We'll handle gradients directly
                    explainer=self
                )(x)
            else:
                x = layer(x)

        # Create and return the new model
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
        """Test custom gradient computation."""
        if not isinstance(X, list):
            X = [X]
        
        # Create joint input
        joint_input = [tf.concat([self.background, x], 0) for x in X]
        
        print("\nTesting custom gradients...")

        X_tensor = tf.convert_to_tensor(X[0], dtype=tf.float32)

        
        # Compute gradients with custom model
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)  # Watch the input tensor
            # Create joint input inside the tape context
            joint = tf.concat([self.background, X_tensor], 0)
            predictions = self.model_custom(joint)
            target_output = predictions[:,self.output_idx] if self.multi_output else predictions
        
        #X = tf.convert_to_tensor(X[0], dtype=tf.float32)
        #custom_grads = tape.gradient(target_output, joint_input)
        custom_grads = tape.gradient(target_output, X_tensor)
        
        # Compare with original model
        '''with tf.GradientTape() as orig_tape:
            orig_tape.watch(joint_input)
            orig_preds = self.model(joint_input[0])
            orig_target = orig_preds[:,self.output_idx] if self.multi_output else orig_preds
        
        orig_grads = orig_tape.gradient(orig_target, joint_input)'''
        
        print("\nGradient Comparison:")
        print("Custom gradients (first 10):", custom_grads[0].numpy().flatten()[:10])
        #print("Original gradients (first 10):", orig_grads[0].numpy().flatten()[:10])
        
        return custom_grads



'''
TODO:
- More general/robust solution for output head:
    -   # Note: BiasAdd is typically the last computation before activation
        output_ops = [op for op in self.compute_ops 
                      if op.type == "BiasAdd" and output_name in op.name]
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