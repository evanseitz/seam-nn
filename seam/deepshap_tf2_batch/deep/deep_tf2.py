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
        self.internal_ops = self._trace_internal_ops()

    def call(self, inputs):
        outputs = self.layer(inputs)
        outputs = tf.identity(outputs)  # Ensure outputs remain connected
        
        # Convert operations to their types for passing to custom_gradient_function
        op_types = [op.type for op in self.internal_ops]
        return custom_gradient_function(outputs, inputs, op_types)
    
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
        
        # Filter out non-tensor operations
        ops = [
            op for op in concrete_fn.graph.get_operations() 
            if op.type not in ['Placeholder', 'Const', 'NoOp', 'ExpandDims', 'Identity', 'ReadVariableOp']
            and any(isinstance(output, tf.Tensor) for output in op.outputs)
        ]
        
        if self.explainer.verbose:
            print(f"\nTraced operations for {self.layer.name}:")
            for op in ops:
                print(f"- {op.type}: {op.name}")
        return ops



@tf.custom_gradient
def custom_gradient_function(outputs, inputs, op_types):
    """Standalone function to apply custom gradients"""

    def grad_fn(upstream, variables=None):
        print("\nCustomGradientLayer grad")
        print(f"Upstream gradient shape: {upstream.shape}")
        print(f"Input shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Operation types: {op_types}")
        
        # Debug initial upstream gradients
        print("\nInitial upstream gradients (first 5):")
        if len(upstream.shape) == 3:
            tf.print(upstream[0, 0, :5])
        elif len(upstream.shape) == 2:
            tf.print(upstream[0, :5])
        else:
            tf.print(upstream[:5])

        modified_grads = upstream

        for op_type in op_types:
            if op_type in op_handlers:
                handler = op_handlers[op_type]
                print(f"\nProcessing operation: {op_type}")
                print(f"Handler: {handler.__name__}")
                
                # Debug gradients before modification
                print(f"Gradients before {op_type} (first 5):")
                if len(modified_grads.shape) == 3:
                    tf.print(modified_grads[0, 0, :5])
                elif len(modified_grads.shape) == 2:
                    tf.print(modified_grads[0, :5])
                else:
                    tf.print(modified_grads[:5])

                if handler == break_dependence:
                    print("Breaking dependence")
                    continue

                if handler == passthrough:
                    print("Using passthrough handler")
                    print(f"Passthrough gradient shape: {modified_grads.shape}")
                    # Force gradient tracking for passthrough
                    modified_grads = tf.identity(modified_grads) #NEW

                else:
                    print(f"Processing op: {op_type}")
                    if handler.__name__ == 'handler' and handler.__qualname__.startswith('nonlinearity_1d.<locals>'):
                        print(f"Using custom handler for {op_type}")

                        input_ind = handler.__closure__[0].cell_contents
                        print(f"Input index: {input_ind}")

                        if len(inputs.shape) == 2:
                            xin0, rin0 = tf.split(inputs[input_ind], 2)
                        else:
                            xin0, rin0 = tf.split(inputs, 2, axis=0)
                        print(f"xin0 shape: {xin0.shape}")
                        print(f"rin0 shape: {rin0.shape}")

                        print("ReLU input values (first 5):")
                        if len(xin0.shape) == 3:
                            tf.print(xin0[0, 0, :5])
                        elif len(xin0.shape) == 2:
                            tf.print(xin0[0, :5])
                        else:
                            tf.print(xin0[:5])
                            
                        print("ReLU reference values (first 5):")
                        if len(rin0.shape) == 3:
                            tf.print(rin0[0, 0, :5])
                        elif len(rin0.shape) == 2:
                            tf.print(rin0[0, :5])
                        else:
                            tf.print(rin0[:5])

                        # Create a more complete mock op object
                        class MockOp:
                            def __init__(self, op_type):
                                self.type = op_type
                                self.inputs = [None]
                                self.outputs = [None]
                                self.name = f"mock_{op_type}"

                        mock_op = MockOp(op_type)
                        modified_grads = handler(None, mock_op, modified_grads, xin0=xin0, rin0=rin0)[0]
                        
                        # Force gradient tracking after modification
                        modified_grads = tf.identity(modified_grads) #NEW
                        
                        print(f"Modified gradient shape: {modified_grads.shape if modified_grads is not None else None}")
                        print(f"Modified gradients after {op_type} (first 5):")
                        if len(modified_grads.shape) == 3:
                            tf.print(modified_grads[0, 0, :5])
                        elif len(modified_grads.shape) == 2:
                            tf.print(modified_grads[0, :5])
                        else:
                            tf.print(modified_grads[:5])
                        
                        if modified_grads is None:
                            print("WARNING: modified_grads is None, using upstream")
                            modified_grads = upstream

                # Debug gradients after operation
                print(f"Final gradients after {op_type} (first 5):")
                if len(modified_grads.shape) == 3:
                    tf.print(modified_grads[0, 0, :5])
                elif len(modified_grads.shape) == 2:
                    tf.print(modified_grads[0, :5])
                else:
                    tf.print(modified_grads[:5])

        print("\nReturning gradients:")
        print(f"Modified gradients shape: {modified_grads.shape}")
        print("Final modified gradients (first 5):")
        if len(modified_grads.shape) == 3:
            tf.print(modified_grads[0, 0, :5])
        elif len(modified_grads.shape) == 2:
            tf.print(modified_grads[0, :5])
        else:
            tf.print(modified_grads[:5])

        # Different return values based on operation type
        if 'Conv2D' in op_types:
            # Conv2D case - return 5 gradients
            # 1 for input, 1 for kernel, 1 for bias, 2 for additional variables
            var_grads = [None] * 4  # Conv2D has 4 additional gradients
            print(f"Conv2D gradients: {var_grads}")
            return (modified_grads,) + tuple(var_grads)
        elif 'MatMul' in op_types or 'BiasAdd' in op_types:
            # Dense layer case - return 4 gradients
            var_grads = [None] * 3  # For kernel, bias, and any additional variable
            print(f"Dense layer gradients: {var_grads}")
            return (modified_grads,) + tuple(var_grads)
        elif 'Relu' in op_types:
            # ReLU case - return 3 gradients
            var_grads = [None] * 2  # ReLU typically has 2 trainable variables
            print(f"ReLU gradients: {var_grads}")
            return (modified_grads,) + tuple(var_grads)
        elif any(op in op_types for op in ['AddV2', 'Rsqrt', 'Mul', 'Sub']):
            # BatchNorm case - return 9 gradients
            # 1 for input, 4 for trainable vars (gamma, beta, mean, var), 
            # and 4 for internal ops
            var_grads = [None] * 8  # BatchNorm has 8 additional gradients
            print(f"BatchNorm gradients: {var_grads}")
            return (modified_grads,) + tuple(var_grads)
        elif 'Reshape' in op_types:
            # Reshape case - return 3 gradients
            # 1 for input, 1 for shape tensor, 1 for any additional variable
            var_grads = [None] * 2  # Reshape has 2 additional gradients
            print(f"Reshape gradients: {var_grads}")
            return (modified_grads,) + tuple(var_grads)
        elif 'MaxPool' in op_types or 'Squeeze' in op_types:
            # MaxPool/Squeeze case - return 4 gradients
            # 1 for input, 3 for configuration variables (ksize, strides, padding)
            var_grads = [None] * 3  # MaxPool/Squeeze has 3 additional gradients
            print(f"MaxPool/Squeeze gradients: {var_grads}")
            return (modified_grads,) + tuple(var_grads)
        else:
            # No operations case - return 2 gradients
            print("No operations - returning 2 gradients")
            return modified_grads, None

    return outputs, grad_fn









'''class CustomGradientLayer(tf.keras.layers.Layer):
    def __init__(self, op_type, op_name, op, handler, explainer):
        super().__init__(name=f'custom_grad_{op_name}')
        self.layer = op
        self.explainer = explainer
        self.handler = handler
        self.op_type = op_type
        self.internal_ops = self._trace_internal_ops()

    @tf.custom_gradient
    def call(self, inputs):
        outputs = self.layer(inputs)
        
        def grad_fn(upstream, variables=None):
            print(f"\nCustomGradientLayer grad for {self.layer.name}")
            print(f"Upstream gradient shape: {upstream.shape}")

            modified_grads = upstream

            for op in self.internal_ops:
                if op.type in self.explainer.op_handlers:
                    handler = self.explainer.op_handlers[op.type]
                    
                    if handler == break_dependence:
                        continue
                        
                    if handler == passthrough:
                        with tf.GradientTape() as tape:
                            tape.watch(inputs)
                            layer_output = self.layer(inputs)
                        modified_grads = tape.gradient(layer_output, inputs, output_gradients=upstream)
                        print("Using passthrough gradients")
                    else:
                        print(f"Processing op: {op.type}")
                        if handler.__name__ == 'handler' and handler.__qualname__.startswith('nonlinearity_1d.<locals>'):
                            print(f"Using custom handler for {op.type}")
                            
                            input_ind = handler.__closure__[0].cell_contents
                            
                            if len(inputs.shape) == 2:
                                xin0, rin0 = tf.split(inputs[input_ind], 2)
                            else:
                                xin0, rin0 = tf.split(inputs, 2, axis=0)
                            modified_grads = handler(self.explainer, op, modified_grads, xin0=xin0, rin0=rin0)[0]
                        else:
                            modified_grads = upstream
            
            # Handle variable gradients
            if variables is not None:
                var_grads = [None] * len(variables)  # Return None for variable gradients
                return modified_grads, var_grads
            return modified_grads

        return outputs, grad_fn


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
        return ops'''



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
        
        # Store number of outputs and whether it's a multi-output model
        self.noutputs = output.shape[1]
        self.multi_output = self.noutputs > 1

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
            if isinstance(layer, tf.keras.layers.InputLayer):
                print(f"Skipping input layer: {layer.name}")
                continue  # Do not wrap InputLayer
            if use_custom_gradients:
                # Wrap each layer with our custom gradient handler
                x = CustomGradientLayer(
                    op_type=layer.name,
                    op_name=layer.name,
                    op=layer,
                    handler=None,  # We'll handle gradients in the layer
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
                print(f"Tiled X shapes (sample {j}):", [t.shape for t in tiled_X])
                
                # Create joint input for this sample:
                # Stack backgrounds into single array to match TF1's shape
                stacked_background = tf.stack(self.background)  # Shape: (3, 249, 4)
                joint_input = [tf.convert_to_tensor(
                    np.concatenate([tiled_X[l], stacked_background], 0),  # Will be (6, 249, 4)
                    dtype=tf.float32) 
                    for l in range(len(X))]
                
                # Process one sample-background pair
                pair_input = [tf.stack([joint_input[0][j], joint_input[0][j + len(self.background)]])]
                print(f"Pair input shapes (sample {j}):", [p.shape for p in pair_input])
                
                # Compute gradients for this sample using the feature index
                with tf.keras.backend.learning_phase_scope(0):  # 0 = inference mode #TODO: is this needed?
                    with tf.GradientTape() as tape:
                        tape.watch(pair_input)
                        predictions = self.model_custom(pair_input[0])
                        target_output = predictions[:,feature_ind] if self.multi_output else predictions
                        print(f'pair_input[0] shape: {pair_input[0].shape}')
                        print(f"Target output shape: {target_output.shape}")

                #tf.print("Pair input contributions:", tape.gradient(target_output, pair_input))

                sample_grads = tape.gradient(target_output, pair_input)
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


'''
CRITICAL ASSESSMENT BY CHATGPT4
-------------------------------







'''