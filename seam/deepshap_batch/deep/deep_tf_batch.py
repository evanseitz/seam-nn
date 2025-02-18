import numpy as np
import warnings
from core.explainer import Explainer
from deep.utils import standard_combine_mult_and_diffref
from distutils.version import LooseVersion
import sys
import tensorflow as tf
from typing import Union, List, Callable, Tuple, Optional

keras = None
tf_ops = None
tf_gradients_impl = None

def tf_maxpool(inputs, layer, grads):
    """Gradient function for MaxPooling layers."""
    out_shape = layer.output_shape
    pool_size = layer.pool_size
    strides = layer.strides
    padding = layer.padding.upper()
    
    # Forward pass to get the mask
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        outputs = tf.nn.max_pool2d(inputs, pool_size, strides, padding)
    
    # Get the mask from the forward pass
    mask = tape.gradient(outputs, inputs)
    return grads * mask

def tf_passthrough(inputs, layer, grads):
    """Gradient function for layers that just pass through gradients."""
    return grads

def tf_avgpool(inputs, layer, grads):
    """Gradient function for AvgPooling layers."""
    out_shape = layer.output_shape
    pool_size = layer.pool_size
    strides = layer.strides
    padding = layer.padding.upper()
    
    # Distribute gradients evenly in each pooling window
    return tf.nn.avg_pool2d_grad(
        inputs.shape,
        grads,
        pool_size,
        strides,
        padding
    )

class TF2DeepExplainer(Explainer):
    """Deep SHAP implementation for TensorFlow 2.x models.
    
    This implementation uses TF2's eager execution and GradientTape for computing
    SHAP values, similar to how PyTorch's autograd system works.
    """
    
    def __init__(self, 
                 model: Union[tf.keras.Model, Tuple[tf.keras.Model, tf.keras.layers.Layer]],
                 data: Union[np.ndarray, tf.Tensor, Callable],
                 combine_mult_and_diffref: Callable = standard_combine_mult_and_diffref):
        """Initialize the TF2DeepExplainer.
        
        Parameters
        ----------
        model : tf.keras.Model or (model, layer)
            A Keras model or a tuple of (model, layer). If a tuple is provided,
            explanations will be computed for the input of the specified layer.
            The layer must be a layer in the model.
            
        data : array-like or callable
            Background data for model explanation. Can be:
            - A numpy array or TF tensor of samples
            - A function that generates background data for a given input
            
        combine_mult_and_diffref : callable
            Function to combine multipliers with input/reference differences.
            Default implements the standard DeepLIFT approach.
        """
        self.combine_mult_and_diffref = combine_mult_and_diffref
        self.layer = None
        self.interim = False
        self.interim_inputs_shape = None
        self.expected_value = None
        
        self.model = model
        
        # check if we have multiple inputs
        self.multi_input = False
        if hasattr(data, '__call__'):
            sample_data = data(None)
        else:
            sample_data = data
            if not isinstance(data, list):
                data = [data]
        if isinstance(sample_data, list):
            self.multi_input = True
        if not isinstance(sample_data, list):
            sample_data = [sample_data]
        self.data = data
        
        # Setup background data
        if callable(data):
            self.bg_data = data(None)
        elif isinstance(data, np.ndarray):
            self.bg_data = tf.convert_to_tensor(data, dtype=tf.float32)
        else:
            self.bg_data = data[0]  # Take first element if list
        
        # Get output dimension using a sample from background data
        dummy_input = sample_data[0][0:1]  # Take first sample from first input
        
        self.output_dim = self.model(dummy_input).shape[-1]
        
        # Store number of background samples
        self.num_background = self.bg_data.shape[0]
        
        # Set model to evaluation mode
        self.model.trainable = False
        
        # Detect multi-output and compute expected values
        outputs = self.model(self.bg_data)
        
        self.multi_output = False
        self.num_outputs = 1
        if len(outputs.shape) > 1 and outputs.shape[1] > 1:
            self.multi_output = True
            self.num_outputs = outputs.shape[1]
            
        if not callable(data):
            self.expected_value = tf.reduce_mean(outputs, axis=0).numpy()

    def _setup_model(self, model: Union[tf.keras.Model, Tuple[tf.keras.Model, tf.keras.layers.Layer]]) -> None:
        """Setup model and handle layer attribution if needed.
        
        Parameters
        ----------
        model : tf.keras.Model or (model, layer)
            Model to explain, optionally with a specific layer
        """
        if isinstance(model, tuple):
            self.interim = True
            model, layer = model
            self.layer = layer
            self._register_layer_hooks(layer)
        else:
            self.interim = False
        
        self.model = model
        self.model.trainable = False  # Ensure we're in inference mode

    def _setup_data(self, data: Union[np.ndarray, tf.Tensor, List, Callable]) -> None:
        """Setup background data and determine input configuration.
        
        Parameters
        ----------
        data : array-like or callable
            Background data or function to generate it
        """
        # Check if we have a data generation function
        if callable(data):
            sample_data = data(None)
        else:
            sample_data = data
            if not isinstance(data, list):
                data = [data]
            
        # Determine if we have multiple inputs
        self.multi_input = isinstance(sample_data, list)
        if not isinstance(sample_data, list):
            sample_data = [sample_data]
        
        # Convert to tensors and check shapes
        sample_data = [tf.convert_to_tensor(d) for d in sample_data]
        
        # Store data and compute expected values
        self.data = data
        outputs = self.model(sample_data[0] if len(sample_data) == 1 else sample_data)
        
        # Determine output configuration
        self.multi_output = False
        self.num_outputs = 1
        if len(outputs.shape) > 1 and outputs.shape[1] > 1:
            self.multi_output = True
            self.num_outputs = outputs.shape[1]
        
        # Compute expected value if data is not callable
        if not callable(data):
            self.expected_value = tf.reduce_mean(outputs, axis=0).numpy()

    def _register_layer_hooks(self, layer: tf.keras.layers.Layer) -> None:
        """Register hooks to capture intermediate layer inputs.
        
        Parameters
        ----------
        layer : tf.keras.layers.Layer
            The layer to hook into
        """
        # Create a wrapper class to capture layer inputs
        class InputCapture(tf.keras.layers.Layer):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
                self.target_input = None  # Matches PyTorch's naming convention
            
            def call(self, inputs):
                self.target_input = inputs
                return self.layer(inputs)
            
            def get_config(self):
                # Required for serialization
                return {"layer": self.layer}
        
        # Replace the original layer with our wrapped version
        wrapped_layer = InputCapture(layer)
        
        # Find and replace the layer in the model
        for i, l in enumerate(self.model.layers):
            if l == layer:
                self.model.layers[i] = wrapped_layer
                self.layer = wrapped_layer
                break
        
        # Ensure the model knows about the change
        self.model._init_graph_network(self.model.inputs, self.model.outputs)

    def _cleanup_hooks(self):
        """Remove hooks and cleanup captured inputs."""
        if self.layer is not None and hasattr(self.layer, 'target_input'):
            del self.layer.target_input

    def _get_expected_value(self, data: tf.Tensor) -> tf.Tensor:
        """Compute the expected value of the model output.
        
        Parameters
        ----------
        data : tf.Tensor
            Background data to compute expectations over
            
        Returns
        -------
        tf.Tensor
            Expected value of model output
        """
        return tf.reduce_mean(self.model(data), axis=0)

    def _variable_inputs(self, op):
        """ Return which inputs of this operation are variable (i.e. depend on the model inputs).
        """
        if op.name not in self._vinputs:
            self._vinputs[op.name] = np.array([t.op in self.between_ops or t.name in [x.name for x in self.model_inputs] for t in op.inputs])
        return self._vinputs[op.name]

    def phi_symbolic(self, i):
        """ Get the SHAP value computation graph for a given model output.
        """
        if self.phi_symbolics[i] is None:

            # replace the gradients for all the non-linear activations
            # we do this by hacking our way into the registry (TODO: find a public API for this if it exists)
            reg = tf_ops._gradient_registry._registry
            for n in op_handlers:
                if n in reg:
                    self.orig_grads[n] = reg[n]["type"]
                    if op_handlers[n] is not passthrough:
                        reg[n]["type"] = self.custom_grad
                elif n in self.used_types:
                    raise Exception(n + " was used in the model but is not in the gradient registry!")
            # In TensorFlow 1.10 they started pruning out nodes that they think can't be backpropped
            # unfortunately that includes the index of embedding layers so we disable that check here
            if hasattr(tf_gradients_impl, "_IsBackpropagatable"):
                orig_IsBackpropagatable = tf_gradients_impl._IsBackpropagatable
                tf_gradients_impl._IsBackpropagatable = lambda tensor: True
            
            # define the computation graph for the attribution values using custom a gradient-like computation
            try:
                out = self.model_output[:,i] if self.multi_output else self.model_output
                self.phi_symbolics[i] = tf.gradients(out, self.model_inputs)

            finally:

                # reinstate the backpropagatable check
                if hasattr(tf_gradients_impl, "_IsBackpropagatable"):
                    tf_gradients_impl._IsBackpropagatable = orig_IsBackpropagatable

                # restore the original gradient definitions
                for n in op_handlers:
                    if n in reg:
                        reg[n]["type"] = self.orig_grads[n]
        return self.phi_symbolics[i]

    def gradient(self, idx, inputs):        
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = self.model(inputs)
            
            selected = outputs[:, idx]
            
        grads = tape.gradient(selected, inputs)
        
        return grads

    def _compute_gradients(self, inputs, output_idx):
        # Get original input shape and flatten background dimension
        input_shape = inputs[0].shape
        batch_size = input_shape[0]
        n_backgrounds = input_shape[1]
        
        # For 1D inputs, reshape to [batch*backgrounds, features]
        # For 2D inputs, reshape to [batch*backgrounds, seq_len, features]
        reshaped_input = tf.reshape(inputs[0], [-1, input_shape[-1]]) if len(input_shape) == 3 else \
                         tf.reshape(inputs[0], [-1] + list(input_shape[2:]))
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(reshaped_input)
            # Forward pass through model
            outputs = self.model(reshaped_input)
            
            # Reshape outputs to include background dimension
            outputs = tf.reshape(outputs, [batch_size, n_backgrounds, -1])
            selected = outputs[..., output_idx]
        
        # Get gradients and reshape back to original input shape
        grads = tape.gradient(selected, reshaped_input)
        grads = tf.reshape(grads, input_shape)
        
        del tape
        return grads

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max", batch_size=50):
        if not isinstance(X, list):
            X = [X]
        X = [tf.convert_to_tensor(x) for x in X]
        
        if not isinstance(self.bg_data, list):
            self.bg_data = [self.bg_data]
        
        shap_values = []
        n_backgrounds = self.bg_data[0].shape[0]
        
        for output_idx in range(self.num_outputs):
            sample_values = []
            num_samples = X[0].shape[0]
            
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_size_actual = batch_end - batch_start
                
                # Get current batch
                batch_X = [x[batch_start:batch_end] for x in X]
                
                # Tile the input for samples - adjust for sequence data
                tiled_X = []
                for x in batch_X:
                    if len(x.shape) == 3:  # sequence data
                        tiled_shape = [1, n_backgrounds, 1, 1]
                    else:  # regular data
                        tiled_shape = [1, n_backgrounds, 1]
                    tiled_X.append(tf.tile(tf.expand_dims(x, 1), tiled_shape))
                
                # Reshape background data for samples
                bg_data_reshaped = []
                for bg in self.bg_data:
                    # Reshape to match tiled_X shape: [batch_size, n_backgrounds, features]
                    bg_tiled = tf.reshape(bg, [1, n_backgrounds] + list(bg.shape[1:]))
                    # Tile to match batch size
                    bg_tiled = tf.tile(bg_tiled, [batch_size_actual, 1] + [1] * len(bg.shape[1:]))
                    bg_data_reshaped.append(bg_tiled)
                
                # Combine the samples
                joint_x = [tf.concat([tx, bg], axis=0) for tx, bg in zip(tiled_X, bg_data_reshaped)]
                
                # Use persistent tape for gradient computation with proper cleanup
                try:
                    with tf.GradientTape(persistent=True) as tape:
                        grads = self._compute_gradients(joint_x, output_idx)
                    tiled_grads = grads[:tiled_X[0].shape[0]]
                finally:
                    # Ensure tape is deleted to free memory
                    del tape
                
                # Compute multipliers
                multipliers = [tx - bg for tx, bg in zip(tiled_X, bg_data_reshaped)]
                
                # Reshape for batch computation while preserving feature dimensions
                grads_reshaped = tf.reshape(tiled_grads, [batch_size_actual, n_backgrounds] + list(X[0].shape[1:]))
                mult_reshaped = tf.reshape(multipliers[0], [batch_size_actual, n_backgrounds] + list(X[0].shape[1:]))
                
                # Use einsum for multiplication, keep reduce_mean for reduction
                values = [tf.reduce_mean(
                    tf.einsum('bij...,bij...->bij...', grads_reshaped, mult_reshaped),
                    axis=1
                )]

                print("\n=== Batch-based Debug ===")
                print("tiled_X shape:", [tx.shape for tx in tiled_X])
                print("bg_data shape:", [bg.shape for bg in bg_data_reshaped])
                print("joint_x shape:", [ji.shape for ji in joint_x])
                print("\nGradients shape:", tiled_grads.shape)
                print("First few gradient values:", tiled_grads[:5,0] if len(tiled_grads.shape) > 1 else tiled_grads[:5])
                print("\nFinal attribution shape:", values[0].shape)
                print("First few attribution values:", values[0][:5,0] if len(values[0].shape) > 1 else values[0][:5])
                print("=== End of Batch-based Debug ===")
                
                # Clear references to large tensors
                del tiled_X, bg_data_reshaped, joint_x, grads, tiled_grads
                del grads_reshaped, mult_reshaped
                
                sample_values.extend(values[0] if len(values) == 1 else values)
            
            values = tf.stack(sample_values)
            shap_values.append(values)
            
            # Clear GPU memory after each output
            tf.keras.backend.clear_session()
        
        if not self.multi_output:
            return shap_values[0]
        return shap_values

    def run(self, out, model_inputs, X):
        """ Runs the model while also setting the learning phase flags to False.
        """
        feed_dict = dict(zip(model_inputs, X))
        for t in self.learning_phase_flags:
            feed_dict[t] = False
        return self.session.run(out, feed_dict)

    def custom_grad(self, op, *grads):
        """ Passes a gradient op creation request to the correct handler.
        """
        return op_handlers[op.type](self, op, *grads)

    def cleanup(self) -> None:
        """Cleanup resources and remove hooks."""
        self._cleanup_hooks()
        tf.keras.backend.clear_session()

    def __enter__(self):
        """Context manager enter."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    @staticmethod
    def supports_model_with_masker(model, masker) -> bool:
        """Check if this explainer supports the given model and masker.
        
        Parameters
        ----------
        model : object
            Model to explain
        masker : object
            Masker to use for explanation
            
        Returns
        -------
        bool
            True if this explainer can handle the model and masker
        """
        return isinstance(model, (tf.keras.Model, tuple)) and masker is None

    def _validate_inputs(self, X: Union[np.ndarray, tf.Tensor, List]) -> List[tf.Tensor]:
        """Validate and convert input data.
        
        Parameters
        ----------
        X : array-like
            Input data to validate
            
        Returns
        -------
        List[tf.Tensor]
            Validated and converted inputs
            
        Raises
        ------
        ValueError
            If inputs are invalid
        """
        if not self.multi_input:
            if isinstance(X, list):
                raise ValueError("Expected a single tensor model input!")
            X = [X]
        else:
            if not isinstance(X, list):
                raise ValueError("Expected a list of model inputs!")
            
        return [tf.convert_to_tensor(x) for x in X]

    def remove_attributes(self, model: tf.keras.Model) -> None:
        """Remove temporary attributes from model layers.
        
        Parameters
        ----------
        model : tf.keras.Model
            Model to clean up
        """
        for layer in model.layers:
            if hasattr(layer, 'x'):
                delattr(layer, 'x')
            if hasattr(layer, 'y'):
                delattr(layer, 'y')
            # Handle nested models
            if hasattr(layer, 'layers'):
                self.remove_attributes(layer)

    def add_interim_values(self, layer: tf.keras.layers.Layer) -> None:
        """Save interim tensors during forward pass.
        
        Parameters
        ----------
        layer : tf.keras.layers.Layer
            Layer to track
        """
        original_call = layer.call
        
        def call_wrapper(inputs, *args, **kwargs):
            # Save input
            setattr(layer, 'x', inputs)
            # Get output
            outputs = original_call(inputs, *args, **kwargs)
            # Save output
            setattr(layer, 'y', outputs)
            return outputs
        
        layer.call = call_wrapper

    def get_target_input(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> None:
        """Save input tensors for layer attribution.
        
        Parameters
        ----------
        inputs : tf.Tensor or List[tf.Tensor]
            Input tensors to save
        """
        if isinstance(inputs, list):
            return inputs[0]
        return inputs

    def wrap_model_layers(self, model: tf.keras.Model) -> None:
        """Wrap all relevant layers in the model for gradient handling.
        
        Parameters
        ----------
        model : tf.keras.Model
            Model to wrap layers for
        """
        def is_container(layer):
            return isinstance(layer, (
                tf.keras.Sequential,
                tf.keras.Model
            ))
        
        wrapped_layers = []
        for layer in model.layers:
            if is_container(layer):
                # Recursively wrap nested model layers
                self.wrap_model_layers(layer)
                wrapped_layers.append(layer)
            else:
                # Skip layers that don't need gradient handling
                if layer.__class__.__name__ in op_handlers:
                    wrapped_layer = LayerWrapper(layer)
                    wrapped_layers.append(wrapped_layer)
                else:
                    wrapped_layers.append(layer)
        
        # Rebuild model with wrapped layers
        model._layers = wrapped_layers

    def _prepare_model(self) -> None:
        """Prepare model for DeepLIFT attribution."""
        # Wrap layers for gradient handling
        self.wrap_model_layers(self.model)
        
        # Set model to inference mode
        self.model.trainable = False
        
        # Clear any existing state
        tf.keras.backend.clear_session()

    def _validate_background_data(self, data: Union[np.ndarray, tf.Tensor, Callable]) -> None:
        """Validate and prepare background data.
        
        Parameters
        ----------
        data : array-like or callable
            Background data or function to generate it
        
        Raises
        ------
        ValueError
            If data format is invalid
        """
        if callable(data):
            sample_data = data(None)
            if not isinstance(sample_data, (list, tuple)):
                sample_data = [sample_data]
        else:
            if not isinstance(data, (list, tuple)):
                data = [data]
            sample_data = data
        
        # Convert to tensors
        sample_data = [tf.convert_to_tensor(d) for d in sample_data]
        
        # Validate shapes match model input
        model_input_shapes = [input.shape[1:] for input in self.model.inputs]
        for i, (data_shape, input_shape) in enumerate(zip(
            [d.shape[1:] for d in sample_data],
            model_input_shapes
        )):
            if data_shape != input_shape:
                raise ValueError(
                    f"Background data shape {data_shape} does not match "
                    f"model input shape {input_shape} for input {i}"
                )


def tensors_blocked_by_false(ops):
    """ Follows a set of ops assuming their value is False and find blocked Switch paths.

    This is used to prune away parts of the model graph that are only used during the training
    phase (like dropout, batch norm, etc.).
    """
    blocked = []
    def recurse(op):
        if op.type == "Switch":
            blocked.append(op.outputs[1]) # the true path is blocked since we assume the ops we trace are False
        else:
            for out in op.outputs:
                for c in out.consumers():
                    recurse(c)
    for op in ops:
        recurse(op)

    return blocked

def backward_walk_ops(start_ops, tensor_blacklist, op_type_blacklist):
    found_ops = []
    op_stack = [op for op in start_ops]
    while len(op_stack) > 0:
        op = op_stack.pop()
        if op.type not in op_type_blacklist and op not in found_ops:
            found_ops.append(op)
            for input in op.inputs:
                if input not in tensor_blacklist:
                    op_stack.append(input.op)
    return found_ops

def forward_walk_ops(start_ops, tensor_blacklist, op_type_blacklist, within_ops):
    found_ops = []
    op_stack = [op for op in start_ops]
    while len(op_stack) > 0:
        op = op_stack.pop()
        if op.type not in op_type_blacklist and op in within_ops and op not in found_ops:
            found_ops.append(op)
            for out in op.outputs:
                if out not in tensor_blacklist:
                    for c in out.consumers():
                        op_stack.append(c)
    return found_ops


def softmax(explainer, op, *grads):
    """ Just decompose softmax into its components and recurse, we can handle all of them :)

    We assume the 'axis' is the last dimension because the TF codebase swaps the 'axis' to
    the last dimension before the softmax op if 'axis' is not already the last dimension.
    We also don't subtract the max before tf.exp for numerical stability since that might
    mess up the attributions and it seems like TensorFlow doesn't define softmax that way
    (according to the docs)
    """
    in0 = op.inputs[0]
    in0_max = tf.reduce_max(in0, axis=-1, keepdims=True, name="in0_max")
    in0_centered = in0 - in0_max
    evals = tf.exp(in0_centered, name="custom_exp")
    rsum = tf.reduce_sum(evals, axis=-1, keepdims=True)
    div = evals / rsum
    explainer.between_ops.extend([evals.op, rsum.op, div.op, in0_centered.op]) # mark these as in-between the inputs and outputs
    out = tf.gradients(div, in0_centered, grad_ys=grads[0])[0]
    del explainer.between_ops[-4:]

    # rescale to account for our shift by in0_max (which we did for numerical stability)
    xin0,rin0 = tf.split(in0, 2)
    xin0_centered,rin0_centered = tf.split(in0_centered, 2)
    delta_in0 = xin0 - rin0
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    return tf.where(
        tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
        out,
        out * tf.tile((xin0_centered - rin0_centered) / delta_in0, dup0)
    )

def maxpool(explainer, op, *grads):
    xin0,rin0 = tf.split(op.inputs[0], 2)
    xout,rout = tf.split(op.outputs[0], 2)
    delta_in0 = xin0 - rin0
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    cross_max = tf.maximum(xout, rout)
    diffs = tf.concat([cross_max - rout, xout - cross_max], 0)
    xmax_pos,rmax_pos = tf.split(explainer.orig_grads[op.type](op, grads[0] * diffs), 2)
    return tf.tile(tf.where(
        tf.abs(delta_in0) < 1e-7,
        tf.zeros_like(delta_in0),
        (xmax_pos + rmax_pos) / delta_in0
    ), dup0)

def gather(explainer, op, *grads):
    #params = op.inputs[0]
    indices = op.inputs[1]
    #axis = op.inputs[2]
    var = explainer._variable_inputs(op)
    if var[1] and not var[0]:
        assert len(indices.shape) == 2, "Only scalar indices supported right now in GatherV2!"

        xin1,rin1 = tf.split(tf.to_float(op.inputs[1]), 2)
        xout,rout = tf.split(op.outputs[0], 2)
        dup_in1 = [2] + [1 for i in xin1.shape[1:]]
        dup_out = [2] + [1 for i in xout.shape[1:]]
        delta_in1_t = tf.tile(xin1 - rin1, dup_in1)
        out_sum = tf.reduce_sum(grads[0] * tf.tile(xout - rout, dup_out), list(range(len(indices.shape), len(grads[0].shape))))
        if op.type == "ResourceGather":
            return [None, tf.where(
                tf.abs(delta_in1_t) < 1e-6,
                tf.zeros_like(delta_in1_t),
                out_sum / delta_in1_t
            )]
        return [None, tf.where(
            tf.abs(delta_in1_t) < 1e-6,
            tf.zeros_like(delta_in1_t),
            out_sum / delta_in1_t
        ), None]
    elif var[0] and not var[1]:
        return [explainer.orig_grads[op.type](op, grads[0]), None] # linear in this case
    else:
        assert False, "Axis not yet supported to be varying for gather op!"

def linearity_1d_nonlinearity_2d(input_ind0, input_ind1, op_func):
    def handler(explainer, op, *grads):
        var = explainer._variable_inputs(op)
        if var[input_ind0] and not var[input_ind1]:
            return linearity_1d_handler(input_ind0, explainer, op, *grads)
        elif var[input_ind1] and not var[input_ind0]:
            return linearity_1d_handler(input_ind1, explainer, op, *grads)
        elif var[input_ind0] and var[input_ind1]:
            return nonlinearity_2d_handler(input_ind0, input_ind1, op_func, explainer, op, *grads)
        else:
            return [None for _ in op.inputs] # no inputs vary, we must be hidden by a switch function
    return handler

def nonlinearity_1d_nonlinearity_2d(input_ind0, input_ind1, op_func):
    def handler(explainer, op, *grads):
        var = explainer._variable_inputs(op)
        if var[input_ind0] and not var[input_ind1]:
            return nonlinearity_1d_handler(input_ind0, explainer, op, *grads)
        elif var[input_ind1] and not var[input_ind0]:
            return nonlinearity_1d_handler(input_ind1, explainer, op, *grads)
        elif var[input_ind0] and var[input_ind1]:
            return nonlinearity_2d_handler(input_ind0, input_ind1, op_func, explainer, op, *grads)
        else: 
            return [None for _ in op.inputs] # no inputs vary, we must be hidden by a switch function
    return handler

def nonlinearity_1d(input_ind):
    def handler(explainer, op, *grads):
        return nonlinearity_1d_handler(input_ind, explainer, op, *grads)
    return handler

def nonlinearity_1d_handler(input_ind, explainer, op, *grads):

    # make sure only the given input varies
    for i in range(len(op.inputs)):
        if i != input_ind:
            assert not explainer._variable_inputs(op)[i], str(i) + "th input to " + op.name + " cannot vary!"
    
    xin0,rin0 = tf.split(op.inputs[input_ind], 2)
    xout,rout = tf.split(op.outputs[input_ind], 2)
    delta_in0 = xin0 - rin0
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    out = [None for _ in op.inputs]
    orig_grads = explainer.orig_grads[op.type](op, grads[0])
    out[input_ind] = tf.where(
        tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
        orig_grads[input_ind] if len(op.inputs) > 1 else orig_grads,
        grads[0] * tf.tile((xout - rout) / delta_in0, dup0)
    )
    return out

def nonlinearity_2d_handler(input_ind0, input_ind1, op_func, explainer, op, *grads):
    assert input_ind0 == 0 and input_ind1 == 1, "TODO: Can't yet handle double inputs that are not first!"
    xout,rout = tf.split(op.outputs[0], 2)
    xin0,rin0 = tf.split(op.inputs[input_ind0], 2)
    xin1,rin1 = tf.split(op.inputs[input_ind1], 2)
    delta_in0 = xin0 - rin0
    delta_in1 = xin1 - rin1
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    out10 = op_func(xin0, rin1)
    out01 = op_func(rin0, xin1)
    out11,out00 = xout,rout
    out0 = 0.5 * (out11 - out01 + out10 - out00)
    out0 = grads[0] * tf.tile(out0 / delta_in0, dup0)
    out1 = 0.5 * (out11 - out10 + out01 - out00)
    out1 = grads[0] * tf.tile(out1 / delta_in1, dup0)

    # see if due to broadcasting our gradient shapes don't match our input shapes
    if (np.any(np.array(out1.shape) != np.array(delta_in1.shape))):
        broadcast_index = np.where(np.array(out1.shape) != np.array(delta_in1.shape))[0][0]
        out1 = tf.reduce_sum(out1, axis=broadcast_index, keepdims=True)
    elif (np.any(np.array(out0.shape) != np.array(delta_in0.shape))):
        broadcast_index = np.where(np.array(out0.shape) != np.array(delta_in0.shape))[0][0]
        out0 = tf.reduce_sum(out0, axis=broadcast_index, keepdims=True)

    # Avoid divide by zero nans
    out0 = tf.where(tf.abs(tf.tile(delta_in0, dup0)) < 1e-7, tf.zeros_like(out0), out0)
    out1 = tf.where(tf.abs(tf.tile(delta_in1, dup0)) < 1e-7, tf.zeros_like(out1), out1)

    return [out0, out1]

def linearity_1d(input_ind):
    def handler(explainer, op, *grads):
        return linearity_1d_handler(input_ind, explainer, op, *grads)
    return handler

def linearity_1d_handler(input_ind, explainer, op, *grads):
    # make sure only the given input varies (negative means only that input cannot vary, and is measured from the end of the list)
    for i in range(len(op.inputs)):
        if i != input_ind:
            assert not explainer._variable_inputs(op)[i], str(i) + "th input to " + op.name + " cannot vary!"
    return explainer.orig_grads[op.type](op, *grads)

def linearity_with_excluded(input_inds):
    def handler(explainer, op, *grads):
        return linearity_with_excluded_handler(input_inds, explainer, op, *grads)
    return handler

def linearity_with_excluded_handler(input_inds, explainer, op, *grads):
    # make sure the given inputs don't vary (negative is measured from the end of the list)
    for i in range(len(op.inputs)):
        if i in input_inds or i - len(op.inputs) in input_inds:
            assert not explainer._variable_inputs(op)[i], str(i) + "th input to " + op.name + " cannot vary!"
    return explainer.orig_grads[op.type](op, *grads)

def passthrough(explainer, op, *grads):
    return explainer.orig_grads[op.type](op, *grads)

def break_dependence(explainer, op, *grads):
    """ This function name is used to break attribution dependence in the graph traversal.
     
    These operation types may be connected above input data values in the graph but their outputs
    don't depend on the input values (for example they just depend on the shape).
    """
    return [None for _ in op.inputs]


op_handlers = {}

# ops that are always linear
op_handlers["Identity"] = passthrough
op_handlers["StridedSlice"] = passthrough
op_handlers["Squeeze"] = passthrough
op_handlers["ExpandDims"] = passthrough
op_handlers["Pack"] = passthrough
op_handlers["BiasAdd"] = passthrough
op_handlers["Unpack"] = passthrough
op_handlers["Add"] = passthrough
op_handlers["Sub"] = passthrough
op_handlers["Merge"] = passthrough
op_handlers["Sum"] = passthrough
op_handlers["Mean"] = passthrough
op_handlers["Cast"] = passthrough
op_handlers["Transpose"] = passthrough
op_handlers["Enter"] = passthrough
op_handlers["Exit"] = passthrough
op_handlers["NextIteration"] = passthrough
op_handlers["Tile"] = passthrough
op_handlers["TensorArrayScatterV3"] = passthrough
op_handlers["TensorArrayReadV3"] = passthrough
op_handlers["TensorArrayWriteV3"] = passthrough

# ops that don't pass any attributions to their inputs
op_handlers["Shape"] = break_dependence
op_handlers["RandomUniform"] = break_dependence
op_handlers["ZerosLike"] = break_dependence
#op_handlers["StopGradient"] = break_dependence # this allows us to stop attributions when we want to (like softmax re-centering)

# ops that are linear and only allow a single input to vary
op_handlers["Reshape"] = linearity_1d(0)
op_handlers["Pad"] = linearity_1d(0)
op_handlers["ReverseV2"] = linearity_1d(0)
op_handlers["ConcatV2"] = linearity_with_excluded([-1])
op_handlers["Conv2D"] = linearity_1d(0)
op_handlers["Switch"] = linearity_1d(0)
op_handlers["AvgPool"] = linearity_1d(0)
op_handlers["FusedBatchNorm"] = linearity_1d(0)

# ops that are nonlinear and only allow a single input to vary
op_handlers["Relu"] = nonlinearity_1d(0)
op_handlers["Elu"] = nonlinearity_1d(0)
op_handlers["Sigmoid"] = nonlinearity_1d(0)
op_handlers["Tanh"] = nonlinearity_1d(0)
op_handlers["Softplus"] = nonlinearity_1d(0)
op_handlers["Exp"] = nonlinearity_1d(0)
op_handlers["Log"] = nonlinearity_1d(0)
op_handlers["ClipByValue"] = nonlinearity_1d(0)
op_handlers["Rsqrt"] = nonlinearity_1d(0)
op_handlers["Square"] = nonlinearity_1d(0)
op_handlers["Max"] = nonlinearity_1d(0)

# ops that are nonlinear and allow two inputs to vary
op_handlers["SquaredDifference"] = nonlinearity_1d_nonlinearity_2d(0, 1, lambda x, y: (x - y) * (x - y))
op_handlers["Minimum"] = nonlinearity_1d_nonlinearity_2d(0, 1, lambda x, y: tf.minimum(x, y))
op_handlers["Maximum"] = nonlinearity_1d_nonlinearity_2d(0, 1, lambda x, y: tf.maximum(x, y))

# ops that allow up to two inputs to vary are are linear when only one input varies
op_handlers["Mul"] = linearity_1d_nonlinearity_2d(0, 1, lambda x, y: x * y)
op_handlers["RealDiv"] = linearity_1d_nonlinearity_2d(0, 1, lambda x, y: x / y)
op_handlers["MatMul"] = linearity_1d_nonlinearity_2d(0, 1, lambda x, y: tf.matmul(x, y))

# ops that need their own custom attribution functions
op_handlers["GatherV2"] = gather
op_handlers["ResourceGather"] = gather
op_handlers["MaxPool"] = maxpool
op_handlers["Softmax"] = softmax


def add_interim_values(module, input, output):
    """The forward hook used to save interim tensors"""

def deeplift_grad(module, grad_input, grad_output):
    """The backward hook which computes the deeplift gradient"""

def tf_linear_1d(layer, grad_input, grad_output):
    """TF2 equivalent of linear_1d."""
    return grad_input

def tf_nonlinear_1d(layer, grad_input, grad_output):
    """TF2 equivalent of nonlinear_1d."""
    delta_out = layer.output[: int(layer.output.shape[0] // 2)] - layer.output[int(layer.output.shape[0] // 2):]
    delta_in = layer.input[: int(layer.input.shape[0] // 2)] - layer.input[int(layer.input.shape[0] // 2):]
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    return tf.where(
        tf.abs(tf.tile(delta_in, dup0)) < 1e-6,
        grad_input,
        grad_output * tf.tile(delta_out / delta_in, dup0)
    )

# Define operation handlers dictionary
op_handlers = {
    # Linear operations
    'Dense': tf_linear_1d,
    'Conv1D': tf_linear_1d,
    'Conv2D': tf_linear_1d,
    'Conv3D': tf_linear_1d,
    'BatchNormalization': tf_linear_1d,
    'AveragePooling1D': tf_linear_1d,
    'AveragePooling2D': tf_linear_1d,
    'AveragePooling3D': tf_linear_1d,
    
    # Nonlinear operations
    'ReLU': tf_nonlinear_1d,
    'LeakyReLU': tf_nonlinear_1d,
    'ELU': tf_nonlinear_1d,
    'Sigmoid': tf_nonlinear_1d,
    'Tanh': tf_nonlinear_1d,
    'Softmax': tf_nonlinear_1d,
    
    # Special cases
    'MaxPooling1D': tf_maxpool,
    'MaxPooling2D': tf_maxpool,
    'MaxPooling3D': tf_maxpool,
    
    # Dropout layers (passthrough)
    'Dropout': tf_passthrough,
    'SpatialDropout1D': tf_passthrough,
    'SpatialDropout2D': tf_passthrough,
    'SpatialDropout3D': tf_passthrough,
}

def tf_passthrough(layer, grad_input, grad_output):
    """No change made to gradients."""
    return grad_input

def tf_maxpool(layer, grad_input, grad_output):
    """TF2 equivalent of PyTorch's maxpool handler."""
    # Split input and reference
    x = layer.input
    y = layer.output
    delta_in = x[: x.shape[0] // 2] - x[x.shape[0] // 2:]
    
    # Create duplication shape for broadcasting
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    
    # Split output into original and reference
    y_orig, y_ref = tf.split(y, 2)
    
    # Compute cross maximum
    cross_max = tf.maximum(y_orig, y_ref)
    diffs = tf.concat([cross_max - y_ref, y_orig - cross_max], 0)
    
    # Compute positions of maxima
    with tf.GradientTape() as tape:
        tape.watch(x)
        pooled = layer(x)
    
    # Get gradients for both original and reference
    grads = tape.gradient(pooled, x) * diffs
    xmax_pos, rmax_pos = tf.split(grads, 2)
    
    # Combine gradients according to DeepLIFT rules
    return tf.where(
        tf.abs(delta_in) < 1e-7,
        tf.zeros_like(delta_in),
        (xmax_pos + rmax_pos) / delta_in
    ) * tf.cast(tf.tile([2], dup0), tf.float32)

class DeepLiftGradient:
    """Wrapper class to handle DeepLIFT gradient computation."""
    
    def __init__(self, layer=None):
        self.layer = layer
    
    def __call__(self, x):
        @tf.custom_gradient
        def grad_wrapper(x):
            def grad(dy):
                if self.layer is None:
                    return dy
                    
                layer_type = self.layer.__class__.__name__
                if layer_type in op_handlers:
                    return op_handlers[layer_type](self.layer, dy, x)
                    
                print(f'Warning: unrecognized layer type: {layer_type}; using regular gradients')
                return dy
                
            return x, grad
        return grad_wrapper(x)

def tf_threshold(layer, grad_input, grad_output):
    """TF2 equivalent of PyTorch's Threshold."""
    return tf_nonlinear_1d(layer, grad_input, grad_output)

def tf_adaptive_avgpool(layer, grad_input, grad_output):
    """TF2 equivalent of PyTorch's AdaptiveAvgPool."""
    return tf_linear_1d(layer, grad_input, grad_output)

# Update op_handlers with additional PyTorch equivalents
op_handlers.update({
    # Additional linear operations
    'ZeroPadding1D': tf_linear_1d,
    'ZeroPadding2D': tf_linear_1d,
    'ZeroPadding3D': tf_linear_1d,
    'GlobalAveragePooling1D': tf_linear_1d,
    'GlobalAveragePooling2D': tf_linear_1d,
    'GlobalAveragePooling3D': tf_linear_1d,
    
    # Additional nonlinear operations
    'ThresholdedReLU': tf_threshold,
    'PReLU': tf_nonlinear_1d,
    'Softplus': tf_nonlinear_1d,
    'GELU': tf_nonlinear_1d,
    
    # Additional special cases
    'GlobalMaxPooling1D': tf_maxpool,
    'GlobalMaxPooling2D': tf_maxpool,
    'GlobalMaxPooling3D': tf_maxpool,
})

# Track layers that need special gradient handling
failure_case_layers = [
    'MaxPooling1D',
    'GlobalMaxPooling1D'
]

class ComplexGradientRegistry:
    """Registry for handling complex gradient cases."""
    
    def __init__(self):
        self.gradients = []
    
    def append(self, grad):
        self.gradients.append(grad)
    
    def pop(self):
        return self.gradients.pop()

complex_module_gradients = ComplexGradientRegistry()

def handle_complex_gradient(layer_type: str, grad: tf.Tensor) -> tf.Tensor:
    """Handle gradients for complex layer types.
    
    Parameters
    ----------
    layer_type : str
        Type of layer
    grad : tf.Tensor
        Gradient tensor
        
    Returns
    -------
    tf.Tensor
        Processed gradient
    """
    if layer_type in failure_case_layers:
        complex_module_gradients.append(grad)
        # Apply layer-specific processing
        if layer_type == 'MaxPooling1D':
            indices = layer.get_indices()
            return tf.gather(grad, indices, axis=-1)
    return grad

class LayerWrapper(tf.keras.layers.Layer):
    """Wrapper to capture layer inputs/outputs and handle gradients."""
    
    def __init__(self, layer: tf.keras.layers.Layer):
        super().__init__()
        self.layer = layer
        self.target_input = None
        self.x = None
        self.y = None
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass with input/output capture."""
        self.x = inputs
        outputs = self.layer(inputs)
        self.y = outputs
        return outputs
    
    def get_config(self):
        return {"layer": self.layer}

"""
=== Implementation Differences ===

1. From TF1 Session-based (deep_tf_session_based.py):
   - Removed session handling and graph-based operations
   - Switched to eager execution model
   - Replaced tf.gradients with GradientTape
   - Removed learning phase flags handling
   - Simplified operator handling (no need for explicit graph traversal)
   - Removed symbolic computation graphs
   - Added batch processing capabilities

2. From Non-batched Version (deep_tf_noBatch.py):
   - Added efficient batch processing of samples
   - Implemented background tiling using tf.repeat
   - Added persistent gradient tape for batch computation
   - Optimized tensor operations with reshape and einsum
   - Added memory management with explicit cleanup
   - Improved tensor shape handling for batches

=== Current Batch Processing ===
- Processes N samples against M backgrounds efficiently
- Uses tiling/repeating for parallel computation
- Shares background data across all samples in batch
- Manages memory with explicit tensor cleanup
- Uses persistent gradient tape for gradient computation
- Employs einsum for efficient batch multiplication

=== Potential Future Optimizations ===

1. Parallel Processing for Multiple Outputs
   Benefits:
   - Faster processing for models with many outputs
   - Better GPU utilization
   Risks:
   - Higher memory usage
   - Potential GPU OOM for large models
   - More complex gradient computation

2. Adaptive Batch Sizing
   Benefits:
   - Prevents OOM errors
   - Optimizes memory usage
   - Handles varying input sizes
   Risks:
   - Overhead from memory checking
   - Potential performance variability
   - More complex implementation

3. Background Sample Batching
   Benefits:
   - Handles large background sets
   - Reduces peak memory usage
   - More flexible memory management
   Risks:
   - More complex implementation
   - Potential slowdown from multiple passes
   - Need to aggregate results carefully

4. Custom Gradient Accumulation
   Benefits:
   - Handles very large models
   - Reduces peak memory usage
   - More stable computation
   Risks:
   - Slower computation
   - More complex implementation
   - Potential numerical precision issues

Note: All optimizations should be thoroughly tested for numerical accuracy
against the original implementation, as SHAP values are sensitive to
computational differences.

=== Additional Optimization Opportunities ===

1. Gradient Computation Functions
   Current:
   - Multiple tf.tile operations
   - Repeated shape calculations
   Opportunities:
   - Combine multiple tiling operations
   - Use broadcasting instead of tiling
   - Cache repeated shape calculations
   Risks:
   - Need careful validation of shape handling
   - Potential numerical precision differences

2. Background Data Handling
   Current:
   - Reshapes background data for each batch
   - Repeats similar transformations
   Opportunities:
   - Pre-compute and cache common transformations
   - Use tf.broadcast_to instead of repeat
   - Implement lazy loading for large datasets
   Risks:
   - Memory tradeoff between caching and recomputation
   - Potential increased complexity in data management

3. Operation Handlers Dictionary
   Current:
   - Static dictionary of handlers
   - Repeated operation pattern matching
   Opportunities:
   - Lazy initialization of handlers
   - Dynamic registration system
   - Cache common operation patterns
   Risks:
   - More complex handler management
   - Potential overhead in registration system

4. Shape Inference and Validation
   Current:
   - Multiple shape checks during execution
   - Repeated shape calculations
   Opportunities:
   - Pre-compute shape information
   - Cache shape inference results
   - Better use of TF's shape inference
   Risks:
   - Dynamic shape handling complexity
   - Potential issues with unknown shapes

5. Memory Management
   Current:
   - Basic cleanup of tensors
   - Simple device placement
   Opportunities:
   - Granular tensor lifecycle management
   - Strategic device context managers
   - Better GPU memory pooling
   Risks:
   - Complex memory management bugs
   - Potential performance overhead

6. Input Validation and Preprocessing
   Current:
   - Scattered validation logic
   - Repeated type conversions
   Opportunities:
   - Centralize input validation
   - Optimize type checking
   - Cache input formats
   Risks:
   - Additional overhead for simple cases
   - Potential complexity in edge cases

7. Tensor Operations Fusion
   Current:
   - Separate operations for related computations
   - Standard TF ops usage
   Opportunities:
   - Combine operations into custom ops
   - Use XLA compilation
   - Implement specialized fused operations
   Risks:
   - More complex debugging
   - Platform-specific issues
   - Potential maintenance challenges

Note: All optimizations should be implemented incrementally with thorough
testing to ensure numerical accuracy is maintained. The complexity of
SHAP value calculations means that even small changes can have significant
impacts on results.

Note on Input Dimensionality Handling:

The _compute_gradients method needs to handle inputs of varying dimensions while maintaining
proper shape information throughout the computation pipeline. Currently:

1. For 1D inputs (e.g., [batch, features]):
   - Reshaped to [batch*backgrounds, features] for model input
   - Works with dense/fully connected layers

2. For 2D inputs (e.g., [batch, seq_len, features]):
   - Reshaped to [batch*backgrounds, seq_len, features] for model input
   - Works with convolutional and sequence models

This implementation should be tested and potentially extended for:
- 3D inputs (e.g., volumetric data)
- Mixed dimensionality inputs
- Models with multiple inputs of different dimensions
- Inputs with dynamic shapes

The critical requirements are:
1. Maintain proper dimensionality for model input
2. Preserve batch and background sample relationships
3. Correctly reshape gradients and outputs
4. Handle arbitrary feature dimensions

Future improvements should focus on making the dimensionality handling more generic
and robust to different input shapes while maintaining the efficiency of the current
implementation.
"""