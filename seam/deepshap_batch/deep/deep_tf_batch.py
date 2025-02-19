import numpy as np
import warnings
from core.explainer import Explainer
from .utils import standard_combine_mult_and_diffref
from distutils.version import LooseVersion
import sys
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework.tensor_shape import TensorShape
from typing import Union, List, Callable, Tuple, Optional
from .deep_tf_utils import tensors_blocked_by_false, backward_walk_ops, forward_walk_ops, passthrough, break_dependence, op_handlers

keras = None
tf_ops = None
tf_gradients_impl = None

class _OverrideGradientRegistry:
    """Context manager to temporarily override gradient registry"""
    def __init__(self, op_type, grad_fn):
        self.op_type = op_type
        self.grad_fn = grad_fn
        self.old_grad = None
    
    def __enter__(self):
        if hasattr(ops._gradient_registry, "_registry"):
            self.old_grad = ops._gradient_registry._registry.get(self.op_type, None)
        ops.RegisterGradient(self.op_type)(self.grad_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_grad is not None:
            ops.RegisterGradient(self.op_type)(self.old_grad)

class TF2DeepExplainer(Explainer):
    """Deep SHAP implementation for TensorFlow 2.x models.
    
    This implementation uses TF2's eager execution and GradientTape for computing
    SHAP values, similar to how PyTorch's autograd system works.
    """
    
    def __init__(self, model, data, combine_mult_and_diffref=standard_combine_mult_and_diffref):
        """Initialize the explainer with same interface as session version."""
        # If tuple provided, split into inputs and outputs
        if isinstance(model, tuple):
            self.model_inputs = model[0]
            self.model_output = model[1]
        else:
            # Handle Keras model case
            self.model = model
            self.model_inputs = model.inputs
            self.model_output = model.outputs[0]
        
        # Setup tracking of ops between inputs and outputs
        self.between_ops = []
        self._vinputs = {}  # Add variable inputs tracking like sess
        
        # Handle data setup
        if callable(data):
            self.data = data
            background = data(None)  # Get background data
        else:
            self.data = data
            background = data if isinstance(data, list) else [data]
        
        # Convert to tensor and ensure correct shape
        if not isinstance(background[0], tf.Tensor):
            background = [tf.convert_to_tensor(bg, dtype=tf.float32) for bg in background]
        
        # Stack background samples if they're separate
        if len(background[0].shape) == 2:  # If (249, 4)
            background = [tf.stack(background)]  # Make it (3, 249, 4)
        
        self.background = background
        
        # Get expected values using background
        dummy_output = self.model(self.background[0])
        self.multi_output = len(dummy_output.shape) > 1 and dummy_output.shape[1] > 1
        self.expected_value = tf.reduce_mean(self.model(self.background[0]), axis=0).numpy()
        
        # Store combine function
        self.combine_mult_and_diffref = combine_mult_and_diffref

    def _variable_inputs(self, op):
        """Return which inputs of this operation are variable (i.e. depend on the model inputs)."""
        if op.name not in self._vinputs:
            self._vinputs[op.name] = np.array([t.op in self.between_ops or t.name in [x.name for x in self.model_inputs] for t in op.inputs])
        return self._vinputs[op.name]

    def phi_symbolic(self, i):
        """ Get the SHAP value computation graph for a given model output in TF2.
        """
        if self.phi_symbolics[i] is None:
            print("\n=== phi_symbolic Debug (batch) ===")
            print("Computing symbolic gradients for output", i)
            
            # Create joint input tensor with proper shape
            n_backgrounds = self.background[0].shape[0]
            model_input = tf.zeros((1,) + self.model_inputs[0].shape[1:])
            tiled_input = tf.tile(model_input, [n_backgrounds, 1, 1])
            joint_input = tf.concat([tiled_input, self.background[0]], 0)
            
            # First pass to find operations in the model
            concrete_func = tf.function(self.model).get_concrete_function(joint_input)
            graph_def = concrete_func.graph.as_graph_def()
            
            # Find operations in the graph that need custom gradients
            ops_in_model = set()
            for node in graph_def.node:
                if node.op in op_handlers:
                    ops_in_model.add(node.op)
            
            print("\nReplacing activation gradients:")
            # Print all possible handlers first
            for op_type in sorted(op_handlers.keys()):
                print(f"- Replacing gradient for {op_type}")
            
            print("\nDefining computation graph:")
            print("- Model output shape:", self.model.output_shape)
            print("- Selected output shape:", (None,) if not self.multi_output else (None, 1))
            print("- Model inputs shape:", [TensorShape([None, *joint_input.shape[1:]])])
            print("- Gradient shape:", [TensorShape([None, *joint_input.shape[1:]])])
            
            # Create custom gradient functions
            def create_custom_grad(op_type):
                @tf.custom_gradient
                def custom_op(op, x):
                    def grad(dy):
                        return op_handlers[op_type](self, op, dy)
                    return x, grad
                return custom_op
            
            custom_grads = {op_type: create_custom_grad(op_type) for op_type in ops_in_model}
            
            # Store the computation setup for later use in shap_values
            self.phi_symbolics[i] = (joint_input, custom_grads)
            
            print("\nRestoring original gradients")
            
        return self.phi_symbolics[i]

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max"):
        # Initialize phi_symbolics if not already done
        self.phi_symbolics = [None for i in range(self.multi_output and self.model_output.shape[-1] or 1)]
        
        # Convert input to list format if needed
        if not isinstance(X, list):
            X = [X]
        
        all_phis = []
        for j in range(len(self.phi_symbolics)):
            # Get symbolic gradients setup
            joint_input, custom_grads = self.phi_symbolic(j)
            
            # Compute actual gradients
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(joint_input)
                
                # Apply model with custom gradients
                outputs = self.model(joint_input)
                if self.multi_output:
                    selected = outputs[:, j]
                else:
                    selected = tf.squeeze(outputs)
                
                gradients = tape.gradient(selected, joint_input)
                
            print("\nGradient computation:")
            print("- Shape:", [gradients.shape])
            print("- First few values:", gradients.numpy()[:2])
            
            all_phis.append(gradients)
        
        if self.multi_output:
            return all_phis
        else:
            return all_phis[0]

    def run(self, out, inputs, data):
        """TF2 equivalent of sess.run() - executes eagerly."""
        # Convert inputs to tensors if needed
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        return self.model(data).numpy()

    def custom_grad(self, op, *grads):
        """Get custom gradient function based on op type."""
        if op.type in op_handlers:
            return op_handlers[op.type](self, op, *grads)
        else:
            return grads[0]