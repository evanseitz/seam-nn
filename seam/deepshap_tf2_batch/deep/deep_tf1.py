import numpy as np
import warnings
from core.explainer import Explainer
from .utils import standard_combine_mult_and_diffref
from distutils.version import LooseVersion
import sys
from .deep_tf1_utils import tensors_blocked_by_false, backward_walk_ops, forward_walk_ops, passthrough, break_dependence, op_handlers
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import gradients_impl as tf_gradients_impl

keras = None
tf = None
tf_ops = None
tf_gradients_impl = None

class TFDeepExplainer(Explainer):
    """
    Using tf.gradients to implement the backgropagation was
    inspired by the gradient based implementation approach proposed by Ancona et al, ICLR 2018. Note
    that this package does not currently use the reveal-cancel rule for ReLu units proposed in DeepLIFT.
    """
    def __init__(self, model, data, session=None, learning_phase_flags=None,
                       combine_mult_and_diffref=
                        standard_combine_mult_and_diffref):
        """ An explainer object for a deep model using a given background dataset.

        Note that the complexity of the method scales linearly with the number of background data
        samples. Passing the entire training dataset as `data` will give very accurate expected
        values, but be unreasonably expensive. The variance of the expectation estimates scale by
        roughly 1/sqrt(N) for N background data samples. So 100 samples will give a good estimate,
        and 1000 samples a very good estimate of the expected values.

        Parameters
        ----------
        model : keras.Model or (input : [tf.Operation], output : tf.Operation)
            A keras model object or a pair of TensorFlow operations (or a list and an op) that
            specifies the input and output of the model to be explained. Note that SHAP values
            are specific to a single output value, so you get an explanation for each element of
            the output tensor (which must be a flat rank one vector).

        data : [numpy.array] or [pandas.DataFrame] or function
            The background dataset to use for integrating out features. DeepExplainer integrates
            over all these samples for each explanation. The data passed here must match the input
            operations given to the model. If a function is supplied, it must be a function that
            takes a particular input example and generates the background dataset for that example
        session : None or tensorflow.Session
            The TensorFlow session that has the model we are explaining. If None is passed then
            we do our best to find the right session, first looking for a keras session, then
            falling back to the default TensorFlow session.

        learning_phase_flags : None or list of tensors
            If you have your own custom learning phase flags pass them here. When explaining a prediction
            we need to ensure we are not in training mode, since this changes the behavior of ops like
            batch norm or dropout. If None is passed then we look for tensors in the graph that look like
            learning phase flags (this works for Keras models). Note that we assume all the flags should
            have a value of False during predictions (and hence explanations).

        combine_mult_and_diffref : function
            This function determines how to combine the multipliers,
             the original input and the reference input to get
             the final attributions. Defaults to
             standard_combine_mult_and_diffref, which just multiplies
             the multipliers with the difference-from-reference (in
             accordance with the standard DeepLIFT formulation) and then
             averages the importance scores across the different references.
             However, different approaches may be applied depending on
             the use case (e.g. for computing hypothetical contributions
             in genomic data)
        """

        self.combine_mult_and_diffref = combine_mult_and_diffref

        global tf, tf_ops, tf_gradients_impl
        if tf is None:
            from tensorflow.python.framework import ops as tf_ops # pylint: disable=E0611
            from tensorflow.python.ops import gradients_impl as tf_gradients_impl # pylint: disable=E0611
            if not hasattr(tf_gradients_impl, "_IsBackpropagatable"):
                from tensorflow.python.ops import gradients_util as tf_gradients_impl
            import tensorflow as tf
            if LooseVersion(tf.__version__) < LooseVersion("1.4.0"):
                warnings.warn("Your TensorFlow version is older than 1.4.0 and not supported.")
        global keras
        if keras is None:
            try:
                import keras
                if LooseVersion(keras.__version__) < LooseVersion("2.1.0"):
                    warnings.warn("Your Keras version is older than 2.1.0 and not supported.")
            except:
                pass

        # determine the model inputs and outputs
        if str(type(model)).endswith("keras.engine.sequential.Sequential'>"):
            self.model_inputs = model.inputs
            self.model_output = model.layers[-1].output
        elif str(type(model)).endswith("keras.models.Sequential'>"):
            self.model_inputs = model.inputs
            self.model_output = model.layers[-1].output
        elif str(type(model)).endswith("keras.engine.training.Model'>"):
            self.model_inputs = model.inputs
            self.model_output = model.layers[-1].output
        elif str(type(model)).endswith("tuple'>"):
            self.model_inputs = model[0]
            self.model_output = model[1]
        else:
            assert False, str(type(model)) + " is not currently a supported model type!"
        assert type(self.model_output) != list, "The model output to be explained must be a single tensor!"
        assert len(self.model_output.shape) < 3, "The model output must be a vector or a single value!"
        self.multi_output = True
        if len(self.model_output.shape) == 1:
            self.multi_output = False

        # check if we have multiple inputs
        self.multi_input = True
        if type(self.model_inputs) != list or len(self.model_inputs) == 1:
            self.multi_input = False
            if type(self.model_inputs) != list:
                self.model_inputs = [self.model_inputs]
        if type(data) != list and (hasattr(data, '__call__')==False):
            data = [data]
        self.data = data
        
        self._vinputs = {} # used to track what op inputs depends on the model inputs
        self.orig_grads = {}
        
        # if we are not given a session find a default session
        if session is None:
            try:
                if keras is not None and hasattr(keras.backend.tensorflow_backend, "_SESSION") and keras.backend.tensorflow_backend._SESSION is not None:
                    self.session = keras.backend.get_session()
                else:
                    #tf1
                    self.session=tf.keras.backend.get_session()
            except:
                #tf2
                self.session = tf.compat.v1.keras.backend.get_session()
        else:
            self.session= session
        # if no learning phase flags were given we go looking for them
        # ...this will catch the one that keras uses
        # we need to find them since we want to make sure learning phase flags are set to False
        if learning_phase_flags is None:
            self.learning_phase_ops = []
            for op in self.session.graph.get_operations():
                if 'learning_phase' in op.name and op.type == "Const" and len(op.outputs[0].shape) == 0:
                    if op.outputs[0].dtype == tf.bool:
                        print(f"Found learning phase op: {op.name}")
                        self.learning_phase_ops.append(op)
            self.learning_phase_flags = [op.outputs[0] for op in self.learning_phase_ops]
            print("Learning phase flag values:", [self.session.run(flag) for flag in self.learning_phase_flags])

        else:
            self.learning_phase_ops = [t.op for t in learning_phase_flags]

        # save the expected output of the model
        # if self.data is a function, set self.expected_value to None
        if (hasattr(self.data, '__call__')):
            self.expected_value = None
        else:
            if self.data[0].shape[0] > 5000:
                warnings.warn("You have provided over 5k background samples! For better performance consider using smaller random sample.")
            self.expected_value = self.run(self.model_output, self.model_inputs, self.data).mean(0)

        # find all the operations in the graph between our inputs and outputs
        tensor_blacklist = tensors_blocked_by_false(self.learning_phase_ops) # don't follow learning phase branches
        dependence_breakers = [k for k in op_handlers if op_handlers[k] == break_dependence]
        print("\nDependence breakers:", dependence_breakers)
        print("\nTensor blacklist:", [t.name for t in tensor_blacklist])
        
        back_ops = backward_walk_ops(
            [self.model_output.op], tensor_blacklist,
            dependence_breakers
        )
        print("\nBackward ops:", [op.type for op in back_ops])
        
        self.between_ops = forward_walk_ops(
            [op for input in self.model_inputs for op in input.consumers()],
            tensor_blacklist, dependence_breakers,
            within_ops=back_ops
        )
        print("\nForward ops:", [op.type for op in self.between_ops])
        
        # Intersect the forward and backward reachable ops
        print("\nBetween ops:", [op.type for op in self.between_ops])

        # Initialize used_types dictionary
        self.used_types = {}
        for op in self.between_ops:
            self.used_types[op.type] = True

        # Make a blank array that will get lazily filled in with the SHAP value computation
        # graphs for each output. Lazy is important since if there are 1000 outputs and we
        # only explain the top 5 it would be a waste to build graphs for the other 995

        if not self.multi_output:
            self.phi_symbolics = [None]
        else:
            noutputs = self.model_output.shape.as_list()[1]
            print("???", noutputs)
            if noutputs is not None:
                self.phi_symbolics = [None for i in range(noutputs)]
            else:
                raise Exception("The model output tensor to be explained cannot have a static shape in dim 1 of None!")

        self.debug_tensors = {}  # Add this to store debug tensors

    def run(self, out, model_inputs, X):
        """ Runs the model while also setting the learning phase flags to False.
        """
        feed_dict = dict(zip(model_inputs, X))
        for t in self.learning_phase_flags:
            feed_dict[t] = False
        result = self.session.run(out, feed_dict)
        #print(f"6. gradient first 10: {[r.flatten()[:10] for r in result]}")
        return result
        
    def _variable_inputs(self, op):
        """ Return which inputs of this operation are variable (i.e. depend on the model inputs).
        """
        print("\n=== _variable_inputs debug ===")
        print(f"Op name: {op.name}")
        print(f"Op inputs: {[t.name for t in op.inputs]}")
        print(f"Model inputs: {[x.name for x in self.model_inputs]}")
    
        if op.name not in self._vinputs:
            self._vinputs[op.name] = np.array([t.op in self.between_ops or t.name in [x.name for x in self.model_inputs] for t in op.inputs])
        print(f"Variable mask: {self._vinputs[op.name]}")
        print("=== _variable_inputs finished ===\n")
        return self._vinputs[op.name]
    
    def phi_symbolic(self, i):
        """Get the SHAP value computation graph for a given model output."""
        print(f"\n=== TF1 phi_symbolic for output {i} ===")
        
        if self.phi_symbolics[i] is None:
            print("Building new computation graph...")
            
            # Get the specific output we want to explain
            print(f"model_output: {self.model_output}")
            out = self.model_output[:,i] if self.multi_output else self.model_output
            print(f"out: {out}")
            #print(f"Output shape: {out.shape}")
            
            print("\nRegistering custom gradients...")
            # Create custom gradient registry
            reg = {}
            for n in op_handlers:
                if n in self.used_types:
                    if op_handlers[n] is not passthrough:
                        #print(f"\nRegistered custom gradient for: {n}")
                        reg[n] = self.custom_grad

            # replace the gradients for all the non-linear activations
            reg = tf_ops._gradient_registry._registry
            for n in op_handlers:
                if n in reg:
                    self.orig_grads[n] = reg[n]["type"]
                    #print(f"\nStored original gradient for {n}")
                    #print(f"Type: {type(self.orig_grads[n])}")
                    if op_handlers[n] is not passthrough:
                        reg[n]["type"] = self.custom_grad

                elif n in self.used_types:
                    raise Exception(n + " was used in the model but is not in the gradient registry!")
            # In TensorFlow 1.10 they started pruning out nodes that they think can't be backpropped
            # unfortunately that includes the index of embedding layers so we disable that check here
            if hasattr(tf_gradients_impl, "_IsBackpropagatable"):
                orig_IsBackpropagatable = tf_gradients_impl._IsBackpropagatable
                tf_gradients_impl._IsBackpropagatable = lambda tensor: True

            try:
                out = self.model_output[:,i] if self.multi_output else self.model_output

                print("\nDEBUG: Detailed gradient computation")
                print(f"1. model_output shape: {self.model_output.shape}")
                print(tf.shape(self.model_output))
                print(f"2. model_output[:,i] shape: {self.model_output[:,i].shape}")
                print(tf.shape(self.model_output[:,i]))
                print(f"3. out shape: {out.shape}")
                print(tf.shape(out))
                print(f"4. model_inputs shapes: {[x.get_shape() for x in self.model_inputs]}")
                self.phi_symbolics[i] = tf.gradients(out, self.model_inputs)
                print(tf.shape(self.phi_symbolics[i]))
                print(f"5. gradient shapes: {[g.get_shape() if g is not None else None for g in self.phi_symbolics[i]]}")
            finally:
                # reinstate the backpropagatable check
                if hasattr(tf_gradients_impl, "_IsBackpropagatable"):
                    tf_gradients_impl._IsBackpropagatable = orig_IsBackpropagatable

                # restore the original gradient definitions
                for n in op_handlers:
                    if n in reg:
                        reg[n]["type"] = self.orig_grads[n]

        # Try to get actual shape from the data
        try:
            output_shape = (self.data[0].shape[0], self.data[0].shape[1], self.data[0].shape[2])
        except:
            output_shape = self.phi_symbolics[i][0].get_shape()
        print(f"Output shape: {output_shape}")
        return self.phi_symbolics[i]

    def custom_grad(self, op, *grads):
        """ Passes a gradient op creation request to the correct handler.
        """
        return op_handlers[op.type](self, op, *grads)
    
    def test_phi_symbolic(self, X):
        """Test that TF1 implementation gives expected results."""
        print("\nTesting phi_symbolic...")
        # Check if we have multiple inputs
        if not self.multi_input:
            if type(X) == list and len(X) != 1:
                assert False, "Expected a single tensor as model input!"
            elif type(X) != list:
                X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"
        model_output_ranks = np.tile(np.arange(len(self.phi_symbolics))[None,:], (X[0].shape[0],1))
        # Compute the attributions
        all_phis = []  # Store all results
        for i in range(model_output_ranks.shape[1]):
            print(f"\n{'='*50}")
            print(f"Output {i}:")
            print(f"{'='*50}")
            phis = []
            for k in range(len(X)):
                phis.append(np.zeros(X[k].shape))
                print(f"Initialized phis[{k}] with shape: {phis[k].shape}")
                sys.stdout.flush()
                
            for j in range(X[0].shape[0]):
                print(f"\n{'-'*40}")
                print(f"Sample {j}:")
                print(f"{'-'*40}")
                if hasattr(self.data, '__call__'):
                    bg_data = self.data([X[l][j] for l in range(len(X))])
                    if type(bg_data) != list:
                        bg_data = [bg_data]
                else:
                    bg_data = self.data
                # tile the inputs to line up with the reference data samples
                print("!!! bg_data shapes:", [b.shape for b in bg_data])

                tiled_X = [np.tile(X[l][j:j+1], (bg_data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape)-1)])) for l in range(len(X))]
                print("!!! tiled_X shapes:", [t.shape for t in tiled_X])

                joint_input = [np.concatenate([tiled_X[l], bg_data[l]], 0) for l in range(len(X))]
                print("!!! joint_input shapes:", [j.shape for j in joint_input])

                # run attribution computation graph
                feature_ind = model_output_ranks[j,i]
                print(f"Feature index: {feature_ind}")
                sys.stdout.flush()

                sample_phis = self.run(self.phi_symbolic(feature_ind), self.model_inputs, joint_input)

                print(f"Sample phis first 10: {[s.flatten()[:10] if s is not None else None for s in sample_phis]}")
                sys.stdout.flush()

                # Store the joint input for debug tensor evaluation
                self._last_joint_input = joint_input
                
                # After all gradients are computed
                print("\nComputation complete, evaluating debug tensors...")
                self.evaluate_debug_tensors()
                
                all_phis.append(sample_phis)  # Store results
                
        return all_phis  # Return all results for comparison
    

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max"):
        """Estimate the SHAP values for a batch of samples."""
        #print("\n=== Computing SHAP values (TF1) ===")
        
        # Check if we have multiple inputs
        if not self.multi_input:
            if type(X) == list and len(X) != 1:
                assert False, "Expected a single tensor as model input!"
            elif type(X) != list:
                X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"
        assert len(self.model_inputs) == len(X), "Number of model inputs (%d) does not match the number given (%d)!" % (len(self.model_inputs), len(X))
        #print(f"Input shapes: {[x.shape for x in X]}")
        # Rank and determine the model outputs that we will explain
        if ranked_outputs is not None and self.multi_output:
            #print("\nRanking outputs...")
            model_output_values = self.run(self.model_output, self.model_inputs, X)
            #print(f"Model output values shape: {model_output_values.shape}")
            
            if output_rank_order == "max":
                model_output_ranks = np.argsort(-model_output_values)[:,:ranked_outputs]
            elif output_rank_order == "min":
                model_output_ranks = np.argsort(model_output_values)[:,:ranked_outputs]
            elif output_rank_order == "max_abs":
                model_output_ranks = np.argsort(-np.abs(model_output_values))[:,:ranked_outputs]
            else:
                assert False, "output_rank_order must be max, min, or max_abs!"
            #print(f"Ranked outputs shape: {model_output_ranks.shape}")
        else:
            model_output_ranks = np.tile(np.arange(len(self.phi_symbolics))[None,:], (X[0].shape[0],1))
            #print(f"Using all outputs: {model_output_ranks.shape}")

        # Compute the attributions
        output_phis = []
        #print("\nComputing attributions...")
        for i in range(model_output_ranks.shape[1]):
            #print(f"\nOutput {i}:")
            phis = []
            for k in range(len(X)):
                phis.append(np.zeros(X[k].shape))
                #print(f"Initialized phis[{k}] with shape: {phis[k].shape}")
                
            for j in range(X[0].shape[0]):
                #print(f"\n  Sample {j}:")
                if hasattr(self.data, '__call__'):
                    bg_data = self.data([X[l][j] for l in range(len(X))])
                    if type(bg_data) != list:
                        bg_data = [bg_data]
                else:
                    bg_data = self.data

                tiled_X = [np.tile(X[l][j:j+1], (bg_data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape)-1)])) for l in range(len(X))]                
                #print(f"  Tiled X shapes: {[t.shape for t in tiled_X]}")
                
                joint_input = [np.concatenate([tiled_X[l], bg_data[l]], 0) for l in range(len(X))]
                #print(f"  Joint input shapes: {[j.shape for j in joint_input]}")

                # run attribution computation graph
                feature_ind = model_output_ranks[j,i]
                #print(f"  Feature index: {feature_ind}")

                sample_phis = self.run(self.phi_symbolic(feature_ind), self.model_inputs, joint_input)
                #print(f"  Sample phis shapes: {[s.shape if s is not None else None for s in sample_phis]}")

                # Combine the multipliers with the difference from reference
                # to get the final attributions
                phis_j = self.combine_mult_and_diffref(
                    mult=[sample_phis[l][:-bg_data[l].shape[0]]
                          for l in range(len(X))],
                    orig_inp=[X[l][j] for l in range(len(X))],
                    bg_data=bg_data)

                # assign the attributions to the right part of the output arrays
                for l in range(len(X)):
                    phis[l][j] = phis_j[l] 

            output_phis.append(phis[0] if not self.multi_input else phis)
            #print(f"Output {i} final shapes: {[o.shape for o in (phis if self.multi_input else [phis[0]])]}")

        #print("\nFinal results:")
        if not self.multi_output:
            #print("Single output shape:", output_phis[0].shape if not self.multi_input else [o.shape for o in output_phis[0]])
            return output_phis[0]
        elif ranked_outputs is not None:
            #print(f"Ranked outputs: {len(output_phis)} outputs")
            #print(f"Ranks shape: {model_output_ranks.shape}")
            return output_phis, model_output_ranks
        else:
            #print(f"All outputs: {len(output_phis)} outputs")
            return output_phis

    def evaluate_debug_tensors(self):
        """Evaluate stored debug tensors using the session."""
        if not hasattr(self, 'model_inputs') or not hasattr(self, '_last_joint_input'):
            print("WARNING: Missing required input data for tensor evaluation")
            return
        
        print("\nEvaluating debug tensors:")
        for op_name, tensors in self.debug_tensors.items():
            print(f"\nDebug values for {op_name}:")
            try:
                # Create feed_dict with the joint input values
                feed_dict = {}
                if isinstance(self.model_inputs, list):
                    for inp, joint in zip(self.model_inputs, self._last_joint_input):
                        feed_dict[inp] = joint
                else:
                    feed_dict[self.model_inputs] = self._last_joint_input[0]
                
                # Evaluate tensors with feed_dict
                for k, v in tensors.items():
                    try:
                        if isinstance(v, (int, float, bool)):  # Handle numeric values directly
                            print(f"{k}: {v}")
                        elif hasattr(v, 'eval'):  # It's a TF tensor
                            evaluated = self.session.run(v, feed_dict=feed_dict)
                            if hasattr(evaluated, 'shape'):
                                print(f"{k} shape: {evaluated.shape}")
                            print(f"{k}: {evaluated}")
                        else:
                            print(f"{k}: {v} (type: {type(v)})")
                    except Exception as e:
                        print(f"Error evaluating {k}: {e}")
            except Exception as e:
                print(f"Error evaluating tensors for {op_name}: {e}")
                print("Feed dict keys:", list(feed_dict.keys()))
                print("Feed dict values shapes:", [v.shape for v in feed_dict.values()])


    def test_gradient_flow(self):
        """Test function to trace gradient flow through the model."""
        print("\n=== Testing gradient flow in TF1 ===")
        
        # Store original gradient registry
        reg = tf_ops._gradient_registry._registry
        orig_registry = {}
        for n in op_handlers:
            if n in reg:
                orig_registry[n] = reg[n]["type"]
        
        # Add debug tensor collection to custom_grad
        def debug_custom_grad(op, *grads):
            # Store tensors we want to evaluate later
            self.debug_tensors[op.name] = {
                "op_type": op.type,
                "input_tensors": op.inputs,
                "output_tensors": op.outputs,
                "input_grads": grads[0] if grads else None
            }
            
            # Get the handler result using original gradient function
            if op.type in self.orig_grads:
                orig_result = self.orig_grads[op.type](op, *grads)
                self.debug_tensors[op.name]["orig_grads"] = orig_result
                
            # Now get our custom handler result
            result = op_handlers[op.type](self, op, *grads)
            
            # Store the output gradients
            if isinstance(result, (list, tuple)):
                self.debug_tensors[op.name]["output_grads"] = result
            else:
                self.debug_tensors[op.name]["output_grads"] = [result]
                
            return result
        
        # Clear existing debug tensors
        self.debug_tensors = {}
        
        # Register our debug custom_grad
        for n in op_handlers:
            if n in reg and op_handlers[n] is not passthrough:
                reg[n]["type"] = debug_custom_grad
        
        try:
            # Clear any existing graph for this output
            self.phi_symbolics[0] = None
            
            # Build and run the graph
            phi_symbolic_output = self.phi_symbolic(0)
            result = self.run(phi_symbolic_output, self.model_inputs, self._last_joint_input)
            
            # Now evaluate all the debug tensors
            self.evaluate_debug_tensors()
            
            print("\nFinal gradients:", [r.flatten()[:5] if r is not None else None for r in result])
        finally:
            # Restore original gradient registry
            for n, grad_fn in orig_registry.items():
                if n in reg:
                    reg[n]["type"] = grad_fn
        
        return result