import numpy as np
import warnings
from core.explainer import Explainer
from .utils import standard_combine_mult_and_diffref
from distutils.version import LooseVersion
import sys
from .deep_tf_utils import tensors_blocked_by_false, backward_walk_ops, forward_walk_ops, passthrough, break_dependence, op_handlers

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
                        self.learning_phase_ops.append(op)
            self.learning_phase_flags = [op.outputs[0] for op in self.learning_phase_ops]
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
        back_ops = backward_walk_ops(
            [self.model_output.op], tensor_blacklist,
            dependence_breakers
        )
        self.between_ops = forward_walk_ops(
            [op for input in self.model_inputs for op in input.consumers()],
            tensor_blacklist, dependence_breakers,
            within_ops=back_ops
        )

        # save what types are being used
        self.used_types = {}
        for op in self.between_ops:
            self.used_types[op.type] = True
            if (op.type not in op_handlers):
                print("Warning: ",op.type,"used in model but handling of op"
                      +" is not specified by shap; will use original "
                      +" gradients")

        # make a blank array that will get lazily filled in with the SHAP value computation
        # graphs for each output. Lazy is important since if there are 1000 outputs and we
        # only explain the top 5 it would be a waste to build graphs for the other 995
        if not self.multi_output:
            self.phi_symbolics = [None]
        else:
            noutputs = self.model_output.shape.as_list()[1]
            if noutputs is not None:
                self.phi_symbolics = [None for i in range(noutputs)]
            else:
                raise Exception("The model output tensor to be explained cannot have a static shape in dim 1 of None!")
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
            print("\n=== phi_symbolic Debug (sess) ===")
            print("Computing symbolic gradients for output", i)

            # replace the gradients for all the non-linear activations
            # we do this by hacking our way into the registry (TODO: find a public API for this if it exists)
            reg = tf_ops._gradient_registry._registry
            print("\nReplacing activation gradients:")
            for n in op_handlers:
                if n in reg:
                    print(f"- Replacing gradient for {n}")
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
                print("\nDefining computation graph:")
                print("- Model output shape:", self.model_output.shape)
                out = self.model_output[:,i] if self.multi_output else self.model_output
                print("- Selected output shape:", out.shape)
                print("- Model inputs shape:", [x.shape for x in self.model_inputs])
                self.phi_symbolics[i] = tf.gradients(out, self.model_inputs)
                print("- Gradient shape:", [x.shape for x in self.phi_symbolics[i]])


            finally:

                # reinstate the backpropagatable check
                print("\nRestoring original gradients")
                if hasattr(tf_gradients_impl, "_IsBackpropagatable"):
                    tf_gradients_impl._IsBackpropagatable = orig_IsBackpropagatable

                # restore the original gradient definitions
                for n in op_handlers:
                    if n in reg:
                        reg[n]["type"] = self.orig_grads[n]
        return self.phi_symbolics[i]

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max",
                             progress_message=None):
        # check if we have multiple inputs
        if not self.multi_input:
            if type(X) == list and len(X) != 1:
                assert False, "Expected a single tensor as model input!"
            elif type(X) != list:
                X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"
        assert len(self.model_inputs) == len(X), "Number of model inputs (%d) does not match the number given (%d)!" % (len(self.model_inputs), len(X))

        # rank and determine the model outputs that we will explain
        if ranked_outputs is not None and self.multi_output:
            model_output_values = self.run(self.model_output, self.model_inputs, X)
            if output_rank_order == "max":
                model_output_ranks = np.argsort(-model_output_values)
            elif output_rank_order == "min":
                model_output_ranks = np.argsort(model_output_values)
            elif output_rank_order == "max_abs":
                model_output_ranks = np.argsort(np.abs(model_output_values))
            else:
                assert False, "output_rank_order must be max, min, or max_abs!"
            model_output_ranks = model_output_ranks[:,:ranked_outputs]
        else:
            model_output_ranks = np.tile(np.arange(len(self.phi_symbolics)), (X[0].shape[0], 1))

        # compute the attributions
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            for k in range(len(X)):
                phis.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):
                if (progress_message is not None):
                    if ((j%progress_message)==0):
                        print("Done",j,"examples of", X[0].shape[0])
                        sys.stdout.flush()
                if (hasattr(self.data, '__call__')):
                    bg_data = self.data([X[l][j] for l in range(len(X))])
                    if type(bg_data) != list:
                        bg_data = [bg_data]
                else:
                    bg_data = self.data

                # tile the inputs to line up with the reference data samples
                tiled_X = [np.tile(X[l][j:j+1], (bg_data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape)-1)])) for l in range(len(X))]
                joint_input = [np.concatenate([tiled_X[l], bg_data[l]], 0) for l in range(len(X))]

                # Inside shap_values method
                print("\n=== DeepSHAP Session Debug ===")
                print("Input shapes:")
                print("- Model input:", [x.shape for x in X])
                print("- Background data:", [x.shape for x in self.data])

                # Get model output using session
                model_output = self.run(self.model_output, self.model_inputs, [X[0]])
                print("\nModel outputs:")
                print("- Shape:", model_output.shape)
                print("- First few values:", model_output[:5])

                # run attribution computation graph
                feature_ind = model_output_ranks[j,i]
                sample_phis = self.run(self.phi_symbolic(feature_ind), self.model_inputs, joint_input)
                
                # After gradient computation
                print("\nGradient computation:")
                print("- Shape:", [g.shape for g in sample_phis])
                print("- First few values:", [g[:2, :5] for g in sample_phis])

                # combine the multipliers with the difference from reference
                print("\n=== Debug sess.py values ===")
                print(f"mult first few values: {sample_phis[0][:-bg_data[0].shape[0]].flatten()[:5]}")
                print(f"orig_inp first few values: {X[0][j].flatten()[:5]}")
                print(f"bg_data first few values: {bg_data[0].flatten()[:5]}")
                phis_j = self.combine_mult_and_diffref(
                    mult=[sample_phis[l][:-bg_data[l].shape[0]]
                          for l in range(len(X))],
                    orig_inp=[X[l][j] for l in range(len(X))],
                    bg_data=bg_data)

                print("\nFinal attribution shape:", [p.shape for p in phis_j])
                print("First few attribution values:", 
                      [p[:5] if len(p.shape) == 1 else p[:5,0] for p in phis_j])
                print("=== End of Session-based Debug ===")

                # After gradient computation
                print("\nGradient computation results:")
                print("Gradient shapes:", [g.shape for g in sample_phis])
                print("First few gradient values:", 
                      [g[:2, :5] if isinstance(g, np.ndarray) else g[:2, :5].numpy() for g in sample_phis])

                # After combine_mult_and_diffref
                print("\n=== Final Attribution Values ===")
                print("Attribution shapes:", [p.shape for p in phis_j])
                print("First few attribution values:", 
                      [p[:5] if len(p.shape) == 1 else p[:5,0] for p in phis_j])

                # assign the attributions to the right part of the output arrays
                for l in range(len(X)):
                    phis[l][j] = phis_j[l]

            output_phis.append(phis[0] if not self.multi_input else phis)

        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis

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