import sys
import numpy as np
import warnings
from core.explainer import Explainer
from deep.utils import standard_combine_mult_and_diffref
from distutils.version import LooseVersion
torch = None


class PyTorchDeepExplainer(Explainer):

    def __init__(self, model, data,
                  combine_mult_and_diffref=standard_combine_mult_and_diffref):
        """
        Parameters
        ----------
        model : an nn.Module object (model), or a tuple (model, layer), where
            both are nn.Module objects.
            The model is an nn.Module object which takes as input a tensor (or list of tensors) of
            shape data, and returns an output of shape (B, 1), where B is the batch dimension.
            Note that the second dimension must be 1. Otherwise, the gradient cannot be computed
            (since the output won't be a scalar).
            If the input is a tuple, the returned shap values will be for the input of the
            layer argument. layer must be a layer in the model, i.e. model.conv2


        data : torch tensor or function  
            If a function is supplied, it must be a function that
            takes a particular input example and generates the background
            dataset for that example. When None is supplied as an input, 
            the function should return values that are in the same format
            as what it would return if an actual input were supplied.

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
        # try and import pytorch
        global torch
        if torch is None:
            import torch
            if LooseVersion(torch.__version__) < LooseVersion("0.4"):
                warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        # check if we have multiple inputs
        self.multi_input = False
        if (hasattr(data, '__call__')):
            sample_data = data(None)
        else:
            sample_data = data
            if type(data) != list:
                data = [data]
        if type(sample_data) == list:
            self.multi_input = True
        if type(sample_data) != list:
            sample_data = [sample_data]
        self.data = data
        self.combine_mult_and_diffref = combine_mult_and_diffref
        self.layer = None
        self.input_handle = None
        self.interim = False
        self.interim_inputs_shape = None
        self.expected_value = None  # to keep the DeepExplainer base happy
        if type(model) == tuple:
            self.interim = True
            model, layer = model
            model = model.eval()
            self.layer = layer
            self.add_target_handle(self.layer)

            # if we are taking an interim layer, the 'data' is going to be the input
            # of the interim layer; we will capture this using a forward hook
            with torch.no_grad():
                _ = model(*sample_data)
                interim_inputs = self.layer.target_input
                if type(interim_inputs) is tuple:
                    # this should always be true, but just to be safe
                    self.interim_inputs_shape = [i.shape for i in interim_inputs]
                else:
                    self.interim_inputs_shape = [interim_inputs.shape]
            self.target_handle.remove()
            del self.layer.target_input
        self.model = model.eval()

        self.multi_output = False
        self.num_outputs = 1
        with torch.no_grad():
            outputs = model(*sample_data)

            # also get the device everything is running on
            self.device = outputs.device

            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]
            # Only set the value of self.expected_value if data
            # is not a callable, because otherwise the expected value
            # is example-dependent.
            if (hasattr(data, '__call__')==False):
                self.expected_value = outputs.mean(0).cpu().numpy()
            
        sample_data = [d.requires_grad_() for d in data[0:1]] # debug

    def add_target_handle(self, layer):
        input_handle = layer.register_forward_hook(get_target_input)
        self.target_handle = input_handle

    def add_handles(self, model, forward_handle, backward_handle):
        """
        Add handles to all non-container layers in the model.
        Recursively for non-container layers
        """
        handles_list = []
        for child in model.children():
            if 'nn.modules.container' in str(type(child)):
                handles_list.extend(self.add_handles(child, forward_handle, backward_handle))
            else:
                handles_list.append(child.register_forward_hook(forward_handle))
                #handles_list.append(child.register_backward_hook(backward_handle))
                handles_list.append(child.register_full_backward_hook(backward_handle))
        return handles_list

    def remove_attributes(self, model):
        """
        Removes the x and y attributes which were added by the forward handles
        Recursively searches for non-container layers
        """
        for child in model.children():
            if 'nn.modules.container' in str(type(child)):
                self.remove_attributes(child)
            else:
                try:
                    del child.x
                except AttributeError:
                    pass
                try:
                    del child.y
                except AttributeError:
                    pass

    def gradient(self, idx, inputs):
        self.model.zero_grad()
        X = [x.requires_grad_() for x in inputs]
        
        # Forward pass through the model
        outputs = self.model(*X)
        
        # Handle both single-output and multi-output cases
        if len(outputs.shape) > 1:
            selected = [val for val in outputs[:, idx]]
        else:
            selected = [val for val in outputs]  # Already single output
        
        # Compute gradients
        # Shape is [N+M, D] where:
        #   N = number of times input is repeated (one for each background comparison)
        #   M = number of background samples
        #   D = number of input features
        # First N rows contain gradients for the repeated input
        # Last M rows contain gradients for the background samples
        # This allows computing attribution between each (input, background) pair
        raw_grads = [torch.autograd.grad(selected, x, retain_graph=True)[0] for x in X]
        
        if self.interim:
            interim_inputs = self.layer.target_input
            grads = [torch.autograd.grad(selected, input, retain_graph=True)[0].cpu().numpy() for input in interim_inputs]
            del self.layer.target_input
            return grads, [i.detach().cpu().numpy() for i in interim_inputs]
        else:
            grads = [g.cpu().numpy() for g in raw_grads]
            
            return grads

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max",
                             progress_message=None):
        
        # X ~ self.model_input
        # X_data ~ self.data

        # check if we have multiple inputs
        if not self.multi_input:
            assert type(X) != list, "Expected a single tensor model input!"
            X = [X]
        else:
            assert type(X) == list, "Expected a list of model inputs!"

        X = [x.to(self.device) for x in X]

        if ranked_outputs is not None and self.multi_output:
            with torch.no_grad():
                model_output_values = self.model(*X)
            # rank and determine the model outputs that we will explain
            if output_rank_order == "max":
                _, model_output_ranks = torch.sort(model_output_values, descending=True)
            elif output_rank_order == "min":
                _, model_output_ranks = torch.sort(model_output_values, descending=False)
            elif output_rank_order == "max_abs":
                _, model_output_ranks = torch.sort(torch.abs(model_output_values), descending=True)
            else:
                assert False, "output_rank_order must be max, min, or max_abs!"
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            model_output_ranks = (torch.ones((X[0].shape[0], self.num_outputs)).int() *
                                  torch.arange(0, self.num_outputs).int())

        # add the gradient handles
        handles = self.add_handles(self.model, add_interim_values, deeplift_grad)
        if self.interim:
            self.add_target_handle(self.layer)

        # compute the attributions
        output_phis = []
        for i in range(model_output_ranks.shape[1]):
            phis = []
            if self.interim:
                for k in range(len(self.interim_inputs_shape)):
                    phis.append(np.zeros((X[0].shape[0], ) + self.interim_inputs_shape[k][1: ]))
            else:
                for k in range(len(X)):
                    phis.append(np.zeros(X[k].shape))
            for j in range(X[0].shape[0]):
                if (progress_message is not None):
                    if ((j%progress_message)==0):
                        print("Done",j,"examples of",
                              X[0].shape[0])
                        sys.stdout.flush()

                if (hasattr(self.data, '__call__')):
                    bg_data = self.data([X[l][j] for l in range(len(X))])
                    if type(bg_data) != list:
                        bg_data = [bg_data]
                else:
                    bg_data = self.data
                
                # tile the inputs to line up with the background data samples
                tiled_X = [X[l][j:j + 1].repeat(
                               (bg_data[l].shape[0],) + tuple([1 for k in range(len(X[l].shape) - 1)])) for l
                           in range(len(X))]
                
                joint_x = [torch.cat((tiled_X[l], bg_data[l]), dim=0) for l in range(len(X))]
                
                # run attribution computation graph
                feature_ind = model_output_ranks[j, i]
                sample_phis = self.gradient(feature_ind, joint_x)
                # assign the attributions to the right part of the output arrays
                if self.interim:
                    sample_phis, output = sample_phis
                    x, data = [], []
                    for i in range(len(output)):
                        x_temp, data_temp = np.split(output[i], 2)
                        x.append(x_temp)
                        data.append(data_temp)
                    phis_j = self.combine_mult_and_diffref(
                        mult=[sample_phis[l][:-bg_data[l].shape[0]]
                              for l in range(len(self.interim_inputs_shape))],
                        orig_inp=x,
                        bg_data=data)
                    for l in range(len(self.interim_inputs_shape)):
                        phis[l][j] = phis_j[l] 
                else:
                    #combine the multipliers with the difference from reference
                    # to get the final attributions
                    phis_j = self.combine_mult_and_diffref(
                        mult=[sample_phis[l][:-bg_data[l].shape[0]]
                              for l in range(len(X))],
                        orig_inp=[X[l][j].detach().cpu().numpy()
                                  for l in range(len(X))],
                        bg_data=[x.detach().cpu().numpy() for x in bg_data])
                    # assign the attributions to the right part of the output arrays
                    for l in range(len(X)):
                        phis[l][j] = phis_j[l] 
            output_phis.append(phis[0] if not self.multi_input else phis)
        # cleanup; remove all gradient handles
        for handle in handles:
            handle.remove()
        self.remove_attributes(self.model)
        if self.interim:
            self.target_handle.remove()

        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis

def deeplift_grad(module, grad_input, grad_output):
    """The backward hook which computes the deeplift
    gradient for an nn.Module
    """
    # first, get the module type
    module_type = module.__class__.__name__
    # first, check the module is supported
    if module_type in op_handler:
        if op_handler[module_type].__name__ not in ['passthrough', 'linear_1d']:
            return op_handler[module_type](module, grad_input, grad_output)
    else:
        print('Warning: unrecognized nn.Module: {}'.format(module_type)
              +'; using regular gradients')
        return grad_input


def add_interim_values(module, input, output):
    """The forward hook used to save interim tensors, detached
    from the graph. Used to calculate the multipliers
    """
    try:
        del module.x
    except AttributeError:
        pass
    try:
        del module.y
    except AttributeError:
        pass
    module_type = module.__class__.__name__
    if module_type in op_handler:
        func_name = op_handler[module_type].__name__
        # First, check for cases where we don't need to save the x and y tensors
        if func_name == 'passthrough':
            pass
        else:
            # check only the 0th input varies
            for i in range(len(input)):
                if i != 0 and type(output) is tuple:
                    assert input[i] == output[i], "Only the 0th input may vary!"
            # if a new method is added, it must be added here too. This ensures tensors
            # are only saved if necessary
            if func_name in ['maxpool', 'nonlinear_1d']:
                # only save tensors if necessary
                if type(input) is tuple:
                    setattr(module, 'x', torch.nn.Parameter(input[0].detach()))
                else:
                    setattr(module, 'x', torch.nn.Parameter(input.detach()))
                if type(output) is tuple:
                    setattr(module, 'y', torch.nn.Parameter(output[0].detach()))
                else:
                    setattr(module, 'y', torch.nn.Parameter(output.detach()))
            if module_type in failure_case_modules:
                input[0].register_hook(deeplift_tensor_grad)


def get_target_input(module, input, output):
    """A forward hook which saves the tensor - attached to its graph.
    Used if we want to explain the interim outputs of a model
    """
    try:
        del module.target_input
    except AttributeError:
        pass
    setattr(module, 'target_input', input)

# From the documentation: "The current implementation will not have the presented behavior for
# complex Module that perform many operations. In some failure cases, grad_input and grad_output
# will only contain the gradients for a subset of the inputs and outputs.
# The tensor hook below handles such failure cases (currently, MaxPool1d). In such cases, the deeplift
# grad should still be computed, and then appended to the complex_model_gradients list. The tensor hook
# will then retrieve the proper gradient from this list.


failure_case_modules = ['MaxPool1d']


def deeplift_tensor_grad(grad):
    return_grad = complex_module_gradients[-1]
    del complex_module_gradients[-1]
    return return_grad


complex_module_gradients = []


def passthrough(module, grad_input, grad_output):
    """No change made to gradients"""
    return None


def maxpool(module, grad_input, grad_output):
    pool_to_unpool = {
        'MaxPool1d': torch.nn.functional.max_unpool1d,
        'MaxPool2d': torch.nn.functional.max_unpool2d,
        'MaxPool3d': torch.nn.functional.max_unpool3d
    }
    pool_to_function = {
        'MaxPool1d': torch.nn.functional.max_pool1d,
        'MaxPool2d': torch.nn.functional.max_pool2d,
        'MaxPool3d': torch.nn.functional.max_pool3d
    }
    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2):]
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # we also need to check if the output is a tuple
    y, ref_output = torch.chunk(module.y, 2)
    cross_max = torch.max(y, ref_output)
    diffs = torch.cat([cross_max - ref_output, y - cross_max], 0)

    # all of this just to unpool the outputs
    with torch.no_grad():
        _, indices = pool_to_function[module.__class__.__name__](
            module.x, module.kernel_size, module.stride, module.padding,
            module.dilation, module.ceil_mode, True)
        xmax_pos, rmax_pos = torch.chunk(pool_to_unpool[module.__class__.__name__](
            grad_output[0] * diffs, indices, module.kernel_size, module.stride,
            module.padding, list(module.x.shape)), 2)
    grad_input = [None for _ in grad_input]
    grad_input[0] = torch.where(torch.abs(delta_in) < 1e-7, torch.zeros_like(delta_in),
                           (xmax_pos + rmax_pos) / delta_in).repeat(dup0)
    if module.__class__.__name__ == 'MaxPool1d':
        complex_module_gradients.append(grad_input[0])
        grad_input[0] = torch.gather(grad_input[0], -1, indices).unsqueeze(1)
    # delete the attributes
    del module.x
    del module.y
    return tuple(grad_input)


def linear_1d(module, grad_input, grad_output):
    """No change made to gradients."""
    return None


def nonlinear_1d(module, grad_input, grad_output):
    delta_out = module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2):]

    delta_in = module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2):]
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # handles numerical instabilities where delta_in is very small by
    # just taking the gradient in those cases
    grads = [None for _ in grad_input]
    grads[0] = torch.where(torch.abs(delta_in.repeat(dup0)) < 1e-6, grad_input[0],
                           grad_output[0] * (delta_out / delta_in).repeat(dup0))

    # delete the attributes
    del module.x
    del module.y
    return tuple(grads)


op_handler = {}

# passthrough ops, where we make no change to the gradient
op_handler['Dropout3d'] = passthrough
op_handler['Dropout2d'] = passthrough
op_handler['Dropout'] = passthrough
op_handler['AlphaDropout'] = passthrough

op_handler['Conv1d'] = linear_1d
op_handler['Conv2d'] = linear_1d
op_handler['Conv3d'] = linear_1d
op_handler['ConvTranspose1d'] = linear_1d
op_handler['ConvTranspose2d'] = linear_1d
op_handler['ConvTranspose3d'] = linear_1d
op_handler['Linear'] = linear_1d
op_handler['AvgPool1d'] = linear_1d
op_handler['AvgPool2d'] = linear_1d
op_handler['AvgPool3d'] = linear_1d
op_handler['AdaptiveAvgPool1d'] = linear_1d
op_handler['AdaptiveAvgPool2d'] = linear_1d
op_handler['AdaptiveAvgPool3d'] = linear_1d
op_handler['BatchNorm1d'] = linear_1d
op_handler['BatchNorm2d'] = linear_1d
op_handler['BatchNorm3d'] = linear_1d

op_handler['LeakyReLU'] = nonlinear_1d
op_handler['ReLU'] = nonlinear_1d
op_handler['Threshold'] = nonlinear_1d
op_handler['ELU'] = nonlinear_1d
op_handler['Sigmoid'] = nonlinear_1d
op_handler["Tanh"] = nonlinear_1d
op_handler["Softplus"] = nonlinear_1d
op_handler['Softmax'] = nonlinear_1d

op_handler['MaxPool1d'] = maxpool
op_handler['MaxPool2d'] = maxpool
op_handler['MaxPool3d'] = maxpool

op_handler['PyTorchModel'] = passthrough
