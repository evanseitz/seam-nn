import numpy as np
import tensorflow as tf
from typing import Union, List, Callable, Tuple, Optional

# Gradient handlers for different layer types
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

def backward_walk_ops_tf2(start_ops, compute_ops, dependence_breakers):
    """Do a backwards walk from the given ops to find all connected ops.
    
    Args:
        start_ops: Operations to start walking from
        compute_ops: List of computation ops to consider
        dependence_breakers: Operation types that break attribution dependence
    """
    found_ops = []
    op_stack = [op for op in start_ops]
    
    while op_stack:
        op = op_stack.pop()
        if op.type not in dependence_breakers and op not in found_ops:
            found_ops.append(op)
            for input_tensor in op.inputs:
                for compute_op in compute_ops:
                    if compute_op.outputs and any(out.name == input_tensor.name for out in compute_op.outputs):
                        op_stack.append(compute_op)
    
    return found_ops

def forward_walk_ops_tf2(start_ops, compute_ops, dependence_breakers):
    """Do a forwards walk from the given ops to find all connected ops.
    
    Args:
        start_ops: Operations to start walking from
        compute_ops: List of computation ops to consider
        dependence_breakers: Operation types that break attribution dependence
    """
    found_ops = []
    op_stack = [op for op in start_ops]
    
    while op_stack:
        op = op_stack.pop()
        if op.type not in dependence_breakers and op not in found_ops:
            found_ops.append(op)
            for output_tensor in op.outputs:
                for compute_op in compute_ops:
                    if any(t.name == output_tensor.name for t in compute_op.inputs):
                        op_stack.append(compute_op)
    
    return found_ops

def get_model_graph(model):
    """Get the full computation graph from a Keras model."""
    # Create dummy input matching model's input shape
    dummy_input = tf.zeros([1] + list(model.input_shape[1:]))
    
    # Get concrete function
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def get_graph(x):
        return model(x)
    
    # Get the concrete function and graph
    concrete_func = get_graph.get_concrete_function(dummy_input)
    graph = concrete_func.graph
    
    # Get all operations from the graph
    return list(graph.get_operations())

def gather(explainer, op, *grads, xin0=None, rin0=None, xin1=None, rin1=None):
    """Handle gather operations with pre-split tensors."""
    var = explainer._variable_inputs(op)
    if var[1] and not var[0]:
        assert len(op.inputs[1].shape) == 2, "Only scalar indices supported right now in GatherV2!"
        
        if xin1 is None or rin1 is None:
            print("WARNING: Using fallback tensor splitting - this should not happen")
            xin1, rin1 = tf.split(tf.cast(op.inputs[1], tf.float32), 2)
        
        xout, rout = tf.split(op.outputs[0], 2)
        dup_in1 = [2] + [1 for i in xin1.shape[1:]]
        dup_out = [2] + [1 for i in xout.shape[1:]]
        delta_in1_t = tf.tile(xin1 - rin1, dup_in1)
        out_sum = tf.reduce_sum(grads[0] * tf.tile(xout - rout, dup_out), 
                               list(range(len(op.inputs[1].shape), len(grads[0].shape))))
        
        result = tf.where(
            tf.abs(delta_in1_t) < 1e-6,
            tf.zeros_like(delta_in1_t),
            out_sum / delta_in1_t
        )
        
        if op.type == "ResourceGather":
            return [None, result]
        return [None, result, None]
        
    elif var[0] and not var[1]:
        return [explainer.orig_grads[op.type](op, grads[0]), None]  # linear in this case
    else:
        assert False, "Axis not yet supported to be varying for gather op!"

### OP HANDLERS ###

def passthrough(explainer, op, *grads):
    """Pass through the gradient unchanged."""
    return grads[0]  # Just return the incoming gradient
    #return explainer.orig_grads[op.type](grads[0])

def break_dependence(explainer, op, *grads):
    """ This function name is used to break attribution dependence in the graph traversal.
     
    These operation types may be connected above input data values in the graph but their outputs
    don't depend on the input values (for example they just depend on the shape).
    """
    return [None for _ in op.inputs]

def linearity_1d(input_ind):
    def handler(explainer, op, *grads, **kwargs):
        return linearity_1d_handler(input_ind, explainer, op, *grads, **kwargs)
    return handler

def linearity_1d_handler(input_ind, explainer, op, *grads, xin0=None, rin0=None):
    """Handle linear operations with one variable input."""
    # make sure only the given input varies
    for i in range(len(op.inputs)):
        if i != input_ind:
            assert not explainer._variable_inputs(op)[i], str(i) + "th input to " + op.name + " cannot vary!"
    
    # For linear ops, just pass through the gradient
    return [grads[0] if i == input_ind else None for i in range(len(op.inputs))]

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

def nonlinearity_1d(input_ind):
    #def handler(explainer, op, *grads, xin0=None, rin0=None):
        #return nonlinearity_1d_handler(input_ind, explainer, op, *grads, xin0=xin0, rin0=rin0)
    def handler(explainer, op, *grads, **kwargs):
        return nonlinearity_1d_handler(input_ind, explainer, op, *grads, **kwargs)
    return handler

def nonlinearity_1d_handler(input_ind, explainer, op, *grads, xin0=None, rin0=None):
    """Handle nonlinear operations with one variable input."""
    print("\n=== nonlinearity_1d_handler called ===")
    print(f"Op type: {op.type}")
    print(f"Op name: {op.name}")
    print(f"Input index: {input_ind}")
    print(f"Number of inputs: {len(op.inputs)}")
    print(f"Input shapes: {[i.shape for i in op.inputs]}")
    print(f"Output shape: {op.outputs[0].shape}")
    print(f"Gradient shape: {grads[0].shape}")

    # make sure only the given input varies
    for i in range(len(op.inputs)):
        if i != input_ind:
            assert not explainer._variable_inputs(op)[i], str(i) + "th input to " + op.name + " cannot vary!"
    
    # Use pre-split tensors if provided, otherwise split them here
    if xin0 is None or rin0 is None:
        print("WARNING: Using fallback tensor splitting - this should not happen")
        xin0, rin0 = tf.split(op.inputs[input_ind], 2)
    
    # Get outputs by applying the operation directly
    xout = getattr(tf.raw_ops, op.type)(features=xin0)
    rout = getattr(tf.raw_ops, op.type)(features=rin0)
    
    delta_in0 = xin0 - rin0
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]

    print(f"Split shapes:")
    print(f"  xin0: {xin0.shape}")
    print(f"  rin0: {rin0.shape}")
    print(f"  xout: {xout.shape}")
    print(f"  rout: {rout.shape}")
    print(f"  delta_in0: {delta_in0.shape}")
    
    print("\nDebug original gradients:")
    print(f"Available op handlers: {list(explainer.orig_grads.keys())}")
    print(f"Looking for op type: {op.type}")
    print(f"Found handler: {explainer.orig_grads.get(op.type)}")
    
    out = [None for _ in op.inputs]
    orig_grads = explainer.orig_grads[op.type](grads[0])
    print(f"Original gradient shape: {orig_grads[input_ind].shape if len(op.inputs) > 1 else orig_grads.shape}")
    
    result = tf.where(
        tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
        orig_grads[input_ind] if len(op.inputs) > 1 else orig_grads,
        grads[0] * tf.tile((xout - rout) / delta_in0, dup0)
    )

    print(f"Final result shape: {result.shape}")
    print("=== handler finished ===\n")
    
    out[input_ind] = result
    return out

def nonlinearity_2d_handler(input_ind0, input_ind1, op_func, explainer, op, *grads, xin0=None, rin0=None, xin1=None, rin1=None):
    """Handle nonlinear operations with two variable inputs."""
    assert input_ind0 == 0 and input_ind1 == 1, "TODO: Can't yet handle double inputs that are not first!"
    
    print("\n=== nonlinearity_2d_handler debug ===")
    
    # Use pre-split tensors if provided
    if xin0 is None or rin0 is None or xin1 is None or rin1 is None:
        print("WARNING: Using fallback tensor splitting - this should not happen")
        xin0, rin0 = tf.split(op.inputs[input_ind0], 2)
        xin1, rin1 = tf.split(op.inputs[input_ind1], 2)
    
    xout, rout = tf.split(op.outputs[0], 2)
    
    delta_in0 = xin0 - rin0
    delta_in1 = xin1 - rin1
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    out10 = op_func(xin0, rin1)
    out01 = op_func(rin0, xin1)
    out11, out00 = xout, rout
    out0 = 0.5 * (out11 - out01 + out10 - out00)
    out0 = grads[0] * tf.tile(out0 / delta_in0, dup0)
    out1 = 0.5 * (out11 - out10 + out01 - out00)
    out1 = grads[0] * tf.tile(out1 / delta_in1, dup0)

    # Handle broadcasting
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

def linearity_1d_nonlinearity_2d(input_ind0, input_ind1, op_func):
    def handler(explainer, op, *grads, xin0=None, rin0=None, xin1=None, rin1=None):
        var = explainer._variable_inputs(op)
        if var[input_ind0] and not var[input_ind1]:
            return linearity_1d_handler(input_ind0, explainer, op, *grads, xin0=xin0, rin0=rin0)
        elif var[input_ind1] and not var[input_ind0]:
            return linearity_1d_handler(input_ind1, explainer, op, *grads, xin0=xin1, rin0=rin1)
        elif var[input_ind0] and var[input_ind1]:
            return nonlinearity_2d_handler(input_ind0, input_ind1, op_func, explainer, op, *grads, 
                                         xin0=xin0, rin0=rin0, xin1=xin1, rin1=rin1)
        else:
            return [None for _ in op.inputs] # no inputs vary, we must be hidden by a switch function
    return handler

def nonlinearity_1d_nonlinearity_2d(input_ind0, input_ind1, op_func):
    def handler(explainer, op, *grads, xin0=None, rin0=None, xin1=None, rin1=None):
        var = explainer._variable_inputs(op)
        if var[input_ind0] and not var[input_ind1]:
            return nonlinearity_1d_handler(input_ind0, explainer, op, *grads, xin0=xin0, rin0=rin0)
        elif var[input_ind1] and not var[input_ind0]:
            return nonlinearity_1d_handler(input_ind1, explainer, op, *grads, xin0=xin1, rin0=rin1)
        elif var[input_ind0] and var[input_ind1]:
            return nonlinearity_2d_handler(input_ind0, input_ind1, op_func, explainer, op, *grads,
                                         xin0=xin0, rin0=rin0, xin1=xin1, rin1=rin1)
        else:
            return [None for _ in op.inputs] # no inputs vary, we must be hidden by a switch function
    return handler

def softmax(explainer, op, *grads, xin0=None, rin0=None):
    """Handle softmax operations with pre-split tensors."""
    if xin0 is None or rin0 is None:
        print("WARNING: Using fallback tensor splitting - this should not happen")
        xin0, rin0 = tf.split(op.inputs[0], 2)
    
    # Center inputs
    xin0_max = tf.reduce_max(xin0, axis=-1, keepdims=True)
    rin0_max = tf.reduce_max(rin0, axis=-1, keepdims=True)
    xin0_centered = xin0 - xin0_max
    rin0_centered = rin0 - rin0_max
    
    # Compute exponentials
    xevals = tf.exp(xin0_centered)
    revals = tf.exp(rin0_centered)
    xsum = tf.reduce_sum(xevals, axis=-1, keepdims=True)
    rsum = tf.reduce_sum(revals, axis=-1, keepdims=True)
    xdiv = xevals / xsum
    rdiv = revals / rsum
    
    # Track intermediate operations if we're in sess mode
    try:
        explainer.between_ops.extend([xevals.op, xsum.op, xdiv.op, xin0_centered.op])
        out = tf.gradients(xdiv, xin0_centered, grad_ys=grads[0])[0]
        del explainer.between_ops[-4:]
    except AttributeError:
        with tf.GradientTape() as tape:
            tape.watch(xin0_centered)
            out = tape.gradient(xdiv, xin0_centered, output_gradients=grads[0])

    # Rescale to account for our shift
    delta_in0 = xin0 - rin0
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    
    return tf.where(
        tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
        out,
        out * tf.tile((xin0_centered - rin0_centered) / delta_in0, dup0)
    )

'''def maxpool(explainer, op, *grads, xin0=None, rin0=None):
    """Handle maxpool operations with pre-split tensors."""
    print('HERE HERE HERE')
    
    # Use pre-split tensors if provided, otherwise split them here
    if xin0 is None or rin0 is None:
        print("WARNING: Using fallback tensor splitting - this should not happen")
        xin0, rin0 = tf.split(op.inputs[0], 2)
    
    print(f"Input shapes - xin0: {xin0.shape}, rin0: {rin0.shape}")
    
    # Add extra dimension for MaxPool1D
    xin0_expanded = tf.expand_dims(xin0, 2)
    rin0_expanded = tf.expand_dims(rin0, 2)
    
    # Apply maxpool to each split tensor
    xout = getattr(tf.raw_ops, op.type)(input=xin0_expanded, ksize=op.get_attr('ksize'),
                                       strides=op.get_attr('strides'),
                                       padding=op.get_attr('padding'))
    rout = getattr(tf.raw_ops, op.type)(input=rin0_expanded, ksize=op.get_attr('ksize'),
                                       strides=op.get_attr('strides'),
                                       padding=op.get_attr('padding'))
    
    # Remove the extra dimension
    xout = tf.squeeze(xout, axis=2)
    rout = tf.squeeze(rout, axis=2)
    
    print(f"Output shapes - xout: {xout.shape}, rout: {rout.shape}")
    print(f"Gradient shape: {grads[0].shape}")
    
    delta_in0 = xin0 - rin0
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    cross_max = tf.maximum(xout, rout)
    diffs = tf.concat([cross_max - rout, xout - cross_max], 0)
    
    # Resize diffs to match gradient shape
    diffs = tf.image.resize(tf.expand_dims(diffs, -1), 
                          size=[grads[0].shape[1], grads[0].shape[2]], 
                          method='nearest')
    diffs = tf.squeeze(diffs, -1)
    
    print(f"Diffs shape after resize: {diffs.shape}")
    
    xmax_pos, rmax_pos = tf.split(explainer.orig_grads[op.type](grads[0] * diffs), 2)

    return tf.tile(tf.where(
        tf.abs(delta_in0) < 1e-7,
        tf.zeros_like(delta_in0),
        (xmax_pos + rmax_pos) / delta_in0
    ), dup0)'''

# UNDER CONSTRUCTION!!!
def maxpool(explainer, op, *grads, **kwargs):
    """Handle maxpool operations with pre-split tensors."""

    # Use pre-split tensors if provided, otherwise split them here
    xin0 = kwargs.get('xin0', None)
    rin0 = kwargs.get('rin0', None)
    
    # Add height dimension for MaxPool
    xin0 = tf.expand_dims(xin0, 2)
    rin0 = tf.expand_dims(rin0, 2)

    print('inputs split')
    print('xin0', xin0.shape)
    print('rin0', rin0.shape)
    
    # Recombine the split tensors
    inputs = tf.concat([xin0, rin0], axis=0)

    print('inputs combined')
    print('inputs', inputs.shape)
    
    # Apply maxpool to combined tensor
    outputs = getattr(tf.raw_ops, op.type)(input=inputs, ksize=op.get_attr('ksize'),
                                          strides=op.get_attr('strides'),
                                          padding=op.get_attr('padding'))
    
    # Split the result back
    xout, rout = tf.split(outputs, 2, axis=0)
    print('outputs split')
    print('xout', xout.shape)
    print('rout', rout.shape)
    
    delta_in0 = xin0 - rin0
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    cross_max = tf.maximum(xout, rout)
    diffs = tf.concat([cross_max - rout, xout - cross_max], 0)
    xmax_pos, rmax_pos = tf.split(explainer.orig_grads[op.type](op, grads[0] * diffs), 2)

    return tf.tile(tf.where(
        tf.abs(delta_in0) < 1e-7,
        tf.zeros_like(delta_in0),
        (xmax_pos + rmax_pos) / delta_in0
    ), dup0)




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
op_handlers["AddV2"] = passthrough
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
op_handlers["ReadVariableOp"] = break_dependence # TF2-specific
op_handlers["NoOp"] = break_dependence # TF2-specific
#op_handlers["StopGradient"] = break_dependence # this allows us to stop attributions when we want to (like softmax re-centering)

if 0:
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

else: # debug mode only
    op_handlers["Relu"] = nonlinearity_1d(0)


    # ops that are linear and only allow a single input to vary
    op_handlers["Reshape"] =  passthrough
    op_handlers["Pad"] =  passthrough
    op_handlers["ReverseV2"] =  passthrough
    op_handlers["ConcatV2"] =  passthrough
    op_handlers["Conv2D"] =  passthrough
    op_handlers["Switch"] =  passthrough
    op_handlers["AvgPool"] =  passthrough
    op_handlers["FusedBatchNorm"] =  passthrough

    # ops that are nonlinear and only allow a single input to vary
    op_handlers["Elu"] =  passthrough
    op_handlers["Sigmoid"] =  passthrough
    op_handlers["Tanh"] =  passthrough
    op_handlers["Softplus"] =  passthrough
    op_handlers["Exp"] =  passthrough
    op_handlers["Log"] =  passthrough
    op_handlers["ClipByValue"] =  passthrough
    op_handlers["Rsqrt"] =  passthrough
    op_handlers["Square"] =  passthrough
    op_handlers["Max"] =  passthrough

    # ops that are nonlinear and allow two inputs to vary
    op_handlers["SquaredDifference"] =  passthrough
    op_handlers["Minimum"] =  passthrough
    op_handlers["Maximum"] =  passthrough

    # ops that allow up to two inputs to vary are are linear when only one input varies
    op_handlers["Mul"] = passthrough
    op_handlers["RealDiv"] = passthrough
    op_handlers["MatMul"] = passthrough

    # ops that need their own custom attribution functions
    op_handlers["GatherV2"] = passthrough
    op_handlers["ResourceGather"] = passthrough
    op_handlers["MaxPool"] = passthrough
    op_handlers["Softmax"] = passthrough



"""
Deep TensorFlow 2 Utilities
==========================

This module provides core functionality for handling gradients and operations in deep learning models,
particularly for attribution methods. Here are the key concepts:

Key Components
-------------
1. Operation Handlers
   - Each operation type (ReLU, MaxPool, etc.) has a specific handler
   - Handlers process gradients differently based on operation characteristics
   - Main types: linear, nonlinear_1d (one input varies), nonlinear_2d (two inputs vary)

2. Tensor Splitting
   - Operations work with split tensors (x and reference parts)
   - Splits happen in forward pass to maintain proper TF2 scoping
   - Handlers receive pre-split tensors as arguments (xin0/rin0 for first input, xin1/rin1 for second)

3. Handler Categories
   - passthrough: ops that pass gradients unchanged (e.g., Identity)
   - break_dependence: ops that don't pass attributions (e.g., Shape)
   - linearity_1d: ops linear in one input (e.g., Reshape)
   - nonlinearity_1d: ops nonlinear in one input (e.g., ReLU)
   - nonlinearity_2d: ops nonlinear in two inputs (e.g., Multiply)

Important Implementation Details
------------------------------
1. Variable Inputs
   - Handlers track which inputs can vary using _variable_inputs
   - Non-variable inputs (constants, parameters) are handled differently

2. Gradient Flow
   - Custom gradients are implemented using tf.custom_gradient
   - Splits happen in forward pass, used in backward pass
   - Proper scoping is crucial for TF2 gradient computation

3. Error Handling
   - Fallback mechanisms exist for tensor splitting
   - Warning messages indicate unexpected scenarios
   - Assertions verify operation requirements

Common Pitfalls
--------------
1. Tensor Scoping
   - Always use pre-split tensors from forward pass
   - Avoid accessing op.inputs directly in gradient computation
   - Watch for TF2 function scope boundaries

2. Operation Types
   - Verify operation characteristics before assigning handlers
   - Check input variability assumptions
   - Handle broadcasting cases properly

3. Gradient Flow
   - Ensure proper gradient chain through operations
   - Watch for numerical stability (divide by zero, etc.)
   - Maintain proper shapes through transformations

For more details on specific handlers or operations, see the individual function
docstrings above.
"""