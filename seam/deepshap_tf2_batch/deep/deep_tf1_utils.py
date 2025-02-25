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

# Operation handlers

def softmax(explainer, op, *grads):
    #print("\nDEBUG NEW VERSION:")
    #print("Input op type:", type(op))
    #print("Input grads type:", type(grads[0]))
    
    in0 = op.inputs[0]
    #print("in0 shape:", in0.shape)
    
    in0_max = tf.reduce_max(in0, axis=-1, keepdims=True, name="in0_max")
    in0_centered = in0 - in0_max
    #print("in0_centered shape:", in0_centered.shape)
    
    evals = tf.exp(in0_centered, name="custom_exp")
    rsum = tf.reduce_sum(evals, axis=-1, keepdims=True)
    div = evals / rsum
    #print("div shape:", div.shape)
    
    # Track intermediate operations if we're in sess mode
    try:
        #print("Trying sess mode...")
        explainer.between_ops.extend([evals.op, rsum.op, div.op, in0_centered.op])
        out = tf.gradients(div, in0_centered, grad_ys=grads[0])[0]
        #print("Sess gradient shape:", out.shape)
        del explainer.between_ops[-4:]
    except AttributeError:
        #print("Using eager mode...")
        with tf.GradientTape() as tape:
            tape.watch(in0_centered)
            out = tape.gradient(div, in0_centered, output_gradients=grads[0])
            #print("Eager gradient shape:", out.shape if out is not None else None)

    # Rescale to account for our shift by in0_max
    xin0,rin0 = tf.split(in0, 2)
    xin0_centered,rin0_centered = tf.split(in0_centered, 2)
    delta_in0 = xin0 - rin0
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    #print("delta_in0 shape:", delta_in0.shape)
    #print("dup0:", dup0)
    
    result = tf.where(
        tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
        out,
        out * tf.tile((xin0_centered - rin0_centered) / delta_in0, dup0)
    )
    #print("Final result shape:", result.shape)
    return result

def maxpool(explainer, op, *grads):
    inputs = op.inputs[0]
    outputs = op.outputs[0]

    print('inputs', inputs.shape)
    print('outputs', outputs.shape)
    print('maxpool')
    xin0,rin0 = tf.split(inputs, 2)
    print('inputs split')
    print('xin0 static shape:', xin0.shape)
    print('xin0 dynamic shape:', tf.shape(xin0))
    print('rin0 static shape:', rin0.shape)
    print('rin0 dynamic shape:', tf.shape(rin0))
    xout,rout = tf.split(outputs, 2)
    print('outputs split')
    print('xout static shape:', xout.shape)
    print('xout dynamic shape:', tf.shape(xout))
    print('rout static shape:', rout.shape)
    print('rout dynamic shape:', tf.shape(rout))
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

# Graph traversal utilities
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

'''def _get_concrete_ops(tensor_or_op):
    """Helper function to get concrete ops from Keras tensors or TF ops."""
    print(f"\nDebug _get_concrete_ops - Input type: {type(tensor_or_op)}")
    
    if isinstance(tensor_or_op, tf.keras.layers.Layer):
        print("- Processing as Keras layer")
        return tensor_or_op.output.op
    elif hasattr(tensor_or_op, '_keras_history'):
        print("- Processing as Keras tensor")
        print(f"- Shape: {tensor_or_op.shape}")
        print(f"- Keras history: {tensor_or_op._keras_history}")
        
        # For Keras tensors, create a concrete function that returns this tensor
        dummy_input = tf.zeros([1] + list(tensor_or_op.shape[1:]))
        print(f"- Created dummy input with shape: {dummy_input.shape}")
        
        @tf.function
        @tf.autograph.experimental.do_not_convert
        def get_tensor(x):
            return x
        
        concrete_func = get_tensor.get_concrete_function(dummy_input)
        print(f"- Concrete function output op: {concrete_func.outputs[0].op}")
        return concrete_func.outputs[0].op
    elif isinstance(tensor_or_op, tf.Tensor):
        print("- Processing as TF tensor")
        return tensor_or_op.op
    elif isinstance(tensor_or_op, tf.Operation):
        print("- Processing as TF operation")
        return tensor_or_op
    else:
        raise TypeError(f"Unsupported type: {type(tensor_or_op)}")'''

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

def filter_computation_ops(ops):
    """Filter out resource and constant ops to focus on computation."""
    compute_ops = []
    # Note: Switch/Merge operations are TF1-specific. In TF2, dropout is handled by 
    # Identity ops during inference and the Keras layer during training
    compute_types = {
        'Conv2D', 'BiasAdd', 'MatMul', 'Relu', 'Sigmoid', 'Tanh',
        'LeakyRelu', 'MaxPool', 'AvgPool', 'Add', 'AddV2', 'Mul', 'Sub', 'Div',
        'ExpandDims', 'Squeeze', 'Reshape', 'Identity'
    }
    
    for op in ops:
        # Skip resource operations
        if op.type.endswith('_resource'):
            continue
        # Skip constant operations
        if op.type == 'Const':
            continue
        # Skip read variable operations
        if op.type == 'ReadVariableOp':
            continue
        # Skip final Identity op
        if op.type == 'Identity' and op.name == 'Identity':
            continue
        # Skip input placeholder
        if op.type == 'Placeholder':
            continue
        # Keep computation ops
        if op.type in compute_types:
            compute_ops.append(op)
            
    return compute_ops

def backward_walk_ops_tf2(start_ops, tensor_blacklist, all_ops, dependence_breakers=None):
    """TF2 version of backward walk that finds all ops between the outputs and inputs.
    
    Args:
        start_ops: List of starting operations to walk back from
        tensor_blacklist: Set of tensors to block walking through
        all_ops: Set of all operations to consider
        dependence_breakers: List of operation types that should break dependencies
    
    Returns:
        List of operations between start_ops and inputs, in forward propagation order
    """
    found_ops = []  # Changed from set to list
    visited = set()  # Use a separate set for checking if we've seen an op
    
    def recurse(op):
        if op not in visited and op in all_ops:
            visited.add(op)
            # Don't continue if this op breaks dependencies
            if dependence_breakers and op.type in dependence_breakers:
                return
            # First recurse through inputs (deeper ops)
            for inp in op.inputs:
                if inp not in tensor_blacklist and hasattr(inp, 'op'):
                    recurse(inp.op)
            # Then add current op (maintains forward prop order)
            found_ops.append(op)
    
    for op in start_ops:
        recurse(op)
    
    # Reverse to get forward propagation order
    found_ops.reverse()
    return found_ops

def forward_walk_ops_tf2(start_ops, tensor_blacklist, all_ops, dependence_breakers=None):
    """TF2 version of forward walk that finds all ops between inputs and outputs.
    
    Args:
        start_ops: List of starting operations to walk forward from
        tensor_blacklist: Set of tensors to block walking through
        all_ops: Set of all operations to consider
        dependence_breakers: List of operation types that should break dependencies
    
    Returns:
        List of operations between start_ops and outputs, in forward propagation order
    """
    found_ops = []  # Changed from set to list
    visited = set()  # Use a separate set for checking if we've seen an op
    
    def recurse(op):
        if op not in visited and op in all_ops:
            visited.add(op)
            found_ops.append(op)  # Add in forward order
            # Don't continue if this op breaks dependencies
            if dependence_breakers and op.type in dependence_breakers:
                return
            for out in op.outputs:
                if out not in tensor_blacklist:
                    for consumer in out.consumers():
                        recurse(consumer)
    
    for op in start_ops:
        recurse(op)
    
    return found_ops

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

'''def nonlinearity_1d_handler(input_ind, explainer, op, *grads):
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
    return out'''

def nonlinearity_1d_handler(input_ind, explainer, op, *grads):
    # make sure only the given input varies
    for i in range(len(op.inputs)):
        if i != input_ind:
            assert not explainer._variable_inputs(op)[i], str(i) + "th input to " + op.name + " cannot vary!"
    
    #print("\nDEBUG nonlinearity_1d_handler:")
    #print("Input op type:", op.type)
    #print("Input grads shape:", grads[0].shape)
    
    xin0,rin0 = tf.split(op.inputs[input_ind], 2)
    xout,rout = tf.split(op.outputs[0], 2)
    delta_in0 = xin0 - rin0
    dup0 = [2] + [1 for i in delta_in0.shape[1:]]
    #print("dup0:", dup0)
    
    out = [None for _ in op.inputs]
    orig_grads = explainer.orig_grads[op.type](op, grads[0])
    
    result = tf.where(
        tf.tile(tf.abs(delta_in0), dup0) < 1e-6,
        orig_grads[input_ind] if len(op.inputs) > 1 else orig_grads,
        grads[0] * tf.tile((xout - rout) / delta_in0, dup0)
    )
    
    out[input_ind] = result
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
    op_handlers["Relu"] =  passthrough
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

