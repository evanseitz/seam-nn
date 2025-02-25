import tensorflow as tf

def trace_layer_ops(layer, input_shape):
    # Create input
    x = tf.random.normal(input_shape)
    
    # Try to trace the operations
    @tf.function
    def trace_ops(inputs):
        with tf.GradientTape() as tape:
            outputs = layer(inputs)
        return outputs, tape.watched_variables()
    
    # Run and print operations
    outputs, watched = trace_ops(x)
    concrete_fn = trace_ops.get_concrete_function(x)
    print("Operations traced:", [op.name for op in concrete_fn.graph.get_operations()])
    return concrete_fn.graph.get_operations()

# Test with a Conv1D layer
if __name__ == "__main__":
    conv_layer = tf.keras.layers.Conv1D(filters=32, kernel_size=3)
    ops = trace_layer_ops(conv_layer, (1, 10, 1))