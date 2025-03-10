import tensorflow as tf

def analyze_original_model_gradients(model, inputs, backgrounds):
    """Analyze gradients through the original model layer by layer."""
    print("\n=== Original Model Gradient Analysis ===")
    
    # Create a list to store intermediate activations
    activations = []
    
    # Get all layers that we care about
    layers_to_watch = [layer for layer in model.layers 
                      if any(op in layer.__class__.__name__ 
                          for op in ['Dense', 'Conv1D', 'ReLU'])]
    
    with tf.GradientTape(persistent=True) as tape:
        # Watch input
        tape.watch(inputs)
        
        # Watch each layer's output
        x = inputs
        for layer in layers_to_watch:
            x = layer(x)
            tape.watch(x)
            activations.append(x)
        
        final_output = x

    # Compute gradients backwards through each watched activation
    upstream_grad = tf.ones_like(final_output)  # Same as our custom model
    
    print("\nGradient propagation through original model:")
    for idx in reversed(range(len(activations))):
        layer = layers_to_watch[idx]
        current_grads = tape.gradient(activations[idx], inputs, output_gradients=upstream_grad)
        
        print(f"\n=== Layer: {layer.__class__.__name__} ===")
        print(f"Gradients (first 5):", current_grads.numpy().flatten()[:5])
        
        # Update upstream gradients for next iteration
        if idx > 0:  # Don't need to compute for first layer
            upstream_grad = tape.gradient(activations[idx], activations[idx-1])

    return tape.gradient(final_output, inputs, output_gradients=upstream_grad)

# In your main test section:
def test_custom_grad():
    # ... existing setup code ...
    
    # First run original model analysis
    print("\nAnalyzing original model gradients...")
    orig_grads_detailed = analyze_original_model_gradients(original_model, inputs, backgrounds)
    
    # Then run custom model (with existing debug prints)
    print("\nAnalyzing custom model gradients...")
    custom_grads = compute_gradients(custom_model, inputs, backgrounds)
    
    # Compare final results
    print("\nFinal Comparison:")
    print("Original model final gradients (first 10):", orig_grads_detailed.numpy().flatten()[:10])
    print("Custom model final gradients (first 10):", custom_grads.numpy().flatten()[:10])

if __name__ == "__main__":
    test_custom_grad() 