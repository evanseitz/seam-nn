import os, sys
sys.dont_write_bytecode = True
import numpy as np
import h5py
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(current_dir, 'deepstarr_assets/'))

from deep.minimal_working import build_custom_gradient_model, nonlinearity_1d

# =============================================================================
# Choose sequence of interest
# =============================================================================
if 0: # AP1-m0
    class_idx = 0
    seq_idx = 13748
elif 1: # Ohler1_mut0
    class_idx = 1
    seq_idx = 20647

# =============================================================================
# Import sequence information
# =============================================================================
X_file = os.path.join(current_dir, 'deepstarr_assets/deepstarr_data.h5')
with h5py.File(X_file, 'r') as dataset:
    X_in = np.array(dataset['x_test']).astype(np.float32)
x_ref = X_in[seq_idx]
seq_length = x_ref.shape[0]
print('seq_length:', seq_length)

x_ref = np.expand_dims(x_ref, 0)
alphabet = ['A','C','G','T']
map_start, map_stop = 0, seq_length

# =============================================================================
# Import model and prepare data
# =============================================================================
import deepstarr_model
model, seq_length, alphabet, num_classes = deepstarr_model.model(version='TF2-v1')

pred = float(model.predict(x_ref)[class_idx][0])
print('Wild-type prediction:', pred)

# =============================================================================
# Import attribution functions
# =============================================================================
num_shuffles = 3
seed = 42
from deep.sequence_utils import batch_dinuc_shuffle
background = batch_dinuc_shuffle(x_ref, num_shuffles=num_shuffles, seed=seed)
background = tf.convert_to_tensor(background, dtype=tf.float32)

# =============================================================================
# Test custom gradient model
# =============================================================================
# Set random seeds
tf.random.set_seed(42)

# Define handlers
op_handlers = {}
if 1:  # Switch for custom nonlinearity handling
    op_handlers["Relu"] = nonlinearity_1d(0)
    print("Registered handlers:", list(op_handlers.keys()))

# Select output head to explain
output_layer = model.outputs[class_idx]
model_output_idx = tf.keras.Model(inputs=model.input, outputs=output_layer)

# Create model with custom gradients
custom_model = build_custom_gradient_model(model_output_idx, op_handlers)

# Print layer operations to verify ReLU is being detected
for layer in custom_model.layers:
    if hasattr(layer, 'internal_ops'):
        print(f"Layer {layer.name} operations:", layer.internal_ops)

# Prepare input with backgrounds
if not isinstance(x_ref, list):
    x_ref = [x_ref]

print("\nTesting custom gradients...")
print("Input shapes:", [x.shape for x in x_ref])
print("Background shapes:", [b.shape for b in background])

noutputs = 1
model_output_ranks = np.tile(np.arange(noutputs)[None,:], (x_ref[0].shape[0],1))

tiled_X = [np.tile(x[0:1], (len(background),) + tuple([1 for k in range(len(x.shape)-1)])) for x in x_ref]

# Create joint input for this sample:
stacked_background = tf.stack(background)  # Shape: (3, 249, 4)
x_with_backgrounds = [tf.convert_to_tensor(
    np.concatenate([tiled_X[l], stacked_background], 0),  # Will be (6, 249, 4)
    dtype=tf.float32) 
    for l in range(len(x_ref))]

# Compare gradients between original and custom model
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x_with_backgrounds)
    orig_pred = model_output_idx(x_with_backgrounds)
    custom_pred = custom_model(x_with_backgrounds)

# Get gradients directly with respect to predictions
orig_grads = tape.gradient(orig_pred, x_with_backgrounds)
custom_grads = tape.gradient(custom_pred, x_with_backgrounds)

print(f"Original gradients first 10:", orig_grads[0].numpy().flatten()[:10])
print(f"Custom gradients first 10:", custom_grads[0].numpy().flatten()[:10])

def analyze_original_model_gradients(base_model, inputs, class_idx):
    """Analyze gradients through the original model layer by layer."""
    print("\n=== Original Model Gradient Analysis ===")
    
    # Create a model that outputs both intermediate activations and final output
    conv1 = base_model.get_layer('Conv1D_1st')
    act1 = base_model.get_layer('activation_72')
    conv2 = base_model.get_layer('Conv1D_2')
    
    intermediate_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            conv1.output,
            act1.output,
            conv2.output,
            base_model.output
        ]
    )
    
    with tf.GradientTape(persistent=True) as tape:
        x = inputs[0]
        tape.watch(x)
        
        # Get all outputs in a single forward pass
        conv1_out, act1_out, conv2_out, final_out = intermediate_model(inputs)
        target_output = final_out[class_idx]
    
    # Get gradients at each point
    input_grads = tape.gradient(target_output, x)
    conv1_grads = tape.gradient(target_output, conv1_out)
    act1_grads = tape.gradient(target_output, act1_out)
    conv2_grads = tape.gradient(target_output, conv2_out)
    
    print("\n=== Input Gradients ===")
    print("Input shape:", x.shape)
    print("Gradients (first 5):", input_grads.numpy().flatten()[:5])
    
    print("\n=== First Conv Layer Gradients ===")
    print("Conv1 output shape:", conv1_out.shape)
    print("Gradients (first 5):", conv1_grads.numpy().flatten()[:5])
    
    print("\n=== First Activation Gradients ===")
    print("Act1 output shape:", act1_out.shape)
    print("Gradients (first 5):", act1_grads.numpy().flatten()[:5])
    
    print("\n=== Second Conv Layer Gradients ===")
    print("Conv2 output shape:", conv2_out.shape)
    print("Gradients (first 5):", conv2_grads.numpy().flatten()[:5])
    
    return input_grads

print("\nAnalyzing original model gradients...")
orig_grads_detailed = analyze_original_model_gradients(model, x_with_backgrounds, class_idx)
