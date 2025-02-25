import os, sys, time
import logging
sys.dont_write_bytecode = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
logging.getLogger("tensorflow").setLevel(logging.FATAL)
from tqdm import tqdm
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import squid
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(current_dir, 'deepstarr_assets/'))

from deep.deep_tf1 import TFDeepExplainer, op_handlers, passthrough

# tested environment (CPU):
    # conda create -n deepstarr2 python=3.11 tensorflow=2.12
    # conda activate deepstarr2
    # conda install -c conda-forge keras 
    # conda install -c anaconda pandas
    # conda install -c anaconda scikit-learn
    # pip install squid-nn
    # pip install logomaker
    # pip install gast==0.2.2
# tested environment (GPU on Citra):
    # conda create --name deepstarr2_gpu python=3.7 tensorflow-gpu
    # pip install keras
    # pip install squid-nn
    # pip install logomaker
    # pip install --upgrade tensorflow

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

# Helper function for tensor conversion
def convert_to_numpy(tensor):
    if isinstance(tensor, tf.Tensor):
        try:
            return tensor.numpy()  # TF2.x
        except AttributeError:
            return tf.keras.backend.get_value(tensor)  # TF1.x
    return tensor

# =============================================================================
# Import model
# =============================================================================
print('Loading model...')
import deepstarr_model
model, seq_length, alphabet, num_classes = deepstarr_model.model(version='TF2-v1') #options: {'TF1', 'TF2-v1', 'TF2-v2'}
out_layer = -1*int(num_classes) + class_idx #e.g., -2 for dev, -1 for hk

pred = float(model.predict(x_ref)[class_idx][0])
print('Wild-type prediction:', pred)

# =============================================================================
# Import deepshap attribution functions
# =============================================================================
num_shuffles = 3
seed = 42
original_dinuc_shuffle = False
original_combine_mult_and_diffref = False

if original_dinuc_shuffle:
    from deep.dinuc_shuffle_orig import dinuc_shuffle
    def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, seed=1234):
        assert len(list_containing_input_modes_for_an_example)==1
        onehot_seq = list_containing_input_modes_for_an_example[0]
        rng = np.random.RandomState(seed)
        to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(num_shuffles)])
        return [to_return]  # Wrap in list for compatibility with multiple modes
    background = dinuc_shuffle_several_times(x_ref, seed=seed)[0]
else:
    from deep.sequence_utils import batch_dinuc_shuffle
    background = batch_dinuc_shuffle(x_ref, num_shuffles=num_shuffles, seed=seed)
    background = convert_to_numpy(background)

if original_combine_mult_and_diffref:
    def standard_combine_mult_and_diffref(mult, orig_inp, bg_data):
        assert len(orig_inp)==1
        projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
        assert len(orig_inp[0].shape)==2
        for i in range(orig_inp[0].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")
            hypothetical_input[:,i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[0])
            hypothetical_contribs = hypothetical_difference_from_reference*mult[0]
            projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1)
        return [np.mean(projected_hypothetical_contribs,axis=0)]
else:
    from deep.utils import standard_combine_mult_and_diffref

# =============================================================================
# Define explainer
# =============================================================================

if 1:
    print("\nTesting original model gradients...")
    
    # IMPORTANT NOTE: There is a critical TensorFlow 1.x quirk where the model weights
    # are transformed after the first prediction call. This means:
    # 1. The first model.predict() call gives the correct output values
    # 2. These values are different from subsequent predictions unless we capture and restore the weights
    # 3. This affects both predictions and gradients
    # 
    # To handle this, we:
    # 1. Make an initial prediction to trigger the weight transformation
    # 2. Capture the transformed weights
    # 3. Restore these weights in our session before computing gradients
    
    # First prediction to trigger weight transformation
    #print("\nFirst prediction:")
    pred = float(model.predict(x_ref)[class_idx][0])
    #print('Wild-type prediction:', pred)
    
    # Verify raw model output
    #print("\nRaw model output:")
    raw_pred = model.predict(x_ref)
    #print('Full output:', raw_pred)
    #print('Selected class:', raw_pred[class_idx][0])
    
    # Capture the transformed weights
    weights = [layer.get_weights() for layer in model.layers if layer.get_weights()]
    
    # Setup for gradient computation
    test_input = tf.convert_to_tensor(x_ref, dtype=tf.float32)
    #print(f"\nInput shape: {test_input.shape}")
    
    # Get output layers
    dev_output = model.layers[-2].output  # Dense_Dev
    hk_output = model.layers[-1].output   # Dense_Hk
    #print(f"Dev output shape: {dev_output.shape}")
    #print(f"Hk output shape: {hk_output.shape}")
    
    # Run the computation in a session
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Critical step: Restore the transformed weights
        weight_idx = 0
        for layer in model.layers:
            if layer.get_weights():
                layer.set_weights(weights[weight_idx])
                weight_idx += 1
        
        # Create feed dict with global learning phase flag set to False
        feed_dict = {model.layers[0].input: x_ref}
        feed_dict[tf.keras.backend.learning_phase()] = False
        
        # Verify predictions match the original model.predict()
        dev_pred = sess.run(dev_output, feed_dict)
        hk_pred = sess.run(hk_output, feed_dict)
        print(f"\nSession predictions:")
        #print(f"Dev prediction: {dev_pred}")
        print(f"Hk prediction: {hk_pred}")
        
        # Compute gradients with the correct weights
        dev_grads = sess.run(tf.gradients(dev_output, model.layers[0].input)[0], feed_dict)
        hk_grads = sess.run(tf.gradients(hk_output, model.layers[0].input)[0], feed_dict)
        
    #print(f"\nDev gradients (first 10): {dev_grads.flatten()[:10]}")
    print(f"Hk gradients (first 10): {hk_grads.flatten()[:10]}")
    print('')

op_handlers["AddV2"] = passthrough
explainer = TFDeepExplainer((model.layers[0].input, model.layers[out_layer].output),
                           data=background, 
                           combine_mult_and_diffref=standard_combine_mult_and_diffref)

print(x_ref.shape)
print('Testing phi_symbolic (TF1)...')
explainer.test_phi_symbolic(x_ref)

if 1:
    attribution = explainer.shap_values(x_ref)[0][0]
    attribution_df = squid.utils.arr2pd(attribution)
    print(attribution_df.head())
    fig = squid.impress.plot_additive_logo(attribution_df, center=True, alphabet=alphabet, fig_size=[20,2.5])#,
    plt.show()