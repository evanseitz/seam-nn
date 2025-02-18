import sys
import os

sys.dont_write_bytecode = True
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

#import logging
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
#logging.getLogger("tensorflow").setLevel(logging.FATAL)
from tqdm import tqdm
import numpy as np
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import squid
import logomaker
#import seam_utilities
import tensorflow as tf

from DeepSHAP_batch.deep.deep_tf_batch import TF2DeepExplainer
from DeepSHAP_batch.deep.deep_torch import PyTorchDeepExplainer

#from keras import backend as K
#sess = K.get_session()


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
# User parameters
# =============================================================================
num_seqs = 1#00000
#gpu = True

# choose sequence of interest
if 0:
    class_idx = 0
    seq_idx = 13748
    name = 'AP1-m0-seq%s' % seq_idx
elif 1:
    class_idx = 1
    seq_idx = 20647
    name = 'Ohler1_mut0_seq%s' % seq_idx

# =============================================================================
# Import sequence information
# =============================================================================
X_file = os.path.join(parent_dir, 'deepstarr_data.h5')
with h5py.File(X_file, 'r') as dataset:
    X_in = np.array(dataset['x_test']).astype(np.float32)
x_ref = X_in[seq_idx]
seq_length = x_ref.shape[0]
print('seq_length:', seq_length)

x_ref = np.expand_dims(x_ref, 0)
alphabet = ['A','C','G','T']
map_start, map_stop = 0, seq_length

# =============================================================================
# Import model
# =============================================================================
print('Loading model...')
import deepstarr_model
model, seq_length, alphabet, num_classes = deepstarr_model.model(version='TF2-v1') #options: {'TF1', 'TF2-v1', 'TF2-v2'}
out_layer = -1*int(num_classes) + class_idx #e.g., -2 for dev, -1 for hk

# =============================================================================
# Prediction for input sequence given model
# =============================================================================
pred = float(model.predict(x_ref)[class_idx][0])
print('Wild-type prediction:', pred)  # Wild-type prediction: 5.298523426055908

# =============================================================================
# Import attribution functions
# =============================================================================
sys.path.append(os.path.join(current_dir, 'deepstarr_assets'))
import attribution_methods # needs: pip install gast==0.2.2
import squid_utils as squid_utils


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.compat.v1.disable_v2_behavior()
#sys.path.append(os.path.join(py_dir, 'deepstarr_assets/kundajelab_shap_29d2ffa/'))
#import shap #import from file (pip uninstall shap; pip install scikit-learn; pip install ipython)
#from shap.explainers.deep.deep_tf import TFDeepExplainer
#from deeplift_dinuc_shuffle import dinuc_shuffle



'''def combine_mult_and_diffref(mult, orig_inp, bg_data):
    assert len(orig_inp)==1
    projected_hypothetical_contribs = np.zeros_like(bg_data[0]).astype("float")
    assert len(orig_inp[0].shape)==2
    for i in range(orig_inp[0].shape[-1]):
        hypothetical_input = np.zeros_like(orig_inp[0]).astype("float")
        hypothetical_input[:,i] = 1.0
        hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[0])
        hypothetical_contribs = hypothetical_difference_from_reference*mult[0]
        projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1)
    return [np.mean(projected_hypothetical_contribs,axis=0)]'''

'''def deepExplainer(one_hot):
    shap_values_hypothetical = explainer.shap_values(one_hot)
    shap_values_contribution = shap_values_hypothetical[0]*one_hot
    return shap_values_hypothetical[0], shap_values_contribution'''

num_shuffles = 100

# Generate backgrounds using both methods with same seed
seed = 42

#print("=== OLD DINUC SHUFFLE ===")
if 0:
    from dinuc_shuffle_oldVersion import dinuc_shuffle

    def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, num_shufs=100, seed=1234):
        assert len(list_containing_input_modes_for_an_example)==1
        onehot_seq = list_containing_input_modes_for_an_example[0]
        rng = np.random.RandomState(seed)
        to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(num_shufs)])
        return [to_return] #wrap in list for compatibility with multiple modes

    background = dinuc_shuffle_several_times(x_ref, num_shufs=num_shuffles, seed=seed)[0]

else:
    from DeepSHAP_batch.utils.sequence_utils import batch_dinuc_shuffle

    background = batch_dinuc_shuffle(x_ref, num_shuffles=num_shuffles, seed=seed)




# =============================================================================
# Define explainer
# =============================================================================
#shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
#explainer = TFDeepExplainer((model.layers[0].input, model.layers[out_layer].output),
            #data=dinuc_shuffle_several_times, combine_mult_and_diffref=combine_mult_and_diffref)#, session=sess)


# Create a model that outputs only the class of interest
# class_idx is 0 for Dev output, 1 for Hk output
output_layer = model.outputs[class_idx]  # Select the appropriate output head
model_output_idx = tf.keras.Model(inputs=model.input, outputs=output_layer)


background = tf.convert_to_tensor(background, dtype=tf.float32)



#pred = float(model_output_idx.predict(x_ref)[0])
#print('pred:', pred)
#print(background.shape)

# Initialize explainer with the class-specific model
explainer = TF2DeepExplainer(model_output_idx, background)


attribution = explainer.shap_values(x_ref).numpy()[0]
print(attribution.shape)
attribution_df = squid.utils.arr2pd(attribution)
print(attribution_df.head())
fig = squid.impress.plot_additive_logo(attribution_df, center=True, alphabet=alphabet, fig_size=[20,2.5])
                                    #view_window=None)
                                    #save_dir=save_dir)
plt.show()

'''attribution = deepExplainer(x_ref)[0][0]
attribution_df = squid.utils.arr2pd(attribution)
fig = squid.impress.plot_additive_logo(attribution_df[map_start:map_stop], center=True, alphabet=alphabet, fig_size=[20,2.5],
                                    view_window=None, save_dir=save_dir)'''

# =============================================================================
# SQUID to generate in silico MAVE dataset
# =============================================================================
# define mutagenesis window for sequence
'''mut_window = [map_start, map_stop]

if 0:
    # set up predictor class for in silico MAVE
    pred_generator = squid.predictor.ScalarPredictor(pred_fun=model.predict_on_batch,
                                                    task_idx=class_idx, batch_size=512)

    # set up mutagenizer class for in silico MAVE
    mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=0.1)

    # generate in silico MAVE
    t1 = time.time()
    mave = squid.mave.InSilicoMAVE(mut_generator, pred_generator, seq_length, mut_window=mut_window, save_window=mut_window)
    x_mut, y_mut = mave.generate(x_ref[0], num_sim=num_seqs)
    t2 = time.time()
    print('Inference time:', t2-t1)

    if 1:
        np.save(os.path.join(save_dir, 'x_mut.npy'), x_mut)
        np.save(os.path.join(save_dir, 'y_mut.npy'), y_mut)

    # plot histogram of deepnet predictions
    fig = squid.impress.plot_y_hist(y_mut, save_dir=save_dir)

    mave_df = seam_utilities.xmut2pd(x_mut, y_mut, alphabet=['A','C','G','T'], hamming=True, encoding=1, save_dir=save_dir, gpu=gpu)

else:
    x_mut = np.load(os.path.join(save_dir, 'x_mut.npy'))


def explainer_in_batches(explainer, x, x_ref, batch_size=attr_batch_size, save_window=None):
    """Function to compute attribution maps in batch mode.

    Parameters
    ----------
    x : numpy.ndarray
        One-hot sequences (shape: (N, L, A)).
    x_ref : numpy.ndarray
        One-hot reference sequence (shape: (1, L, A)).
    batch_size : int
        The number of predictions per batch of model inference.
    save_window : [int, int]
        Window used for delimiting sequences that are exported in 'x_mut' array

    Returns
    -------
    numpy.ndarray
        Attribution maps.
    """

    x_ref = x_ref.astype('uint8')

    if save_window is not None:
        if x_ref.ndim == 2:
            x_ref = x_ref[np.newaxis,:]

    N, L, A = x.shape
    num_batches = np.floor(N/attr_batch_size).astype(int)
    attribution_values = []
    for i in tqdm(range(num_batches), desc="Attribution"):
        x_batch = x[i*attr_batch_size:(i+1)*attr_batch_size]
        if save_window is not None:
            x_ref_start = np.broadcast_to(x_ref[:,:save_window[0],:], (x_batch.shape[0],save_window[0],x_ref.shape[2]))
            x_ref_stop = np.broadcast_to(x_ref[:,save_window[1]:,:], (x_batch.shape[0],x_ref.shape[1]-save_window[1],x_ref.shape[2]))
            x_batch = np.concatenate([x_ref_start, x_batch, x_ref_stop], axis=1)

        if map_type == 'deepshap':
            batch_values = explainer.shap_values(x_batch)[0]
        elif map_type == 'saliency':
            batch_values = explainer.saliency(x_batch, batch_size=attr_batch_size)
        elif map_type == 'smoothgrad':
            batch_values = explainer.smoothgrad(x_batch)
        elif map_type == 'intgrad':
            batch_values = explainer.intgrad(x_batch, baseline_type='zeros', num_steps=50)
            batch_values = batch_values[None, :, :]
        elif map_type == 'ism':
            batch_values = squid_utils.ISM_single(x=x_batch, model=model, class_idx=class_idx, example='DeepSTARR', get_prediction=deepstarr_model.get_prediction,
                                                    unwrap_prediction=deepstarr_model.unwrap_prediction, compress_prediction=deepstarr_model.compress_prediction,
                                                    pred_transform=None, pred_trans_delimit=None, log2FC=False,
                                                    max_in_mem=False, save=False, save_dir=None,
                                                    start=map_start, stop=map_stop)
            batch_values = batch_values[None, :, :] # shape: (1, L, A)

        attribution_values.append(batch_values) # shape: (attr_batch_size, L, A)

    if num_batches*attr_batch_size < N:
        x_batch = x[num_batches*attr_batch_size:]
        if save_window is not None:
            x_ref_start = np.broadcast_to(x_ref[:,:save_window[0],:], (x_batch.shape[0],save_window[0],x_ref.shape[2]))
            x_ref_stop = np.broadcast_to(x_ref[:,save_window[1]:,:], (x_batch.shape[0],x_ref.shape[1]-save_window[1],x_ref.shape[2]))
            x_batch = np.concatenate([x_ref_start, x_batch, x_ref_stop], axis=1)

        if map_type == 'deepshap':
            batch_values = explainer.shap_values(x_batch)[0]
        elif map_type == 'saliency':
            batch_values = explainer.saliency(x_batch, batch_size=attr_batch_size)
        elif map_type == 'smoothgrad':
            batch_values = explainer.smoothgrad(x_batch)
        elif map_type == 'intgrad':
            batch_values = explainer.intgrad(x_batch, baseline_type='zeros', num_steps=50)
            batch_values = batch_values[None, :, :]
        elif map_type == 'ism':
            batch_values = squid_utils.ISM_single(x=x_batch, model=model, class_idx=class_idx, example='DeepSTARR', get_prediction=deepstarr_model.get_prediction,
                                                    unwrap_prediction=deepstarr_model.unwrap_prediction, compress_prediction=deepstarr_model.compress_prediction,
                                                    pred_transform=None, pred_trans_delimit=None, log2FC=False,
                                                    max_in_mem=False, save=False, save_dir=None,
                                                    start=map_start, stop=map_stop)
            batch_values = batch_values[None, :, :] # shape: (1, L, A)

        attribution_values.append(batch_values)

    return np.vstack(attribution_values)

t1 = time.time()
map_stack = explainer_in_batches(explainer, x_mut, x_ref) # shape: (N, L, A)
t2 = time.time()
print('Attribution time:', t2-t1)

if 0:
    np.save(os.path.join(save_dir, 'maps_%s.npy' % map_type), map_stack)'''