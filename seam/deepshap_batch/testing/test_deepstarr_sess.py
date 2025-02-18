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
import logomaker
import seam_utilities
import tensorflow as tf

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
num_seqs = 100000
num_shufs = 100
gpu = True

# choose sequence of interest
if 0:
    class_idx = 0
    seq_idx = 13748 # rank_0
    name = 'AP1-m0-seq%s' % seq_idx

elif 1:
    class_idx = 1
    seq_idx = 20647 # rank_53
    name = 'Ohler1_mut0_seq%s' % seq_idx

# =============================================================================
# Import sequence information
# =============================================================================
py_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(py_dir)

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
# Import deepshap attribution functions
# =============================================================================
sys.path.append(os.path.join(parent_dir))
import squid_utils as squid_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.disable_v2_behavior()
sys.path.append(os.path.join(parent_dir, 'kundajelab_shap_29d2ffa/'))
import shap #import from file (pip uninstall shap; pip install scikit-learn; pip install ipython)
from shap.explainers.deep.deep_tf import TFDeepExplainer
from DeepSHAP_session_based.dinuc_shuffle_oldVersion import dinuc_shuffle

def dinuc_shuffle_several_times(list_containing_input_modes_for_an_example, seed=1234):
    assert len(list_containing_input_modes_for_an_example)==1
    onehot_seq = list_containing_input_modes_for_an_example[0]
    rng = np.random.RandomState(seed)
    to_return = np.array([dinuc_shuffle(onehot_seq, rng=rng) for i in range(num_shufs)])
    return [to_return] #wrap in list for compatibility with multiple modes

def combine_mult_and_diffref(mult, orig_inp, bg_data):
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

# =============================================================================
# Import model
# =============================================================================
print('Loading model...')
import deepstarr_model
model, seq_length, alphabet, num_classes = deepstarr_model.model(version='TF2-v1') #options: {'TF1', 'TF2-v1', 'TF2-v2'}
out_layer = -1*int(num_classes) + class_idx #e.g., -2 for dev, -1 for hk

# =============================================================================
# Define explainer
# =============================================================================
shap.explainers.deep.deep_tf.op_handlers["AddV2"] = shap.explainers.deep.deep_tf.passthrough
explainer = TFDeepExplainer((model.layers[0].input, model.layers[out_layer].output),
            data=dinuc_shuffle_several_times, combine_mult_and_diffref=combine_mult_and_diffref)#, session=sess)

# =============================================================================
# Prediction for input sequence given model
# =============================================================================
pred = float(model.predict(x_ref)[class_idx][0])
print('Wild-type prediction:', pred)

if 1:
    attribution = explainer.shap_values(x_ref)[0][0]
    attribution_df = squid.utils.arr2pd(attribution)
    print(attribution_df.head())
    fig = squid.impress.plot_additive_logo(attribution_df, center=True, alphabet=alphabet, fig_size=[20,2.5])#,
    plt.show()