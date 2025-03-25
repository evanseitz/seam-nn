"""
SEAM example of local sequence-mechanism relationships with DeepSTARR
as shown in SEAM Figure 2 # TODO double check this

Tested using:
    - Python 3.11.8
    - TensorFlow 2.16.1
    - NumPy 1.26.1
    - SEAM 0.4.3

Parameters:
    - 30,000 sequences # TODO: set to 10,000 for faster runtime
    - 10% mutation rate
    - Integrated gradients
    - Hierarchical clustering (ward)
"""

import os, sys, time
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import squid
from urllib.request import urlretrieve
import pandas as pd
from tqdm import tqdm
if 1: # Use this for local install (must 'pip uninstall seam' first)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from seam import Compiler, Attributer, Clusterer, MetaExplainer, Identifier
from seam.logomaker_batch.batch_logo import BatchLogo

'''
print(sys.version) # 3.10.12
print(tf.__version__) # 2.16.1
print(np.__version__) # 1.26.1
print(seam.__version__) # 0.4.3
'''

# =============================================================================
# Overhead user settings
# =============================================================================
# TODO: set up for a clean run before pushing to github (and change to intgrad with 30000 seqs)

seq_index = 20647  # Example locus from DeepSTARR test set used in SEAM Figure 2
task_index = 1  # Housekeeping (Hk) program

mut_rate = 0.1 # mutation rate for in silico mutagenesis
num_seqs = 2 # number of sequences to generate
attribution_method = 'ism' # {saliency, smoothgrad, intgrad, deepshap, ism} # TODO: deepshap under construction

gpu = len(tf.config.list_physical_devices('GPU')) > 0 # Whether to use GPU (Boolean)
print(f"Using {'GPU' if gpu else 'CPU'}")
save_figs = True # Whether to save quantitative figures (Boolean)
render_logos = True # Whether to render sequence logos (Boolean)
save_logos = True # Whether to save sequence logos (Boolean); render_logos must be True
dpi = 200 # DPI for saved figures
save_data = True # Whether to save output data (Boolean)
delete_downloads = False # Whether to delete downloaded models/data after use (Boolean)
# TODO: view_dendrogram = False for even faster debugging

# If starting from scratch, set all to False:
load_previous_library = False # Whether to load previously-generated x_mut and y_mut (Boolean)
load_previous_attributions = False # Whether to load previously-generated attribution maps (Boolean)
load_previous_linkage = False # Whether to load previously-generated linkage matrix (Boolean)

# =============================================================================
# Initial setup based on user settings
# =============================================================================
if save_data:# or load_previous_library or load_previous_attributions or load_previous_linkage:
    py_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(py_dir, f'outputs_deepstarr_test_attributions')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_figs:
        save_path_figs = save_path
    else:
        save_path_figs = None
else:
    save_path = None

if load_previous_library is True: # save_data must be True
    x_mut = np.load(os.path.join(save_path, 'x_mut.npy'))
    y_mut = np.load(os.path.join(save_path, 'y_mut.npy'))

if load_previous_attributions: # save_data must be True
    attributions = np.load(os.path.join(save_path, f'attributions_{attribution_method}.npy'))

if load_previous_linkage: # save_data must be True
    link_method = 'ward'
    linkage = np.load(os.path.join(save_path, f'hierarchical_linkage_{link_method}_{attribution_method}.npy'))
    clusterer_hier = Clusterer(
        attributions,
        gpu=gpu
    )

# =============================================================================
# Download and import DeepSTARR model and data
# =============================================================================
import h5py

def download_if_not_exists(url, filename):
    """Download a file if it doesn't exist locally."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urlretrieve(url, filename)
    else:
        print(f"Using existing {filename}")

# Define URLs and filenames
files = {
    'deepstarr.model.json': 'https://www.dropbox.com/scl/fi/y1mwsqpv2e514md9t68jz/deepstarr.model.json?rlkey=cdwhstqf96fibshes2aov6t1e&st=9a0c5skz&dl=1',
    'deepstarr.model.h5': 'https://www.dropbox.com/scl/fi/6nl6e2hofyw70lh99h3uk/deepstarr.model.h5?rlkey=hqfnivn199xa54bjh8dn2jpaf&st=l4jig4ky&dl=1',
    'deepstarr_data.h5': 'https://www.dropbox.com/scl/fi/cya4ntqk2o8yftxql52lu/deepstarr_data.h5?rlkey=5ly363vqjb3vaw2euw2dhsjo3&st=6eod6fg8&dl=1'
}

# Download files
for filename, url in files.items():
    download_if_not_exists(url, filename)

keras_model_weights = 'deepstarr.model.h5'
keras_model_json = 'deepstarr.model.json'

with h5py.File('deepstarr_data.h5', 'r') as dataset:
    X_test = np.array(dataset['x_test']).astype(np.float32)

# TODO: replace above with zenodo wget

from keras.models import load_model, model_from_json
keras_model = model_from_json(open(keras_model_json).read(),custom_objects={'Functional': tf.keras.Model})
np.random.seed(113)
random.seed(0)
keras_model.load_weights(keras_model_weights)
model = keras_model
num_tasks = 2  # Dev [0] and Hk [1]

alphabet = ['A','C','G','T']

x_ref = X_test[seq_index]
x_ref = np.expand_dims(x_ref,0)

# Define mutagenesis window for sequence
seq_length = x_ref.shape[1]
mut_window = [0, seq_length]  # [start_position, stop_position]

# Forward pass to get output for the specific head
output = model(x_ref)
pred = model.predict(x_ref)[task_index]
print(f"\nWild-type prediction: {pred[0][0]}")

# =============================================================================
# SQUID API
# Create in silico mutagenesis library
# =============================================================================
if load_previous_library is False:
    # Set up predictor class for in silico MAVE
    pred_generator = squid.predictor.ScalarPredictor(pred_fun=model.predict_on_batch,
        task_idx=task_index,
        batch_size=512
    )

    # Set up mutagenizer class for in silico MAVE
    mut_generator = squid.mutagenizer.RandomMutagenesis(
        mut_rate=mut_rate
    )

    # Generate in silico MAVE
    mave = squid.mave.InSilicoMAVE(
        mut_generator,
        pred_generator,
        seq_length,
        mut_window=mut_window
    )
    x_mut, y_mut = mave.generate(x_ref[0], num_sim=num_seqs)

    if save_data:
        np.save(os.path.join(save_path, 'x_mut.npy'), x_mut)
        np.save(os.path.join(save_path, 'y_mut.npy'), y_mut)

# plot histogram of deepnet predictions
fig = squid.impress.plot_y_hist(
    y_mut,
    save_dir=save_path_figs
)
if not save_path_figs:
    plt.show()

# =============================================================================
# SEAM API
# Compile sequence analysis data into a standardized format
# =============================================================================
# Initialize compiler
compiler = Compiler(
    x=x_mut,
    y=y_mut,
    x_ref=x_ref,
    y_bg=None,
    alphabet=alphabet,
    gpu=gpu
)

mave_df = compiler.compile()
ref_index = 0 # index of reference sequence (zero by default)
print(mave_df)

if save_data:
    mave_df.to_csv(os.path.join(save_path, 'mave_df.csv'), index=False)

# =============================================================================
# SEAM API
# Compute attribution maps for each sequence in library
# =============================================================================
if load_previous_attributions is False:
    attributer = Attributer(
        model,
        method=attribution_method,
        task_index=task_index
    )

    # Show params for specific method
    attributer.show_params(attribution_method)

    t1 = time.time()
    attributions = attributer.compute(
        x=x_mut,
        x_ref=x_ref,
        save_window=None,
        batch_size=256,
        #log2fc=True,
        gpu=gpu
    )
    t2 = time.time() - t1
    print('Attribution time:', t2)

    if save_data:
        np.save(os.path.join(save_path, f'attributions_{attribution_method}.npy'), attributions)

# Render logo of attribution map for reference sequence
ref_index = 0
if render_logos is True:
    reference_logo = BatchLogo(attributions[ref_index:ref_index+1],
        alphabet=alphabet,
        fig_size=[20,2.5],
        center_values=True,
        batch_size=1
    )

    reference_logo.process_all()

    fig, ax = reference_logo.draw_single(
        ref_index,
        border=False
    )
    if save_logos:
        fig.savefig(os.path.join(save_path, 'reference_logo_%s_gpu%s.png' % (attribution_method, gpu)), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()