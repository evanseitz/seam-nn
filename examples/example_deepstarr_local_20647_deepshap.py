"""
# SEAM example of local library using DeepSTARR enhancer to annotate all TFBSs and biophysical states (Fig.2a)

Model:
    - DeepSTARR

Parameters:
    - 30,000 sequences
    - 10% mutation rate
    - Integrated gradients # TODO deepshap?
    - Hierarchical clustering (ward)

Tested using:
    - Python 3.11.8
    - TensorFlow 2.16.1
    - NumPy 1.26.1
    - SEAM 0.4.3
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
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # TODO: turn this off
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
num_seqs = 10000#30000 # number of sequences to generate
attribution_method = 'deepshap' # {saliency, smoothgrad, intgrad, deepshap, ism} # TODO: deepshap under construction

gpu = len(tf.config.list_physical_devices('GPU')) > 0 # Whether to use GPU (Boolean)
save_figs = True # Whether to save quantitative figures (Boolean)
render_logos = True # Whether to render sequence logos (Boolean)
save_logos = True # Whether to save sequence logos (Boolean)
dpi = 200 # DPI for saved figures
save_data = True # Whether to save output data (Boolean)
delete_downloads = False # Whether to delete downloaded models/data after use (Boolean)
# TODO: view_dendrogram for even faster debugging

load_previous_library = True # Whether to load previously-generated x_mut and y_mut (Boolean)
load_previous_attributions = True # Whether to load previously-generated attribution maps (Boolean)
load_previous_linkage = True # Whether to load previously-generated linkage matrix (Boolean)

# =============================================================================
# Initial setup based on user settings
# =============================================================================
if save_data:
    py_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(py_dir, f'outputs_deepstarr_local_{seq_index}_{attribution_method}')
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
    pred_generator = squid.predictor.ScalarPredictor(
        pred_fun=model.predict_on_batch,
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
        gpu=gpu
    )
    t2 = time.time() - t1
    print('Attribution time:', t2)

    if save_data:
        np.save(os.path.join(save_path, f'attributions_{attribution_method}.npy'), attributions)

# Render logo of attribution map for reference sequence
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
        fig.savefig(os.path.join(save_path, 'reference_logo.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# =============================================================================
# SEAM API
# Cluster attribution maps directly using Hierarchical Clustering
# =============================================================================
if load_previous_linkage is False:
    clusterer_hier = Clusterer(
        attributions,
        gpu=gpu
    )

    # Perform hierarchical clustering directly on attribution maps
    link_method = 'ward'
    linkage = clusterer_hier.cluster(
        method='hierarchical',
        link_method=link_method
    )

    if save_data:
        np.save(os.path.join(save_path, f'hierarchical_linkage_{link_method}_{attribution_method}.npy'), linkage)

# Cut tree to get a specific number of clusters
n_clusters = 30

labels_n, cut_level = clusterer_hier.get_cluster_labels(
    linkage,
    criterion='maxclust',
    n_clusters=n_clusters
)

# Plot dendrogram to visualize hierarchy
clusterer_hier.plot_dendrogram(
    linkage,
    cut_level=cut_level,
    figsize=(15, 10),
    leaf_rotation=90,
    leaf_font_size=8,
    save_path=save_path_figs,
    dpi=dpi
)

# =============================================================================
# SEAM API
# Generate meta-explanations and related statistics
# =============================================================================
sort_method = 'median' # sort clusters by median DNN prediction (default)

# Initialize MetaExplainer
meta = MetaExplainer(
    clusterer=clusterer_hier,
    mave_df=mave_df,
    attributions=attributions,
    sort_method=sort_method,
    ref_idx=0,
    mut_rate=mut_rate
)

# Plot boxplot distribution for DNN predictions associated with each cluster
pred_boxplots = meta.plot_cluster_stats(
    plot_type='box',
    metric='prediction',
    show_ref=True,
    show_fliers=False,
    save_path=save_path_figs,
    dpi=dpi
)

# Plot occupancy distribution for attribution maps associated with each cluster
pred_boxplots = meta.plot_cluster_stats(
    plot_type='bar',
    metric='occupancy',
    show_ref=True,
    show_fliers=False,
    save_path=save_path_figs,
    dpi=dpi
)

# Show sequences from a given cluster
seqs_cluster = meta.show_sequences(0) # e.g., cluster index 0
seqs_cluster.head()

# Retrieve attribution maps assigned to a given cluster
maps_cluster = meta.get_cluster_maps(0)  # e.g., cluster index 0
print("%s attribution maps in cluster" % maps_cluster.shape[0])

# Generate Mechanism Summary Matrix (MSM)
msm = meta.generate_msm(
    gpu=gpu
)

# Plot MSM with different options
view_window = [50,170]

meta.plot_msm(column='Entropy',
    square_cells=True,
    view_window=view_window,
    save_path=save_path_figs,
    dpi=dpi
)

meta.plot_msm(column='Reference',
    square_cells=True,
    view_window=view_window,
    save_path=save_path_figs,
    dpi=dpi
)

# =============================================================================
# SEAM API
# Plot meta-attribution maps for each cluster
# =============================================================================
# Generate attribution logos
if render_logos is True:
    logo_type = 'average' # {average, pwm, enrichment}

    meta_logos = meta.generate_logos(logo_type=logo_type,
        background_separation=False,
        font_name='sans',
        center_values=True
    )

    if save_logos is True:
        save_path_logos = os.path.join(save_path, 'cluster_logos')
        if not os.path.exists(save_path_logos):
            os.makedirs(save_path_logos)
        save_path_logos_average = os.path.join(save_path_logos, logo_type)
        if not os.path.exists(save_path_logos_average):
            os.makedirs(save_path_logos_average)

    for cluster_index in tqdm(range(n_clusters), desc='Generating logos'):
        fig, ax = meta_logos.draw_single(
            cluster_index,
            fixed_ylim=True, # fixed y-axis limits as defined over all cluster-averaged logos
            fig_size=(10, 2.5),
            border=False,
            view_window=view_window
        )
        if save_logos:
            fig.savefig(os.path.join(save_path_logos_average, 'cluster_%s.png' % cluster_index), facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # Generate variability logo, representing the overlap of all averaged cluster logos
    fig, ax = meta_logos.draw_variability_logo(
        fig_size=(10,2.5),
        view_window=view_window
    )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos_average, '_variability_logo.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# =============================================================================
# SEAM API
# Background separation
# =============================================================================
# Plot variation in attribution values across attribution maps for each nucleotide position
meta.plot_attribution_variation(
    scope='clusters',  # {'all', 'clusters'}
    metric='std',  # {'std', 'var'}
    view_window=view_window,
    save_path=save_path_figs,
    dpi=dpi
)

# Separate individual clusters from average background over clusters
background_multiplier = 0.5  # default threshold factor for background separation

# View clusters without background
meta_logos_no_bg = meta.generate_logos(
    logo_type='average',
    background_separation=True,
    mut_rate=mut_rate,
    entropy_multiplier=background_multiplier,
    figsize=(20, 2.5)
)

if render_logos is True:
    if sort_method is not None:
        ref_cluster = meta.membership_df.loc[ref_index, 'Cluster_Sorted']
    else:
        ref_cluster = ref_index

    # Compare reference map with and without noise reduction andbackground separation
    # Reference logo
    fig, ax = reference_logo.draw_single(
        0,
        fixed_ylim=False,
        fig_size=(10,2.5),
        border=False,
        view_window=view_window
        )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos, '1_reference_logo.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Reference cluster: noise reduction via averaging
    fig, ax = meta_logos.draw_single(
        ref_cluster,
        fixed_ylim=False,
        fig_size=(10,2.5),
        border=False,
        view_window=view_window
        )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos, '2_reference_cluster.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Reference cluster: noise reduction and background separation
    fig, ax = meta_logos_no_bg.draw_single(
        ref_cluster,
        fixed_ylim=False,
        fig_size=(10,2.5),
        border=False,
        view_window=view_window
        )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos, '3_reference_cluster_no_bg.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# =============================================================================
# SEAM API
# Identify positions of individual motifs
# =============================================================================
identifier = Identifier(
    msm,
    meta,
    column='Entropy'
)

identifier.plot_msm_covariance_triangular(
    xtick_spacing=5,
    view_window=view_window,
    save_path=save_path_figs,
    dpi=dpi
)

# Cluster the covariance matrix to isolate motifs
identifier.cluster_msm_covariance(
    method='average',
    n_clusters=4  # or e.g., cut_height=0.3
)

# Plot dendrogram of clustered covariance matrix
fig, ax = identifier.plot_msm_covariance_dendrogram(
    figsize=(8,8),
    save_path=save_path_figs,
    dpi=dpi
)

fig, ax = identifier.plot_msm_covariance_square(
    view_window=view_window,
    show_clusters=True,
    view_linkage_space=False,  # Default: shows in nucleotide position space
    save_path=save_path_figs,
    dpi=dpi
)

# Set threshold and update state
identifier.set_entropy_multiplier(0.8)  # Adjust until desired TFBS positions are visually enclosed

# Plot MSM with TFBS positions enclosed
fig, ax = identifier.meta_explainer.plot_msm(
    column='Entropy',
    square_cells=True,
    view_window=view_window,
    show_tfbs_clusters=True,
    tfbs_clusters=identifier.tfbs_clusters,
    cov_matrix=identifier.cov_matrix,
    row_order=identifier.row_order,
    revels=identifier.revels,
    entropy_multiplier=identifier.entropy_multiplier,
    save_path=save_path_figs,
    dpi=dpi
)

# Get TFBS positions with active clusters
tfbs_positions = identifier.get_tfbs_positions(
    active_clusters=meta.active_clusters_by_tfbs
)
print("\nTFBS Positions and Active Clusters:")
print(tfbs_positions)

# Get and display state matrix
state_matrix = identifier.get_state_matrix(
    active_clusters=meta.active_clusters_by_tfbs
)

# Visualize state matrix
fig, ax = identifier.plot_state_matrix(
    active_clusters=meta.active_clusters_by_tfbs,
    mode='binary',
    orientation='horizontal',
    save_path=save_path_figs,
    dpi=dpi
)

fig, ax = identifier.plot_state_matrix(
    active_clusters=meta.active_clusters_by_tfbs,
    mode='continuous',
    orientation='horizontal',
    save_path=save_path_figs,
    dpi=dpi
)

# =============================================================================
# SEAM API
# Plot meta-attribution maps after background separation and TFBS identification
# =============================================================================
position_lists = tfbs_positions['Positions'].tolist()
active_clusters = tfbs_positions['Active_Clusters'].tolist()

# Create fixed colors for each TFBS
tfbs_colors = [plt.cm.Pastel1(i % 9) for i in range(len(position_lists))]

if render_logos is True:
    if save_logos is True:
        save_path_logos_average_no_bg = os.path.join(save_path_logos, 'average_no_bg')
        if not os.path.exists(save_path_logos_average_no_bg):
            os.makedirs(save_path_logos_average_no_bg)

    for cluster_index in tqdm(range(n_clusters), desc='Generating logos'):
        # Only include position lists and their corresponding colors for active TFBSs
        active_positions = []
        active_colors = []
        for positions, clusters, color in zip(position_lists, active_clusters, tfbs_colors):
            if cluster_index in clusters:
                active_positions.append(positions)
                active_colors.append(color)

        # Plot clusters with background removed and active TFBSs highlighted
        fig, ax = meta_logos_no_bg.draw_single(
            cluster_index,
            fixed_ylim=True,
            fig_size=(10, 2.5),
            border=False,
            view_window=view_window,
            highlight_ranges=active_positions if active_positions else None,
            highlight_colors=active_colors if active_positions else None
        )
        if save_logos:
            fig.savefig(os.path.join(save_path_logos_average_no_bg, 'cluster_%s.png' % cluster_index), facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    # Generate background-separated variability logo
    fig, ax = meta_logos_no_bg.draw_variability_logo(
        fig_size=(10,2.5),
        view_window=view_window
    )
    if save_logos:
        fig.savefig(os.path.join(save_path_logos_average_no_bg, '_variability_logo.png'), facecolor='w', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# =============================================================================
# Clean up downloaded files if requested
# =============================================================================
if delete_downloads:
    print("Cleaning up downloaded files...")
    for filename in files.keys():
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Deleted {filename}")




