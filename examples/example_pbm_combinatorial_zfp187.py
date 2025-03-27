"""
SEAM analysis of combinatorial-complete sequence library using PBM data (Fig.TBD) # TODO double check this

Model:
    - PBM oracle (no deep learning model)

Parameters:
    - 4^8 = 65,536 sequences
    - Empirical attribution maps via ISM
    - Hierarchical clustering (ward)

Tested using:
    - Python 3.11.8
    - TensorFlow 2.16.1
    - NumPy 1.26.1
    - SEAM 0.4.3
"""

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import squid
import pandas as pd
from tqdm import tqdm
import gdown
from itertools import product
if 1: # Use this for local install (must 'pip uninstall seam' first)
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # TODO: turn this off
from seam import Compiler, Attributer, Clusterer, MetaExplainer

'''
print(sys.version) # 3.10.12
print(tf.__version__) # 2.16.1
print(np.__version__) # 1.26.1
print(seam.__version__) # 0.4.3
'''

# =============================================================================
# Overhead user settings
# =============================================================================
protein = 'hnf4a' # {'zfp187', 'hnf4a'}
attribution_method = 'ism' # analogous to empirical attribution

gpu = len(tf.config.list_physical_devices('GPU')) > 0 # Whether to use GPU (Boolean)
save_figs = True # Whether to save quantitative figures (Boolean)
render_logos = True # Whether to render sequence logos (Boolean)
save_logos = True # Whether to save sequence logos (Boolean)
dpi = 200 # DPI for saved figures
save_data = True # Whether to save output data (Boolean)

load_previous_library = False # Whether to load previously-generated x_mut and y_mut (Boolean)
load_previous_attributions = False # Whether to load previously-generated attribution maps (Boolean)
load_previous_linkage = False # Whether to load previously-generated linkage matrix (Boolean)

# =============================================================================
# Initial setup based on user settings
# =============================================================================
if save_data:
    py_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(py_dir, f'outputs_pbm_combinatorial_{protein}')
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
# Download and convert PBM data
# =============================================================================
seq_length = 8
mut_window = [0, seq_length]  # [start_position, stop_position]
alphabet = ['A','C','G','T']

if load_previous_library is False:
    if protein == 'zfp187': # e-score replicates in columns 3 and 6
        # Data accessed (2024): http://the_brain.bwh.harvard.edu/uniprobe/details2.php?id=82
        file_id = '1dPJEHggnPLVhi9kpHSha5PXZKSO2tMxZ'
        output = 'Zfp187_2626_contig8mers.txt'
    elif protein == 'hnf4a': # e-score replicates in columns 3 and 6
        # Data accessed (2024): http://the_brain.bwh.harvard.edu/uniprobe/details2.php?id=66
        file_id = '1shZPti6-YACz3BvvDJ_FsRjx4IazQROF'
        output = 'Hnf4a_2640_contig8mers.txt'
    gdown.download(id=file_id, output=output, quiet=True)
    df = pd.read_csv(output, delimiter='\t', header=None)

    # Convert PBM data to sequence-function library
    kmers = [''.join(mer) for mer in product(alphabet, repeat=seq_length)]
    forward_seqs = df.iloc[:, 0].str
    reverse_seqs = df.iloc[:, 1].str
    e_scores = []
    
    # Need to avoid the 256/2 = 128 additional entries, which are palindromes
    for mer in tqdm(kmers, desc='kmer'):
        # Search in forward sequences
        forward_matches = forward_seqs.contains(mer)
        if forward_matches.any():
            idx = forward_matches.idxmax()
            e_scores.append((df.iloc[idx, 3] + df.iloc[idx, 6])/2) # average e-scores from replicates
        else:
            # Search in reverse sequences
            reverse_matches = reverse_seqs.contains(mer)
            idx = reverse_matches.idxmax()
            e_scores.append((df.iloc[idx, 3] + df.iloc[idx, 6])/2) # average e-scores from replicates

    L, A = seq_length, len(alphabet)
    x_mut = np.zeros(shape=(A**L, L, A), dtype='int8')

    for i in range(A**L):
        x_mut[i,:,:] = squid.utils.seq2oh(kmers[i])

    y_mut = np.array(e_scores, dtype='float32')
    y_mut = np.expand_dims(y_mut, axis=1)

    if save_data:
        np.save(os.path.join(save_path, 'x_mut.npy'), x_mut)
        np.save(os.path.join(save_path, 'y_mut.npy'), y_mut)

# Plot histogram of E-scores
plt.hist(y_mut, bins=100)
plt.xlabel('E-score')
plt.ylabel('Frequency')
if save_figs:
    plt.savefig(os.path.join(save_path_figs, 'escore_distribution.png'), dpi=dpi)
else:
    plt.show()

# =============================================================================
# SEAM API
# Compile sequence analysis data into a standardized format
# =============================================================================
# Initialize compiler
compiler = Compiler(
    x=x_mut,
    y=y_mut,
    x_ref=None,
    y_bg=None,
    alphabet=alphabet,
    gpu=gpu
)

mave_df = compiler.compile()
print(mave_df)

if save_data:
    mave_df.to_csv(os.path.join(save_path, 'mave_df.csv'), index=False)

# =============================================================================
# SEAM API
# Compute attribution maps for each sequence in library
# =============================================================================
if load_previous_attributions is False:
    # Create wrapper for calling sequence-function library as an oracle
    class OracleWrapper:
        """Wrapper to make oracle function look like a TensorFlow model."""
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
            # Create lookup dictionary for fast sequence matching
            self.lookup = {}
            # Convert sequences to hashable tuples efficiently
            keys = [tuple(map(tuple, seq)) for seq in X.astype(int)]
            # Create lookup dictionary in one go
            self.lookup = dict(zip(keys, Y.ravel()))

        def __call__(self, x):
            """Find the experimental measurement for input sequence(s)."""
            # Convert TensorFlow tensor to numpy if needed
            if tf.is_tensor(x):
                x = x.numpy()

            if len(x.shape) == 2:  # Single sequence
                x = x[np.newaxis, ...]

            # Convert sequences to hashable tuples
            keys = [tuple(map(tuple, seq)) for seq in x.astype(int)]

            # Use list comprehension for lookup
            try:
                results = [self.lookup[key] for key in keys]
            except KeyError:
                raise ValueError("The input sequence does not exist in X.")

            return tf.convert_to_tensor(results, dtype=tf.float32)

    oracle = OracleWrapper(x_mut, y_mut)

    if 0:  # example single-instance usage of OracleWrapper
        x_test = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])  # i.e., X[0]
        y_wrapper = oracle(x_test)
        print('OracleWrapper test:')
        print('x:', x_test)
        print('y:', y_wrapper.numpy())  # Convert tensor back to numpy for comparison

    attributer = Attributer(
        oracle,
        method=attribution_method,
        func=lambda x: x
    )

    t1 = time.time()
    attributions = attributer.compute(
        x=x_mut,
        gpu=gpu,
        log2fc=True,
        batch_size=1024
    )
    t2 = time.time() - t1
    print('Attribution time:', t2)

    if save_data:
        np.save(os.path.join(save_path, f'attributions_{attribution_method}.npy'), attributions)

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
n_clusters = 200

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
)

# Plot boxplot distribution for DNN predictions associated with each cluster
pred_boxplots = meta.plot_cluster_stats(
    plot_type='box',
    metric='prediction',
    show_ref=False,
    show_fliers=False,
    compact=True,
    save_path=save_path_figs,
    dpi=dpi
)

# Plot occupancy distribution for attribution maps associated with each cluster
pred_boxplots = meta.plot_cluster_stats(
    plot_type='bar',
    metric='occupancy',
    show_ref=False,
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
meta.plot_msm(column='Entropy',
    square_cells=True,
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
            fig_size=(10, 5),
            border=False,
        )
        if save_logos:
            fig.savefig(os.path.join(save_path_logos_average, 'cluster_%s.png' % cluster_index), facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()