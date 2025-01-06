import os, sys, time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from tqdm import tqdm
py_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(py_dir)


# =============================================================================
# User parameters
# =============================================================================

# to do: add umap, pca, etc.

gpu = True

attr_map = 'deepshap'
alphabet = ['A','C','G','T']
#data_dir = os.path.join(py_dir, 'outputs_AP1_mut0_seq13748') # location of required data (see below)
#dir_name = 'examples_deepmel2/outputs_REVO-idx45-class16-tw4-wl5101520-tm20-tr3'
dir_name = 'examples_chrombpnet/outputs_local_PPIF_promoter_counts/mut_pt01/N100k_fold0'
data_dir = os.path.join(parent_dir, 'examples/%s' % dir_name)
dist_fname = os.path.join(data_dir, 'distances.dat')
batch_size = 10000
link_method = 'ward' # linkage method for hierarchical clustering

# =============================================================================
# Load in required data
# =============================================================================

mave_fname = os.path.join(data_dir, 'mave.csv')
mave = pd.read_csv(mave_fname)
nS = len(mave)
seq_length = len(mave['Sequence'][0]) # or just set manually
dim = (seq_length)*len(''.join(alphabet))
maps_fname = os.path.join(data_dir, 'maps_%s.npy' % attr_map)
maps = np.load(maps_fname, mmap_mode='r')
maps = maps.reshape((nS, seq_length*len(alphabet)))

# =============================================================================
# Calculate distance matrix
# =============================================================================
if 1: # if distances not previously calculated
    print('Computing distances...')

    if gpu is False:
        t1 = time.time()
        # compute distances:
        D_upper = np.zeros(shape=(nS, nS))
        p = 2. # Minkowski distance metric: p1=Manhattan, p2=Euclidean, etc.
        for i in tqdm(range(0, nS), desc='Index'):
            for j in range(0, nS):
                if i < j:
                    D_upper[i,j] = np.linalg.norm(maps[i,:]-maps[j,:])
        D = D_upper + D_upper.T - np.diag(np.diag(D_upper))
        D_flat = squareform(D)
        D_flat_CPU = D_flat # for comparison below only
        t2 = time.time()
        print('Distances time:', t2-t1)

    elif gpu is True:
        import tensorflow as tf

        def distance_batch(A, B, batch_size=4096, output_file='distances.dat'):
            A = tf.cast(A, tf.float32)
            B = tf.cast(B, tf.float32)
            num_A = A.shape[0]
            num_B = B.shape[0]
            distance_matrix = np.memmap(output_file, dtype=np.float32, mode='w+', shape=(num_A, num_B))

            def pairwise_distance(A_batch, B_batch):
                v_A = tf.expand_dims(tf.reduce_sum(tf.square(A_batch), 1), 1)
                v_B = tf.expand_dims(tf.reduce_sum(tf.square(B_batch), 1), 1)
                p1 = tf.reshape(v_A, (-1, 1))
                p2 = tf.reshape(v_B, (1, -1))
                dist_squared = tf.add(p1, p2) - 2 * tf.matmul(A_batch, B_batch, transpose_b=True)
                dist_squared = tf.maximum(dist_squared, 0.0) # ensure non-negative values before taking sqrt
                return tf.sqrt(dist_squared)

            for i in tqdm(range(0, num_A, batch_size), desc='Distance batch'):
                A_batch = A[i:i + batch_size]
                for j in range(0, num_B, batch_size):
                    B_batch = B[j:j + batch_size]
                    distances = pairwise_distance(A_batch, B_batch).numpy()
                    distance_matrix[i:i + batch_size, j:j + batch_size] = distances # save intermediate results to memmap

            distance_matrix.flush() # flush memmap to ensure all data is written to disk

            return distance_matrix

        
        t1 = time.time()
        D = distance_batch(tf.Variable(maps), tf.Variable(maps), batch_size=batch_size, output_file=dist_fname)
        D = np.array(D).astype(np.float32)
        np.fill_diagonal(D, 0)
        D = (D + D.T) / 2 # matrix may be antisymmetric due to precision errors
        D_flat = squareform(D)
        os.remove(dist_fname)
        t2 = time.time()
        print('Distances time:', t2-t1)

    #print(np.allclose(D_flat_CPU, D_flat, atol=1e-8)) # sanity check

    if 0:
        np.save(os.path.join(data_dir, 'dist_upper_flat.npy'), D_flat)

else:
    D_flat = np.load(os.path.join(data_dir, 'dist_upper_flat.npy'))

# =============================================================================
# Hierarchical clustering on distance matrix
# =============================================================================
if 1:
    print('Computing hierarchical clusters...')
    t1 = time.time()
    linkage = hierarchy.linkage(D_flat, method=link_method, metric='euclidean')
    np.save(os.path.join(data_dir, 'hierarchical_linkage_%s_%s.npy' % (link_method, attr_map)), linkage)
    t2 = time.time()
    print('Linkage time:', t2-t1)


#if 0: # correlate the pairwise distances of all samples to those implied by the hierarchical clustering
#    from scipy.spatial import distance
#    c, coph_dists = hierarchy.cophenet(linkage, distance.pdist(maps))
#    print(c)
