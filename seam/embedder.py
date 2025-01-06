import os, sys
sys.dont_write_bytecode = True
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
py_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(py_dir)


nS = 100000
embedding = 'umap'
#embedding = 'phate'
#embedding = 'pca'
#embedding = 'tsne'
#embedding = 'ae'


if 0:
    maps_dir = 'outputs_local_chrombpnet_PPIF_promoter/mut_pt10/seed2_N100k_allFolds'
    seq_length = 768
    map_type = 'deepshap'
    start, stop = 278, 385 #186, 406
elif 1:
    maps_dir = 'examples_deepstarr/outputs_local_Ohler1-m0-seq20647'
    seq_length = 249
    map_type = 'deepshap'


inputs_dir = os.path.join(parent_dir, 'examples/%s' % maps_dir)
if embedding == 'ae':
    maps = np.load(os.path.join(inputs_dir, 'maps_%s.npy' % map_type))
else:
    maps = np.load(os.path.join(inputs_dir, 'maps_%s.npy' % map_type), mmap_mode='r')
mave = pd.read_csv(os.path.join(inputs_dir, 'mave.csv'))

if 0:
    crop = True
    maps_crop = maps[:,start:stop,:]
    maps = maps_crop
    seq_length = stop - start
    crop_label = '_crop_%s_%s' % (start, stop)
else:
    crop = False
    
maps = maps.reshape((nS, seq_length*4))


t0 = time.time()

if embedding == 'umap':
    import umap # conda install -c conda-forge umap-learn; may also need 'pip install pynndescent==0.5.8'
    fit = umap.UMAP() #n_components=2)
    U = fit.fit_transform(maps)

elif embedding == 'phate':
    import phate # pip install --user phate
    phate_op = phate.PHATE()
    #phate_op.set_params(gamma=0, t=120)
    U = phate_op.fit_transform(maps)

elif embedding == 'tsne':
    from openTSNE import TSNE # pip install openTSNE
    TSNE = TSNE(perplexity=30, metric="euclidean", n_jobs=8, random_state=42, exaggeration=2, verbose=True)
    U = TSNE.fit(maps)

elif embedding == 'pca':
    if 0:
        u,s,v = np.linalg.svd(maps.T, full_matrices=False)
        vals = s**2 #eigenvalues
        vecs = u #eigenvectors
        U = maps.dot(vecs)
    else:
        from sklearn.decomposition import PCA
        U = PCA(n_components=2).fit_transform(maps)

elif embedding == 'dm':
    from scipy.spatial.distance import squareform
    dists = np.load(os.path.join(inputs_dir, 'clusters_hierarchical/dist_upper_flat.npy'))
    dists = squareform(dists)
    sys.path.append(os.path.join(py_dir, 'archives'))
    import diffusion_maps
    #eps = None # automated discovery of Gaussian bandwidth
    eps = 1
    vals, U = diffusion_maps.op(dists, eps)

elif embedding == 'ae':
    sys.path.append(os.path.join(py_dir, 'archives'))
    if 0:
        import autoencoder_sklearn
        U = autoencoder_sklearn.AE(maps)
    else:
        import autoencoder_pytorch # conda install pytorch torchvision -c pytorch
        U, colors = autoencoder_pytorch.autoencoder(maps, mave['DNN'], 'AE')

elif embedding == 'vae':
    import autoencoder_pytorch
    U, colors = autoencoder_pytorch.autoencoder(maps, mave['DNN'], 'VAE')


t1 = time.time()
print('Embedding time:', t1-t0)


if crop is False:
    np.save(os.path.join(inputs_dir, 'manifold_%s_%s.npy' % (map_type, embedding)), U)
elif crop is True:
    np.save(os.path.join(inputs_dir, 'manifold_%s_%s%s.npy' % (map_type, embedding, crop_label)), U)