import os, sys
sys.dont_write_bytecode = True
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import random
random.seed(123)

py_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(py_dir)
inputs_dir = os.path.join(parent_dir, 'examples/outputs_local_chrombpnet_PPIF_promoter/mut_pt10/seed2_N100k_allFolds')
#inputs_dir = os.path.join(parent_dir, 'examples/outputs_local_enformer_DNase_PPIF_promoter/seed2')


mave = pd.read_csv(os.path.join(inputs_dir, 'mave.csv'))

manifold = np.load(os.path.join(inputs_dir, 'manifold_deepshap_tsne.npy'))
#manifold = np.load('/Volumes/SEITZ_2023/Education/5_CSHL/6_SEAM/seam-nn/examples/outputs_local_chrombpnet_PPIF_promoter/mut_pt10/seed2_N100k_allFolds/manifold_deepshap_umap.npy')
print(manifold.shape)

#clusters = pd.read_csv(os.path.join(inputs_dir, 'clusters_kmeans/all_clusters.csv'))
clusters = pd.read_csv(os.path.join(inputs_dir, 'clusters_hierarchical/maxd1pt5/all_clusters.csv'))
#clusters = pd.read_csv(os.path.join(inputs_dir, 'clusters_kmeans_k200/all_clusters.csv'))


num_clusters = len(np.unique(clusters['Cluster']))
print('Clusters:', num_clusters)
palette = sns.color_palette('Spectral', n_colors=num_clusters)


if 0: # shuffle colormap
    shuffle_idx = [[i] for i in range(num_clusters)]
    random.shuffle(shuffle_idx)
    shuffle_idx = np.array(shuffle_idx)
    palette_shuffle = []
    for i in shuffle_idx:
        palette_shuffle.append(palette[int(i)])



if 1: # select specific cluster to visualize
    for cl in range(num_clusters):
        #cl = 1 #[1, 2, ..., 10]
        cluster_idx = clusters.loc[(clusters['Cluster'] == cl)].index
        if 0:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
            ax1.scatter(manifold_orig[:,0], manifold_orig[:,1], s=1, c='k', zorder=-100)
            ax1.scatter(manifold_orig[cluster_idx,0], manifold_orig[cluster_idx,1], s=1, c='r', zorder=0)
        else:
            fig, ax2 = plt.subplots(figsize=(5,5))
        ax2.scatter(manifold[:,0], manifold[:,1], s=1, c='k', zorder=-100)
        ax2.scatter(manifold[cluster_idx,0], manifold[cluster_idx,1], s=1, c='r', zorder=0)
        fig.suptitle('Cluster %s' % cl)
        plt.savefig('/Users/evanseitz/Documents/figures/cluster_%s.png' % cl, dpi=100)
        plt.close()
elif 0: # visualize all clusters
    plt.scatter(manifold[:,0], manifold[:,1], s=1, c=clusters['Cluster'],
                cmap=mpl.colors.ListedColormap(palette_shuffle)) #, cmap='tab10')
    plt.colorbar()
else: # visualize DNN scores

    if 1: # ascending view
        zord = np.argsort(np.array(mave['DNN']))
    elif 0: # descending view
        zord = np.argsort(np.array(mave['DNN']))[::-1]
    else:
        zord = np.arange(len(mave))

    plt.scatter(manifold[zord][:,0], manifold[zord][:,1], c=mave['DNN'][zord], s=1, cmap='jet', linewidth=0)
    plt.colorbar()

plt.tight_layout()
plt.show()
#plt.savefig(os.path.join(outputs_dir, 'all_clusters.png'), facecolor='w', dpi=dpi)
plt.close()