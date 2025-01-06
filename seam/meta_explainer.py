import os, sys
sys.dont_write_bytecode = True
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
from matplotlib.ticker import FormatStrFormatter
from Bio import motifs # 'pip install biopython'
import logomaker
import squid.utils as squid_utils # 'pip install squid-nn'
import seaborn as sns # pip install seaborn
from scipy.stats import entropy
from tqdm import tqdm
import gc
py_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(py_dir)

try:
    from . import utilities
except ImportError:
    import utilities

try:
    from . import impress
except ImportError:
    import impress

# to do items:
    # move 'generate_regulatory_df' into impress.py

# ZULU: clean up below function and the duplicate in _null.py and send both to utils.py
def generate_regulatory_df(mave, clusters_labels, clusters_idx, ref_seq): 
    # mave should already have sequence delimited
    mave['Cluster'] = clusters_labels
    seqs_all = mave['Sequence']
    seq_length = len(mave['Sequence'][0])
    regulatory_df = pd.DataFrame(columns = ['Cluster', 'Position', 'Reference', 'Consensus', 'Entropy'], index=range(len(clusters_idx)*seq_length))
    occ_idx = 0
    for k in tqdm(clusters_idx, desc='Clusters'):
        k_idxs =  mave.loc[mave['Cluster'] == k].index
        seqs_k = seqs_all[k_idxs]
        seq_array_cluster = motifs.create(seqs_k, alphabet=alphabet)
        pfm_cluster = seq_array_cluster.counts # position frequency matrix
        consensus_seq = pfm_cluster.consensus
        consensus_oh = squid_utils.seq2oh(consensus_seq, alphabet)
        # calculate matches to reference sequence:
        if ref_seq is not None:
            pos_occ1 = np.diagonal(np.eye(len(ref_seq))*ref_oh.dot(np.array(pd.DataFrame(pfm_cluster)).T))
            pos_occ1 = 100 - (pos_occ1/len(k_idxs))*100
        # calculate matches to cluster's consensus sequence:
        pos_occ2 = np.diagonal(np.eye(len(consensus_seq))*consensus_oh.dot(np.array(pd.DataFrame(pfm_cluster)).T))
        pos_occ2 = (pos_occ2/len(k_idxs))*100
        for pos in range(seq_length):
            occ = seqs_k.str.slice(pos, pos+1)
            vc = occ.value_counts(normalize=True).sort_index()
            regulatory_df.at[occ_idx, 'Cluster'] = k
            regulatory_df.at[occ_idx, 'Position'] = pos
            regulatory_df.at[occ_idx, 'Entropy'] = entropy(np.array(vc), base=2) # calculate Shannon entropy
            if ref_seq is not None:
                regulatory_df.at[occ_idx, 'Reference'] = pos_occ1[pos]
            else:
                regulatory_df.at[occ_idx, 'Reference'] = pd.NA
            regulatory_df.at[occ_idx, 'Consensus'] = pos_occ2[pos]
            occ_idx += 1
    return regulatory_df

# defaults
ref_idx = 0
plot_profiles = False
profiles_ylim = None

if 1:
    #dir_name = 'examples_deepstarr/outputs_local_AP1-m1-seq22612'
    #dir_name = 'examples_deepstarr/outputs_local_Ohler1-m0-seq20647'
    dir_name = 'examples_deepstarr/outputs_local_AP1-m0-seq13748/archives/AP1-rank0_origVersion'
    #dir_name = 'examples_deepstarr/outputs_local_Ohler1-m0-seq22627'
    #dir_name = 'examples_deepstarr/outputs_local_Ohler6-m0-seq171'
    #dir_name = 'examples_deepstarr/outputs_local_DRE-m2-seq4071'
    #dir_name = 'examples_deepstarr/outputs_local_AP1-m1-seq21069'
    #dir_name = 'examples_deepstarr/outputs_local_Ohler1-m0-seq20647/_other_librarySizes/librarySize_150k'
    ref_seq = 'First row'
    seq_stop = 249
    #attr_map = 'ism'
    #attr_map = 'saliency'
    #attr_map = 'smoothgrad'
    #attr_map = 'intgrad'
    attr_map = 'deepshap'
    #map_length = 249
elif 0:
    #dir_name = 'examples_deepstarr/outputs_global_AP1-N'
    dir_name = 'examples_deepstarr/outputs_global_CREB-NN'
    ref_seq = None
    seq_stop = 249
    attr_map = 'deepshap'
elif 0:
    dir_name = 'outputs_local_spliceAI_U2SURP/max'
    # U2SURP:
    ref_seq = 'GGGGGGACTACTGTATTATAAAAACTAATAATTGTCTTCTTTTCTCCCCTTTAAGTGGACGCGCCTTCAAGAAGAAATAGATCATCTGGTGGTAATACAGTTTTTTGCTCTTTTAATCGATAAATTT'
    seq_stop = 127
    attr_map = 'saliency'
elif 0:
    dir_name = 'examples_enformer/outputs_local_DNase_PPIF_promoter/seed2'
    #dir_name = 'examples_enformer/outputs_local_CAGE_PPIF_promoter/seed2'
    ref_seq = 'First row'
    attr_map = 'saliency'
    #map_length = 1024 # enhancer
    #map_length = 768 # promoter
elif 0:
    #dir_name = 'examples_chrombpnet/outputs_local_PPIF_promoter_counts/mut_pt10/seed2_N100k_allFolds'#fold4'
    #dir_name = 'examples_chrombpnet/outputs_local_PPIF_promoter_counts/mut_pt10/seed1and2_N200k_allFolds'
    dir_name = 'examples_chrombpnet/outputs_local_PPIF_promoter_counts/mut_pt40/N100k_fold0'
    ref_seq = 'First row'
    attr_map = 'deepshap'
    #map_length = 2048
elif 0:
    dir_name = 'outputs_local_enformer_DNase_PPIF_promoter/seed5_mutatePro_targetPro/promoter'
    ref_seq = 'First row'
    attr_map = 'saliency'
    #map_length = 768
elif 0:
    mode = 'quantity'
    #mode = 'profile' # {note for clipnet: cut=0.0001, 0.0002, 0.0004, 0.0008, 0.001, 0.003, 0.007}
    dir_name = 'examples_clipnet/outputs_local_PIK3R3/heterozygous_pt01/%s_nfolds9' % mode
    ref_seq = 'First row'
    attr_map = 'deepshap'
    #map_length = 1000
elif 0:
    #dir_name = 'examples_tfbind/outputs_8mer_SIX6'
    dir_name = 'examples_tfbind/outputs_8mer_Hnf4a' # 43728: GTAAACA
    #dir_name = 'examples_tfbind/outputs_8mer_Foxa2' # 11268: AGTAAACA
    #dir_name = 'examples_tfbind/outputs_8mer_Zfp187'
    #dir_name = 'examples_tfbind/outputs_8mer_Rfx3'
    #dir_name = 'examples_tfbind/outputs_8mer_Oct-1'
    ref_seq = 11268
    attr_map = 'ism'
    #map_length = 8
elif 0:
    tm = 20 # total mutations per EFS run
    #dir_name = 'examples_deepmel2/outputs_EFS-idx22-class16-traj0'
    #dir_name = 'examples_deepmel2/outputs_EFS-idx22-class16-traj20'
    #dir_name = 'examples_deepmel2/outputs_EFS-idx17-class16-traj20'
    #dir_name = 'examples_deepmel2/outputs_BEAM-idx45-class16-width2-muts16'
    #dir_name = 'examples_deepmel2/outputs_REVO-idx22-class16-tw3-wl20-tm20-tr5'
    dir_name = 'examples_deepmel2/outputs_REVO-idx45-class16-tw4-wl5101520-tm%s-tr3' % tm
    ref_seq = 'First row'
    attr_map = 'deepshap'
    #map_length = 500
    ref_idx = tm+1
elif 0:
    mode = 'profile'
    #mode = 'counts'
    dir_name = 'examples_procapnet/outputs_MYC_%s_fold0_r10' % mode
    ref_seq = 'First row'
    attr_map = 'deepshap'
    #map_length = 500
    profiles_ylim = [-769.2708160400391, 833.5737335205079] # profiles ylim over all seqs in library

aesthetic_lvl = 1 # {0, 1, 2} : modulates the aesthetic appearance of the logo, with the caveat that higher aesthetic levels have higher computation times

if 0:
    embedding = 'umap'#'pca'#'umap'
    if 0:
        clusterer = 'dbscan'
        n_clusters = None
    elif 1:
        clusterer = 'kmeans'
        n_clusters = 200
else:
    embedding = None
    clusterer = 'hierarchical'
    link_method = 'ward'
    if 0:
        cut_criterion = 'distance'
        cut_value = 8#10 # choose max cutoff distance for dendrogram
    else:
        cut_criterion = 'maxclust'
        cut_value = 200#100#500 # choose the desired number of clusters (same use as 'n_clusters' above)
    plot_dendro = True#False
    n_clusters = None # unused (see 'cut_value' above)

    
attr_logo_fixed = True
attr_logo_adaptive = True
seq_logo_pwm = False
seq_logo_enrich = False#True
plot_profiles = False # plot overlay of profiles associated with each cluster; requires file 'y_profiles.npy' corresponding to predictions for sequences in 'mave.csv' (both in same directory)
plot_bias = False#False # plot the per-cluster bias logos in the 'clusters_bias' folder
sort_median = True # sort the boxplots and regulatory matrix by the median DNN score of each cluster
bias_removal = False # removes bias based on subtracting the average of background attribution maps from each cluster 
mut_rate = .10 #.01 # mutation rate used for generating sequence library (if 'bias_removal' = True)
subset_clusters = False # focus on a specific subset of clusters (selected after first viewing all clusters; see below)
# ZULU: currently, sort_median must be false if subset_clusters is True

#sort_fig = 'visual'


inputs_dir = os.path.join(parent_dir, 'examples/%s' % dir_name)
if ref_seq == 'First row':
    ref_seq = pd.read_csv(os.path.join(inputs_dir, 'mave.csv'))['Sequence'][0]
    seq_stop = len(ref_seq)
    print('Sequence length:', seq_stop)
elif isinstance(ref_seq, int):
    ref_seq = pd.read_csv(os.path.join(inputs_dir, 'mave.csv'))['Sequence'][ref_seq]
    seq_stop = len(ref_seq)
    print('Sequence length:', seq_stop)


dpi = 200
eig1 = 0
eig2 = 1


def normalize(_d, to_sum=False, copy=True): # normalize all eigenvectors
    d = _d if not copy else np.copy(_d) # d is an (n x dimension) array
    d -= np.min(d, axis=0)
    d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0)) # normalize to [0,1]
    return d


mave_fname = os.path.join(inputs_dir, 'mave.csv')
mave = pd.read_csv(mave_fname)
nS = len(mave)
maps_fname = os.path.join(inputs_dir, 'maps_%s.npy' % attr_map)

'''if 0: # only needed if column names in mave.csv are inconsistent with those used here
    mave.rename(columns={'y_DNN': 'DNN', 'x': 'Sequence'}, inplace=True)
    mave.to_csv(os.path.join(inputs_dir, 'mave_new.csv'), index=False)
    print('')
    print("Columns in 'mave.csv' renamed and saved as 'mave_new.csv'. To replace the old version, manually overwrite 'mave.csv' with 'mave_new.csv', then disable this code before rerunning.")
    print('')
    exit()'''

#alphabet = ['A','C','G','T']
alphabet = sorted(list(set(mave['Sequence'][0:100].apply(list).sum())))


if embedding is not None:
    output_name = '%s_%s_%s' % (attr_map, embedding, clusterer)
else:
    if clusterer == 'hierarchical':
        output_name = '%s_%s_%s%s' % (attr_map, clusterer, cut_criterion, cut_value)
    else:
        output_name = '%s_%s' % (attr_map, clusterer)
if n_clusters is not None:
    output_name = output_name + str(int(n_clusters))


if 1: # use cropped attribution maps
    # PPIF promoter:#seq_start = 186 #seq_stop = 406 # PPIF enhancer:
    if 1: # AP1-rank0 (DeepSTARR)
        seq_start = 110
        seq_stop = 137
    elif 0: # Global (DeepSTARR)
        seq_start = 114
        seq_stop = 143#2#1#3 DRE, AP1-N, CREB-NN
    elif 0: # PPIF promoter (ChromBPNet)
        seq_start = 190
        seq_stop = 690


    manifold_fname = os.path.join(inputs_dir, 'manifold_%s_%s_crop_%s_%s.npy' % (attr_map, embedding, seq_start, seq_stop))
    if sort_median is True:
        outputs_dir = os.path.join(inputs_dir, 'clusters_%s_crop_%s_%s_sortMedian' % (output_name, seq_start, seq_stop))
    else:
        outputs_dir = os.path.join(inputs_dir, 'clusters_%s_crop_%s_%s' % (output_name, seq_start, seq_stop))
    fontsize = 8 # for 30-nt (cropped) sequence
    map_length = seq_stop - seq_start
else: # use full attribution maps
    manifold_fname = os.path.join(inputs_dir, 'manifold_%s_%s.npy' % (attr_map, embedding))
    seq_start = 0

    if sort_median is True:
        outputs_dir = os.path.join(inputs_dir, 'clusters_%s_sortMedian' % output_name)
    else:
        outputs_dir = os.path.join(inputs_dir, 'clusters_%s' % output_name)
    fontsize = 4 # for 249-nt sequence


if not os.path.exists(outputs_dir):
    os.mkdir(outputs_dir)

seq_length = len(mave['Sequence'][0])
dim = (seq_length)*len(''.join(alphabet))
if 'map_length' in locals():
    pass
else:
    map_length = seq_length
print('Attribution length:', map_length)

if map_length < 10:
    figsize = [3, 1.5]
elif 10 <= map_length < 50:
    figsize = [10, 2.5]
elif 50 <= map_length < 250:
    figsize = [20, 2.5]
else:
    figsize = [20, 1.5]

if embedding is not None:
    manifold = np.load(manifold_fname)
    manifold = normalize(manifold)

try:
    maps = np.load(maps_fname)
    if maps.ndim == 2: # ZULU, consistency
        maps = maps.reshape((nS, seq_length, len(''.join(alphabet))))
except: # try to load numpy mmemap
    maps = np.memmap(maps_fname, dtype='float32', mode='r', shape=(nS, seq_length, len(''.join(alphabet))))


seq_length = seq_stop - seq_start
try:
    if seq_start != 0 and seq_stop != len(mave['Sequence'][0]):
        mave['Sequence'].str.slice(seq_start, seq_stop)
        maps = maps[:,seq_start:seq_stop,:]
        if ref_seq is not None:
            ref_seq = ref_seq[seq_start:seq_stop]
            ref_oh = squid_utils.seq2oh(ref_seq, alphabet)
except:
    pass


print('Clustering data...')
if embedding is not None:
    if clusterer == 'kmeans':
        from sklearn.cluster import KMeans # pip install -U scikit-learn
        clusters = KMeans(init='k-means++', n_clusters=n_clusters, random_state=0, n_init=10)
        clusters.fit(manifold)
        clusters_labels = clusters.labels_ # membership of each element to a given cluster
    elif clusterer == 'dbscan':
        from sklearn.cluster import DBSCAN
        clusters = DBSCAN(eps=0.01, min_samples=10).fit(manifold)
        clusters_labels = clusters.labels_ # membership of each element to a given cluster

    clusters_idx = np.arange(max(clusters_labels)+1)

    if 1:
        plt.scatter(manifold[:,eig1], manifold[:,eig2], c=clusters_labels, cmap='tab10', #edgecolor='none',
                    s=2.5, linewidth=.1, edgecolors='k')
        #plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'all_clusters.png'), facecolor='w', dpi=600)
        plt.close()

    all_clusters_df = pd.DataFrame(columns = ['Cluster', 'Psi1', 'Psi2'], index=range(nS))
    all_clusters_df['Cluster'] = clusters_labels
    mave['Cluster'] = clusters_labels
    all_clusters_df['Psi1'].iat[0] = eig1
    all_clusters_df['Psi2'].iat[0] = eig2
    sort_fig = 'visual'


else: # cluster attribution maps without embedding them
    if clusterer == 'hierarchical':
        from scipy.cluster import hierarchy
        linkage = np.load(os.path.join(inputs_dir, 'hierarchical_linkage_%s_%s.npy' % (link_method,attr_map)))

        if plot_dendro is True: # plot dendrogram
            sys.setrecursionlimit(100000) # fix: https://stackoverflow.com/questions/57401033/how-to-fixrecursionerror-maximum-recursion-depth-exceeded-while-getting-the-st
            plt.figure(figsize=(15, 10)) # (25,10)
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('sample index')
            plt.ylabel('distance')
            with plt.rc_context({'lines.linewidth': 2}):
                hierarchy.dendrogram(linkage,
                                    leaf_rotation=90., # rotates the x-axis labels
                                    leaf_font_size=8., # font size for the x-axis labels
                                    )
            plt.xticks([])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.xlabel('Clusters') 
            plt.savefig(os.path.join(os.path.dirname(outputs_dir), 'dendrogram.png'), facecolor='w', dpi=dpi)
            plt.close()
        
        clusters_labels = hierarchy.fcluster(linkage, cut_value, criterion=cut_criterion) # membership of each element to a given cluster
        clusters_labels -= 1 # zero-index clusters to match other methods

        '''if cut_criterion == 'maxclust': # optionally, print the cut level (the height at which the dendrogram is cut)
            max_d = 0
            for i in range(1, len(linkage) + 1):
                if len(np.unique(hierarchy.fcluster(linkage, i, criterion='maxclust'))) == cut_value:
                    max_d = linkage[i-1, 2]  # The height (or distance) at the last merge to achieve the desired clusters
                    break
            print("Cut level for desired number of clusters:", max_d)'''
        
        clusters_idx = np.unique(clusters_labels) #np.arange(max(clusters_labels)) #
        all_clusters_df = pd.DataFrame(columns = ['Cluster'], index=range(nS))
        all_clusters_df['Cluster'] = clusters_labels
        mave['Cluster'] = clusters_labels
        sort_fig = None


if subset_clusters is True: # option to only focus on a subselection of clusters (default=0)
    ref_cluster = all_clusters_df.loc[ref_idx, 'Cluster']
    #cluster_sorted_indices = [ref_cluster, 71, 147, 96, 91, 50, 79, 0, 160, 93, 32, 80, 45, 131, 179, 55] # circle
    cluster_sorted_indices = [0,160,93,32,80,45,131,179,55,ref_cluster]
    ref_cluster = 9#0 # reset to match updated position in list above
    all_clusters_df = all_clusters_df[all_clusters_df['Cluster'].isin(cluster_sorted_indices)]
    mave = mave[mave['Cluster'].isin(cluster_sorted_indices)]
    clusters_idx = [item for item in clusters_idx if item in cluster_sorted_indices]
    clusters_labels = [item for item in clusters_labels if item in cluster_sorted_indices]
    # create a dictionary to map the original clusters to their indices in cluster_sorted_indices
    cluster_index_map = {cluster: index for index, cluster in enumerate(cluster_sorted_indices)}
    # sort clusters_idx to match the order in cluster_sorted_indices
    clusters_idx = sorted(clusters_idx, key=lambda x: cluster_index_map[x])
    # update 'all_clusters.csv' with new column containing sorted indices (for user reference only)
    mapping_dict = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(cluster_sorted_indices)}
    all_clusters_df['Cluster_sort'] = all_clusters_df['Cluster'].map(mapping_dict)
    sort_median = False
    sort_fig = 'predefined'


all_clusters_df.to_csv(os.path.join(outputs_dir, 'all_clusters.csv'), index=False)

mave['Sequence'] = mave['Sequence'].str.slice(seq_start, seq_stop)
#seqs_all = mave['Sequence'].str.slice(seq_start, seq_stop)
seq_array_background = motifs.create(mave['Sequence'], alphabet=alphabet)#seqs_all)
pfm_background = seq_array_background.counts

if 1: # render boxplots for all clusters
    print('Rendering histograms and boxplots...')

    boxplot_data = []
    for k in tqdm(clusters_idx, desc='Clusters'):
        if k >= 0: # this line is a placeholder for optionally inserting code for a specific cluster
            k_idxs =  mave.loc[mave['Cluster'] == k].index
            #if len(k_idxs) >= 50: #optional threshold to handle oversplitting
            boxplot_data.append(mave['DNN'][k_idxs])

    # calculate the average IQR
    iqr_values = [np.percentile(data, 75) - np.percentile(data, 25) for data in boxplot_data]
    average_iqr = np.mean(iqr_values)

    if sort_median is True: # sort boxplots by ascending median
        cluster_medians = [np.median(sublist) for sublist in boxplot_data] # calculate the median for each boxplot
        cluster_sorted_indices = sorted(range(len(cluster_medians)), key=lambda i: cluster_medians[i]) # get the indices of the boxplots sorted based on the median values
        boxplot_data = [boxplot_data[i] for i in cluster_sorted_indices]
        clusters_idx = [clusters_idx[i] for i in cluster_sorted_indices]
        sort_fig = 'predefined'
        # update 'all_clusters.csv' with new column containing sorted indices (for user reference only)
        mapping_dict = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(cluster_sorted_indices)}
        all_clusters_df['Cluster_sort'] = all_clusters_df['Cluster'].map(mapping_dict)
        all_clusters_df.to_csv(os.path.join(outputs_dir, 'all_clusters.csv'), index=False)
    else:
        cluster_sorted_indices = None

    # render boxplots showing DNN scores across all clusters
    #fig, ax = plt.subplots(figsize=(10,5))
    if len(boxplot_data) <= 100:
        box = plt.boxplot(boxplot_data[::-1], vert=False, showfliers=False, medianprops={'color': 'black'})
        plt.yticks([i for i in range(1, len(boxplot_data)+1)], [str(i) for i in range(len(boxplot_data))][::-1], fontsize=4)
    else:
        for pos, values in enumerate(boxplot_data):
            values = np.array(values)            
            median = np.median(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            plt.plot([q1, q3], [pos, pos], color='gray', lw=.5) # plot the IQR line
            plt.plot(median, pos, 'o', color='k', markersize=1, zorder=100) # plot the median point
        plt.yticks([])
        plt.gca().invert_yaxis()
    plt.ylabel('Clusters')
    plt.xlabel('DNN')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Average IQR: %.2f' % average_iqr)
    if ref_seq is not None:
        if sort_median is True:
            ref_cluster = cluster_sorted_indices.index(all_clusters_df.loc[ref_idx, 'Cluster'])
        elif sort_median is False and subset_clusters is False:
            ref_cluster = all_clusters_df.loc[ref_idx, 'Cluster']
        plt.axvline(np.median(boxplot_data[ref_cluster]), c='red', label='Ref', zorder=-100)
        plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'clusters_boxplot.png' % k), facecolor='w', dpi=dpi)
    plt.cla() 
    plt.clf()
    plt.close()

    if 1: # render plots used in manuscript figures
        # plot just the median values for clusters in a bar plot
        bar_height = 0.8#1.0 #0.6
        median_values = []
        for i in range(len(boxplot_data)):
            median_values.append(np.median(boxplot_data[i]))
        plt.figure(figsize=(1,5))
        y_positions = np.arange(len(median_values))
        if ref_seq is not None:
            for i in range(len(median_values)):
                if i == ref_cluster:
                    plt.barh(i, median_values[i], color='red', height=bar_height)
                else:
                    plt.barh(i, median_values[i], color='C0', height=bar_height)
        else:
            plt.barh(y_positions, median_values, height=bar_height)
        plt.gca().invert_yaxis()
        plt.axvline(x=0, color='black', linewidth=.5, zorder=100)
        #plt.xlim(np.amin(median_values),np.amax(median_values))

        plt.yticks(y_positions, y_positions)

        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'clusters_median_barplot.png'), facecolor='w', dpi=dpi)
        plt.cla()
        plt.clf()
        plt.close()

        # plot the occupancy of each cluster in a bar plot
        occupancy_values = []
        for i in range(len(boxplot_data)):
            occupancy_values.append(len(boxplot_data[i]))
        occupancy_values = occupancy_values[::-1]
        plt.figure(figsize=(1.5,5))
        if ref_seq is not None:
            for i in range(len(occupancy_values)):
                if i == ref_cluster:
                    plt.barh(i, occupancy_values[i], color='red', height=1.0)
                else:
                    plt.barh(i, occupancy_values[i], color='C0', height=1.0)
        else:
            plt.barh(y_positions, occupancy_values, color='C0', height=1.0)

        plt.gca().invert_yaxis()
        plt.axvline(x=0, color='black', linewidth=.5, zorder=100)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'clusters_occupancy_barplot.png'), facecolor='w', dpi=dpi)
        plt.cla()
        plt.clf()
        plt.close()


if 1:
    print('Computing sequence statistics...')
    regulatory_df = generate_regulatory_df(mave, clusters_labels, clusters_idx, ref_seq)

    regulatory_df.to_csv(os.path.join(outputs_dir, 'compare_clusters.csv'), index=False)

    if 1:
        regulatory_df_csv = pd.read_csv(os.path.join(outputs_dir, 'compare_clusters.csv'))
        # plot regulatory matrix using different metrics:
        fig = plt.figure(figsize=(20, 5))
        ax, cax, reordered_ind, revels = impress.plot_clusters_matches_2d_gui(regulatory_df_csv, fig, sort=sort_fig, column='Entropy', sort_index=cluster_sorted_indices)
        ax.set_title('')
        plt.savefig(os.path.join(outputs_dir, 'compare_clusters_entropy.png'), facecolor='w', dpi=dpi)
        plt.close()
        if ref_seq is not None:
            fig = plt.figure(figsize=(20, 5))
            ax, cax, reordered_ind, revels = impress.plot_clusters_matches_2d_gui(regulatory_df_csv, fig, sort=sort_fig, column='Reference', sort_index=cluster_sorted_indices)
            ax.set_title('')
            plt.savefig(os.path.join(outputs_dir, 'compare_clusters_reference.png'), facecolor='w', dpi=dpi)
            plt.close()


if 1: # create percent mismatch table
    if ref_seq is not None:
        print('Calculating percent mismatch table...')
        mismatch_df = impress.mismatch2csv(regulatory_df_csv, all_clusters_df, mave, ref_seq,
                                        threshold=90, # minimum percent mismatch at any position
                                        sort_index=cluster_sorted_indices,
                                        alphabet=alphabet, save_dir=outputs_dir)


if 1: # render logos for all clusters

    if seq_logo_pwm is True or seq_logo_enrich is True:
        if not os.path.exists(os.path.join(outputs_dir, 'clusters_seq')):
            os.mkdir(os.path.join(outputs_dir, 'clusters_seq'))
    if bias_removal is False:
        clusters_avg_fname = 'clusters_avg'
        if not os.path.exists(os.path.join(outputs_dir, clusters_avg_fname)):
            os.mkdir(os.path.join(outputs_dir, clusters_avg_fname))
    elif bias_removal is True:
        clusters_bias_fname = 'clusters_bias'
        if not os.path.exists(os.path.join(outputs_dir, clusters_bias_fname)):
            os.mkdir(os.path.join(outputs_dir, clusters_bias_fname))
        clusters_avg_fname = 'clusters_avg_nobias'
        if not os.path.exists(os.path.join(outputs_dir, clusters_avg_fname)):
            os.mkdir(os.path.join(outputs_dir, clusters_avg_fname))

    if plot_profiles is True:
        y_profiles = np.load(os.path.join(inputs_dir, 'y_profiles.npy'))
        if not os.path.exists(os.path.join(outputs_dir, 'clusters_profiles')):
            os.mkdir(os.path.join(outputs_dir, 'clusters_profiles'))

    if attr_logo_fixed is True:
        print('Finding global y_min/max of attribution maps...')
        y_min, y_max = 0., 0.
        for k in tqdm(clusters_idx, desc='Clusters'):
            k_idxs =  mave.loc[mave['Cluster'] == k].index
            map_avg = np.mean(maps[k_idxs], axis=0)
            map_avg -= np.expand_dims(np.mean(map_avg, axis=1), axis=1) # position-dependent centering (gauge fix)

            positive_mask = map_avg > 0
            positive_matrix = map_avg * positive_mask
            positive_summations = positive_matrix.sum(axis=1)
            positive_max_position = np.argmax(positive_summations)
            if positive_summations[positive_max_position] > y_max:
                y_max = positive_summations[positive_max_position]

            negative_mask = map_avg < 0
            negative_matrix = map_avg * negative_mask
            negative_summations = negative_matrix.sum(axis=1)
            negative_max_position = np.argmin(negative_summations)
            if negative_summations[negative_max_position] < y_min:
                y_min = negative_summations[negative_max_position]

    if bias_removal is True:
        null_rate = 1 - mut_rate
        background_entropy = entropy(np.array([null_rate, (1-null_rate)/3, (1-null_rate)/3,(1-null_rate)/3]), base=2) #e.g., 0.096 for r=0.01
        entropy_threshold = background_entropy*0.5
        cluster_bias = np.zeros(shape=(len(clusters_idx), maps.shape[1], maps.shape[2]))
        k_name = 0 # necessary for keeping logo filenames in correct order if using 'sort_median' above
        for k in tqdm(clusters_idx, desc='Bias'):
            k_idxs = mave.loc[mave['Cluster'] == k].index
            child_maps = maps[k_idxs]
            entropic_positions = regulatory_df[(regulatory_df['Cluster'] == k) & 
                                            (regulatory_df['Entropy'] > entropy_threshold)]['Position'].values
            if len(entropic_positions) == 0:
                continue
            for ep in entropic_positions:
                for child in child_maps:
                    cluster_bias[k,ep,:] += child[ep,:]
            cluster_bias[k] /= len(child_maps)

            if plot_bias is True:
                try:
                    cluster_bias_map = squid_utils.arr2pd(cluster_bias[k], alphabet)
                except:
                    cluster_bias_map = squid_utils.arr2pd(cluster_bias[k]) # ZULU: may need a separate option for this to deal with two-hot encodings (e.g., CLIPNET)
                if attr_logo_fixed is True:
                    y_min_max = [y_min, y_max]
                else:
                    y_min_max = None
                logo_fig = impress.plot_logo(cluster_bias_map, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl, figsize=figsize, y_min_max=y_min_max)
                plt.savefig(os.path.join(outputs_dir, 'clusters_bias/cluster_%s.png' % k_name), facecolor='w', dpi=dpi)
                plt.close()
            k_name += 1
        all_bias = np.mean(cluster_bias, axis=0)
        np.save(os.path.join(outputs_dir, 'bias_matrix.npy'), all_bias)


    if attr_logo_fixed is True: # render individual reference map using global y_min/max
        ref_map = np.copy(maps[ref_idx])
        if bias_removal is True:
            ref_map -= all_bias
        try:
            ref_map = squid_utils.arr2pd(ref_map, alphabet)
        except:
            ref_map = squid_utils.arr2pd(ref_map) # ZULU: may need a separate option for this to deal with two-hot encodings (e.g., CLIPNET)
        logo_fig = impress.plot_logo(ref_map, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl, y_min_max=[y_min, y_max], figsize=figsize)
        if bias_removal is False and ref_seq is not None:
            plt.savefig(os.path.join(outputs_dir, 'individual_logo_reference_y_fixed.png'), facecolor='w', dpi=dpi)
        elif bias_removal is True and ref_seq is not None:
            plt.savefig(os.path.join(outputs_dir, 'individual_logo_reference_y_fixed_nobias.png'), facecolor='w', dpi=dpi)
        plt.close()
        if 1: # render the map with the highest prediction score using global y_min/max (useful for optimization analysis)
            max_idx = mave['DNN'].argmax()
            print('Max idx:', max_idx)
            max_map = squid_utils.arr2pd(maps[max_idx], alphabet)
            logo_fig = impress.plot_logo(max_map, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl, y_min_max=[y_min, y_max], figsize=figsize)
            plt.savefig(os.path.join(outputs_dir, 'individual_logo_maxpred.png'), facecolor='w', dpi=dpi)
            plt.close()
        if 1: # render the average of all maps
            try:
                all_maps_avg = squid_utils.arr2pd(np.mean(maps, axis=0), alphabet)
            except: #ZULU
                all_maps_avg = squid_utils.arr2pd(np.mean(maps, axis=0))
            logo_fig = impress.plot_logo(all_maps_avg, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl, y_min_max=[y_min, y_max], figsize=figsize)
            plt.savefig(os.path.join(outputs_dir, 'all_maps_average_y_fixed.png'), facecolor='w', dpi=dpi)
            plt.close()


    if attr_logo_adaptive is True: # render individual reference map using adaptive y-axis
        ref_map = np.copy(maps[ref_idx])
        if bias_removal is True:
            ref_map -= all_bias
        try:
            ref_map = squid_utils.arr2pd(ref_map, alphabet)
        except:
            ref_map = squid_utils.arr2pd(ref_map) # ZULU: may need a separate option for this to deal with two-hot encodings (e.g., CLIPNET)
        logo_fig = impress.plot_logo(ref_map, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl, figsize=figsize)
        if bias_removal is False and ref_seq is not None:
            plt.savefig(os.path.join(outputs_dir, 'individual_logo_reference_y_adaptive.png'), facecolor='w', dpi=dpi)
        elif bias_removal is True and ref_seq is not None:
            plt.savefig(os.path.join(outputs_dir, 'individual_logo_reference_y_adaptive_nobias.png'), facecolor='w', dpi=dpi)
        plt.close()

        if 1: # render the average of all maps
            try:
                all_maps_avg = squid_utils.arr2pd(np.mean(maps, axis=0), alphabet)
            except: #ZULU
                all_maps_avg = squid_utils.arr2pd(np.mean(maps, axis=0))
            logo_fig = impress.plot_logo(all_maps_avg, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl, figsize=figsize)
            plt.savefig(os.path.join(outputs_dir, 'all_maps_average_y_adaptive.png'), facecolor='w', dpi=dpi)
            plt.close()

    if bias_removal is True:
        try:
            bias_map = squid_utils.arr2pd(all_bias, alphabet)
        except:
            bias_map = squid_utils.arr2pd(all_bias) # ZULU: may need a separate option for this to deal with two-hot encodings (e.g., CLIPNET)
        if attr_logo_adaptive is True:
            logo_fig = impress.plot_logo(bias_map, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl, figsize=figsize)
            plt.savefig(os.path.join(outputs_dir, 'bias_y_adaptive.png'), facecolor='w', dpi=dpi)
            plt.close()
        if attr_logo_fixed is True:
            logo_fig = impress.plot_logo(bias_map, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl,  y_min_max=[y_min, y_max], figsize=figsize)
            plt.savefig(os.path.join(outputs_dir, 'bias_y_fixed.png'), facecolor='w', dpi=dpi)
            plt.close()

    print('Rendering logos...')

    consensus_df = pd.DataFrame(columns = ['Cluster','Median','Mean','Centroid','Consensus'], index=range(len(clusters_idx))) # records useful information for later review
    clusters_avg_matrices = np.zeros(shape=(len(clusters_idx), maps.shape[1], maps.shape[2]))
    k_name = 0 # necessary for keeping logo filenames in correct order if using 'sort_median' above
    for k in tqdm(clusters_idx, desc='Clusters'):
        if k >= 0: # this line is a placeholder for optionally inserting code for a specific cluster
            k_idxs = mave.loc[mave['Cluster'] == k].index
            seqs_k = mave['Sequence'][k_idxs]

            # save consensus sequence of each cluster to dataframe
            seq_array_cluster = motifs.create(seqs_k, alphabet=alphabet)
            pfm_cluster = seq_array_cluster.counts # position frequency matrix
            consensus_df.at[k_name, 'Cluster'] = k_name
            consensus_seq = pfm_cluster.consensus
            consensus_df.at[k_name, 'Consensus'] = consensus_seq
            # save the centroid sequence of each cluster to dataframe
            hamming_distances = [utilities.hamming_distance(consensus_seq, seq) for seq in seqs_k]
            centroid_index = np.argmin(hamming_distances)
            centroid_seq = seqs_k.iloc[centroid_index]
            consensus_df.at[k_name, 'Centroid'] = centroid_seq
            try:
                consensus_df.at[k_name, 'Median'] = np.median(boxplot_data[k_name])
                consensus_df.at[k_name, 'Mean'] = np.mean(boxplot_data[k_name])
            except:
                pass
        
            if plot_profiles is True:
                fig, ax = plt.subplots(figsize=(20,1.2))#.5))
                for i in k_idxs:
                    ax.plot(y_profiles[i,0],
                        color='C0', linewidth=.5, zorder=0, alpha=0.1)
                    ax.plot(-1.*y_profiles[i,1],
                            color='C3', linewidth=.5, zorder=-10, alpha=0.1)
                #if profiles_ylim is not None:
                #    ax.set_ylim(profiles_ylim)
                plt.tight_layout()
                plt.xlim(250, 750)
                plt.savefig(os.path.join(outputs_dir, 'clusters_profiles/profiles_%s.png' % k_name), dpi=200)
                plt.close()

            if seq_logo_pwm is True or seq_logo_enrich is True:
                seq_array_cluster = motifs.create(seqs_k, alphabet=alphabet)
                pfm_cluster = seq_array_cluster.counts # position frequency matrix
                pseudocounts = 0.5
                pwm_cluster = pfm_cluster.normalize(pseudocounts=pseudocounts) #https://biopython-tutorial.readthedocs.io/en/latest/notebooks/14%20-%20Sequence%20motif%20analysis%20using%20Bio.motifs.html
                if seq_logo_pwm is True:
                    if 0: # standard PWM
                        seq_logo = pd.DataFrame(pwm_cluster)
                    else: # compute Position Probability Matrix (PPM) and compute information content
                        ppm_df = pd.DataFrame(pwm_cluster)#, index=list(alphabet))
                        background = np.array([1.0 / len(alphabet)] * len(alphabet))  # assuming uniform background
                        ppm_df += 1e-6  # to avoid log(0), add a small value
                        info_content = np.sum(ppm_df * np.log2(ppm_df / background), axis=1)# + np.log2(4)
                        seq_logo = ppm_df.multiply(info_content, axis=0)

                if 1:
                    logo_fig, axis = impress.plot_logo(seq_logo, logo_type='sequence', ref_seq=None, aesthetic_lvl=aesthetic_lvl, center_values=False, figsize=figsize)
                    logo_dir = os.path.join(outputs_dir, 'clusters_seq/pwm')
                    if not os.path.exists(logo_dir):
                        os.mkdir(logo_dir)
                    plt.savefig(os.path.join(logo_dir, 'cluster_%s.png' % k_name), facecolor='w', dpi=dpi)
                    plt.cla()
                    plt.clf()
                    plt.close()

                if seq_logo_enrich is True: # enrichment logo (see https://logomaker.readthedocs.io/en/latest/examples.html#ars-enrichment-logo)
                    enrichment_ratio = (pd.DataFrame(pfm_cluster) + pseudocounts) / (pd.DataFrame(pfm_background) + pseudocounts)
                    seq_logo = np.log2(enrichment_ratio)
                    logo_fig, axis = impress.plot_logo(seq_logo, logo_type='sequence', ref_seq=ref_seq, aesthetic_lvl=aesthetic_lvl, figsize=figsize)
                    logo_dir = os.path.join(outputs_dir, 'clusters_seq/enrich')
                    if not os.path.exists(logo_dir):
                        os.mkdir(logo_dir)
                    plt.savefig(os.path.join(logo_dir, 'cluster_%s.png' % k_name), facecolor='w', dpi=dpi)
                    plt.cla() 
                    plt.clf()
                    plt.close()

            if attr_logo_fixed is True or attr_logo_adaptive is True:
                maps_avg = np.mean(maps[k_idxs], axis=0)
                if bias_removal is True:
                    maps_avg -= all_bias

                clusters_avg_matrices[k_name] = maps_avg

                try:
                    maps_avg = squid_utils.arr2pd(maps_avg, alphabet)
                except:
                    maps_avg = squid_utils.arr2pd(maps_avg) # ZULU: may need a separate option for this to deal with two-hot encodings (e.g., CLIPNET)                
                if attr_logo_adaptive is True:
                    logo_fig, ax = impress.plot_logo(maps_avg, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl, figsize=figsize)
                    logo_dir = os.path.join(outputs_dir, '%s/y_adaptive' % clusters_avg_fname)
                    if not os.path.exists(logo_dir):
                        os.mkdir(logo_dir)
                    plt.savefig(os.path.join(logo_dir, 'cluster_%s.png' % k_name), facecolor='w', dpi=dpi)
                    plt.cla()
                    plt.clf()
                    plt.close()
                if attr_logo_fixed is True:
                    if 1:
                        logo_fig, axis = impress.plot_logo(maps_avg, logo_type='attribution', ref_seq=None, aesthetic_lvl=aesthetic_lvl, y_min_max=[y_min, y_max], figsize=figsize)
                    else: # color attribution logo using wild-type sequence
                        logo_fig, axis = impress.plot_logo(maps_avg, logo_type='sequence', ref_seq=ref_seq, aesthetic_lvl=aesthetic_lvl, y_min_max=[y_min, y_max], figsize=figsize)
                    logo_dir = os.path.join(outputs_dir, '%s/y_fixed' % clusters_avg_fname)
                    if not os.path.exists(logo_dir):
                        os.mkdir(logo_dir)
                    plt.savefig(os.path.join(logo_dir, 'cluster_%s.png' % k_name), facecolor='w', dpi=dpi)
                    plt.cla() 
                    plt.clf()
                    plt.close()

        k_name += 1
        if k%10==0:
            plt.close('all')
            gc.collect()
    plt.close()
    consensus_df.to_csv(os.path.join(outputs_dir, 'clusters_consensus.csv'), index=False)
    np.save(os.path.join(outputs_dir, '%s/clusters_avg_matrices.npy' % clusters_avg_fname), clusters_avg_matrices)

    if 1:
        if attr_logo_fixed is True:
            print('Rendering variability logo...')
            from PIL import Image

            def darken_blend(image1, image2):
                '''
                Blend two images using the 'Darken' blend mode. The 'Darken' blend mode
                compares each pixel of two images and takes the darker value of each pixel.
                '''
                arr1 = np.array(image1)
                arr2 = np.array(image2)
                darkened_array = np.minimum(arr1, arr2)
                return Image.fromarray(darkened_array)

            # overlay all PNG images in the specified folder using 'Darken' mode.
            images = []
            logo_dir = os.path.join(outputs_dir, '%s/y_fixed' % clusters_avg_fname)
            for file_name in os.listdir(logo_dir): # load all PNG images from the folder
                if file_name.endswith('.png'):
                    image_path = os.path.join(logo_dir, file_name)
                    images.append(Image.open(image_path))

            if images:
                result_image = images[0] # initialize the result with the first image
                for image in tqdm(images[1:], desc='Image'): # apply the darken blend mode to all images
                    result_image = darken_blend(result_image, image)

                if bias_removal is False:
                    result_image.save(os.path.join(outputs_dir, 'variability_logo.png'))
                elif bias_removal is True:
                    result_image.save(os.path.join(outputs_dir, 'variability_logo_nobias.png'))
                print('Variability logo saved.')
            else:
                print('No PNG images found in the specified folder.')