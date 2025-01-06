import os
import pandas as pd
import matplotlib.pyplot as plt

py_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(py_dir)

dir_name = os.path.join(parent_dir, 'examples/outputs_local_chrombpnet_PPIF_promoter/mut_pt10/seed2_N100k_allFolds')

compare_df = pd.read_csv(os.path.join(dir_name, 'clusters_kmeans/compare_clusters.csv'))
#compare_df = pd.read_csv(os.path.join(dir_name, 'clusters_hierarchical/maxd1pt5/compare_clusters.csv'))

#compare_df_csv = compare_df_csv.dropna()
#compare_df_csv['Cluster'] = compare_df_csv['Cluster'] - 1

import impress
fig = plt.figure(figsize=(20, 5))

impress.plot_clusters_matches_2d_gui(compare_df, fig, sort=True, column='Entropy', delta=True)
plt.show()