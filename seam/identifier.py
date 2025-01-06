import os, sys
import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial import distance
from scipy.cluster import hierarchy
import itertools
from scipy.stats import entropy

py_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(py_dir)


if 1:
    #dir_name = 'examples/examples_clipnet/outputs_local_PIK3R3/heterozygous_pt10/quantity_nfolds9'
    #dir_name = 'examples/examples_deepstarr/outputs_local_AP1-m1-seq22612'
    #dir_name = 'examples/examples_deepstarr/outputs_local_Ohler1-m0-seq20647' # cut=0.3
    dir_name = 'examples/examples_deepstarr/outputs_local_AP1-m0-seq13748/archives/AP1-rank0_origVersion'
    #dir_name = 'examples/examples_deepstarr/outputs_local_Ohler1-m0-seq22627'
    #dir_name = 'examples/examples_deepstarr/outputs_local_Ohler6-m0-seq171' # cut=0.06
    #dir_name = 'examples/examples_deepstarr/outputs_local_DRE-m0-seq1134'
    #dir_name = 'examples/examples_deepstarr/outputs_local_DRE-m2-seq22619'
    #dir_name = 'examples/examples_deepstarr/outputs_local_DRE-m2-seq24760'
    #dir_name = 'examples/examples_deepstarr/outputs_local_AP1-m1-seq21069'



# Import Mechanism Summary Matrix (MSM) generated previously in meta_explainer.py
#df = pd.read_csv(os.path.join(parent_dir, dir_name + '/clusters_deepshap_hierarchical_maxclust30_sortMedian/compare_clusters.csv'))
df = pd.read_csv(os.path.join(parent_dir, dir_name + '/clusters_deepshap_umap_kmeans200_crop_110_137_sortMedian/compare_clusters.csv'))

#df = pd.read_csv(os.path.join(parent_dir, dir_name + '/clusters_hierarchical_distance10/compare_clusters.csv'))
column = 'Entropy'

nC = df['Cluster'].max() + 1
nP = df['Position'].max() + 1
revels = df.pivot(columns='Position', index='Cluster', values=column)

cov_matrix = revels.cov()

def plot_with_updated_toolbar(matrix, original_indices, title="Covariance Matrix"):
    """
    Plots a heatmap of the given reordered matrix and updates the toolbar to display
    original x, y positions and covariance values.

    Parameters:
        matrix (pd.DataFrame): The reordered covariance matrix to plot.
        original_indices (list): The original row/column indices corresponding to the reordered matrix.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, cmap='seismic', center=0, cbar_kws={'label': 'Covariance'}, ax=ax)

    def format_coord(x, y):
        try:
            # Convert x, y to integer indices
            col, row = int(x), int(y)
            # Ensure indices are within bounds
            if 0 <= col < matrix.shape[1] and 0 <= row < matrix.shape[0]:
                # Map reordered indices back to original indices
                original_x = original_indices[col]
                original_y = original_indices[row]
                value = matrix.iloc[row, col]  # Get the covariance value
                return f"x={original_x}, y={original_y}, z={value:.2f}"
            else:
                return f"x={x:.2f}, y={y:.2f}"  # Default for out-of-bounds
        except (ValueError, IndexError):
            return f"x={x:.2f}, y={y:.2f}"  # Default for invalid input

    ax.format_coord = format_coord
    plt.title(title)

if 1:
    plot_with_updated_toolbar(cov_matrix, cov_matrix.index.tolist(), title="Original Covariance Matrix")
    plt.show

# Identify and remove the largest cluster
def remove_largest_cluster(tfbs_clusters):
    # Find the cluster with the maximum number of positions
    largest_cluster = max(tfbs_clusters, key=lambda k: len(tfbs_clusters[k]))
    
    # Remove the largest cluster; i.e., the cluster for all the positions with near zero covariance
    #print(f"Removing largest cluster (all 'white' positions): {largest_cluster}")
    filtered_tfbs_clusters = {k: v for k, v in tfbs_clusters.items() if k != largest_cluster}

    return filtered_tfbs_clusters

if 1: # resort covariance matrix
    cluster_method = 'average'
    # Compute the linkage for rows
    row_linkage = hierarchy.linkage(distance.pdist(cov_matrix), method=cluster_method)
    row_dendrogram = hierarchy.dendrogram(row_linkage, no_plot=True, color_threshold=-np.inf)
    row_order = row_dendrogram['leaves'] # Reordered row indices

    # Compute the linkage for columns
    col_linkage = hierarchy.linkage(distance.pdist(cov_matrix.T), method=cluster_method)
    col_dendrogram = hierarchy.dendrogram(col_linkage, no_plot=True, color_threshold=-np.inf)
    col_order = col_dendrogram['leaves'] # Reordered column indices

    # Reorder the covariance matrix
    reordered_cov_matrix = cov_matrix.iloc[row_order, :].iloc[:, col_order]

    if 0:
        plot_with_updated_toolbar(reordered_cov_matrix, row_order, title="Reordered Covariance Matrix with Original Indices")
        plt.show()

    if 0: # Find the largest gap between successive distances
        distances = row_linkage[:, 2]
        largest_gap = np.argmax(np.diff(distances))
        optimal_cut = distances[largest_gap]  # The distance threshold for cutting
        print(f"Largest gap found at distance: {optimal_cut}")

    elif 0: # Find the n-th largest gap between successive distances (should ignore the largest cluster with all near-zero covariance)
        def find_nth_largest_gap(row_linkage, nth_gap=2):
            """
            Finds the nth largest gap in the hierarchical clustering linkage distances and returns the optimal cut.
            
            Parameters:
                row_linkage: ndarray
                    The linkage matrix from hierarchical clustering.
                nth_gap: int
                    Specifies which largest gap to use (e.g., 2 for second-largest, 3 for third-largest, etc.).
            
            Returns:
                optimal_cut: float
                    The distance threshold for cutting the dendrogram based on the nth largest gap.
            """
            distances = row_linkage[:, 2]  # Get distances from linkage
            gaps = np.diff(distances)  # Compute gaps between successive distances

            # Find indices of sorted gaps in descending order
            sorted_gap_indices = np.argsort(gaps)[::-1]

            # Debug: Print the gaps and their corresponding indices
            print("Gaps (sorted by size):", gaps[sorted_gap_indices])
            print("Gap indices (sorted by size):", sorted_gap_indices)

            # Ensure nth_gap is within valid range
            if nth_gap > len(sorted_gap_indices):
                raise ValueError(f"nth_gap ({nth_gap}) exceeds the number of gaps ({len(sorted_gap_indices)}).")

            # Select the nth largest gap
            nth_largest_gap_idx = sorted_gap_indices[nth_gap - 1]  # Subtract 1 because nth_gap is 1-based index

            # Map to the corresponding distance threshold
            optimal_cut = distances[nth_largest_gap_idx + 1]  # Add 1 because 'gaps' corresponds to distances[i+1] - distances[i]

            # Debug: Print information about the nth largest gap
            #print(f"{nth_gap}th largest gap value: {gaps[nth_largest_gap_idx]}")
            #print(f"{nth_gap}th largest gap corresponds to distances[{nth_largest_gap_idx + 1}] = {optimal_cut}")

            return optimal_cut

        # Example usage
        nth_gap = 2  # Choose the nth largest gap (e.g., 2 for second-largest)
        optimal_cut = find_nth_largest_gap(row_linkage, nth_gap=nth_gap)
        #print(f"Optimal cut determined by the {nth_gap}th largest gap: {optimal_cut}")

    else:
        optimal_cut = .3#.06

    # Generate cluster labels based on the largest gap
    cluster_labels = hierarchy.fcluster(row_linkage, t=optimal_cut, criterion='distance')

    # Map clusters to original positions
    tfbs_clusters = {f"TFBS {label}": [] for label in set(cluster_labels)}
    for idx, cluster in enumerate(cluster_labels):
        original_position = cov_matrix.index[row_order[idx]]  # Map reordered index back to original position
        tfbs_clusters[f"TFBS {cluster}"].append(original_position)

    # Apply the function to remove the largest cluster
    tfbs_clusters = remove_largest_cluster(tfbs_clusters)

    # Draw rectangles for clusters using reordered indices
    plot_with_updated_toolbar(cov_matrix, cov_matrix.index.tolist(), title="Covariance Matrix with TFBS Clusters")

    for cluster, positions in tfbs_clusters.items():
        # Map original positions back to reordered indices
        reordered_positions = [row_order.index(cov_matrix.index.get_loc(pos)) for pos in positions]
        start = min(reordered_positions)
        end = max(reordered_positions)
        rect = patches.Rectangle((start, start), end - start + 1, end - start + 1,  # Add 1 to include the full extent
                                linewidth=1, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)

    plt.show()

    # Print clusters for inspection
    for cluster, positions in tfbs_clusters.items():
        reordered_positions = [row_order.index(cov_matrix.index.get_loc(pos)) for pos in positions]
        print(f"{cluster}: {reordered_positions}")

    # Plot the dendrogram
    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(
                        row_linkage,
                        labels=reordered_cov_matrix.index,  # Use original indices for better interpretability
                        leaf_rotation=90,  # Rotate labels for readability
                        leaf_font_size=10,  # Adjust font size for readability
                        )
    plt.axhline(optimal_cut, c='red')
    plt.title("Dendrogram for TFBS Clustering")
    plt.xlabel("Position")
    plt.ylabel("Distance")
    plt.show()


if 0: # manually merge oversplit TFBS clusters
    reassignment = [[1,11,12,13], [2,3,4,19,20], [5,6,7,8,15], [21,22], [9,10,16,17]]  # e.g., reassignment = [[1,2]] to merge TFBS 1 and TFBS 2
else:
    reassignment = []

def merge_tfbs_clusters(tfbs_clusters, reassignment):
    # Create a new dictionary for merged clusters
    merged_clusters = {}

    # Track which clusters have already been merged
    merged_keys = set(itertools.chain.from_iterable(reassignment))

    # Merge specified clusters
    for i, merge_group in enumerate(reassignment, start=1):
        # Combine all positions from the specified clusters
        merged_positions = []
        for idx in merge_group:
            cluster_name = f"TFBS {idx}"
            if cluster_name in tfbs_clusters:
                merged_positions.extend(tfbs_clusters.pop(cluster_name))
        # Create a new merged cluster
        merged_clusters[f"TFBS {i}"] = sorted(merged_positions)

    # Add remaining clusters with unique names
    next_cluster_id = len(merged_clusters) + 1
    remaining_clusters = {
        f"TFBS {next_cluster_id + i}": positions
        for i, (key, positions) in enumerate(tfbs_clusters.items())
    }

    # Combine merged and remaining clusters
    final_clusters = {**merged_clusters, **remaining_clusters}

    return final_clusters


# Remap tfbs_clusters using the provided code snippet
remapped_tfbs_clusters = {}
for cluster, positions in tfbs_clusters.items():
    reordered_positions = [
        row_order.index(cov_matrix.index.get_loc(pos)) for pos in positions
    ]
    remapped_tfbs_clusters[cluster] = reordered_positions

# Perform the merging
tfbs_clusters = merge_tfbs_clusters(remapped_tfbs_clusters, reassignment)

# Print the updated clusters
print("\nUpdated Clusters:")
for cluster, positions in tfbs_clusters.items():
    print(f"{cluster}: {positions}")


if 1:
    # Define a near-zero entropy threshold
    mut_rate = 0.10 # Needs to match the mutation rate used to generate the sequence libary
    null_rate = 1 - mut_rate
    background_entropy = entropy(np.array([null_rate, (1-null_rate)/3, (1-null_rate)/3,(1-null_rate)/3]), base=2) #e.g., 0.096 for r=0.01
    entropy_threshold = background_entropy*0.5

    if 1:  # Reorder dataframe based on visual patterns
        row_linkage = hierarchy.linkage(distance.pdist(revels), method='ward')
        dendrogram = hierarchy.dendrogram(row_linkage, no_plot=True, color_threshold=-np.inf)
        reordered_ind = dendrogram['leaves']
        revels = revels.reindex(reordered_ind)
    else:
        reordered_ind = revels.index # i.e., the original indices

    # Plot the entropy matrix
    fig, ax = plt.subplots(figsize=(20, 5))

    palette = sns.color_palette('rocket', n_colors=100)
    palette.reverse()
    heatmap = ax.pcolormesh(revels, cmap=mpl.colors.ListedColormap(list(palette)), vmin=0, vmax=2)

    # Add rectangles based on largest contiguous low-entropy regions
    for cluster, positions in tfbs_clusters.items():
        if isinstance(positions[0], int):  # Already remapped
            reordered_positions = positions
        else:  # Needs remapping
            reordered_positions = [row_order.index(cov_matrix.index.get_loc(pos)) for pos in positions]
        if reordered_positions:  # Ensure the cluster has valid positions
            start = min(reordered_positions)
            end = max(reordered_positions)

            # Identify rows with near-zero entropy in the reordered dataframe
            low_entropy_rows = revels.iloc[:, start:end + 1].lt(entropy_threshold).any(axis=1)
            low_entropy_row_indices = low_entropy_rows[low_entropy_rows].index.tolist()

            # Need to map back to the correct ordering if revels was reordered above
            reverse_reordered_ind = {original: reordered for reordered, original in enumerate(reordered_ind)}
            low_entropy_row_indices = [reverse_reordered_ind[row] for row in low_entropy_row_indices]

            # Group reordered rows into contiguous blocks
            for _, block in itertools.groupby(enumerate(low_entropy_row_indices), lambda x: x[0] - x[1]):
                block_rows = list(map(lambda x: x[1], block))
                if block_rows:  # If a valid block is found
                    rect_start = min(block_rows)  # Topmost row
                    rect_end = max(block_rows)  # Bottommost row
                    rect_height = rect_end - rect_start + 1  # Height of the rectangle

                    # Draw rectangle
                    rect = patches.Rectangle((start, rect_start), end - start + 1, rect_height,
                                            linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(heatmap, cax=cax, orientation='vertical')#, format='%.0f')
    heatmap.set_clim(0, 2)

    cbar.ax.set_ylabel("Shannon entropy (bits)", rotation=270, fontsize=6, labelpad=9)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xlabel('Position', fontsize=6)
    ax.set_ylabel('Cluster', fontsize=6)
    ax.invert_yaxis()
    plt.show()

if 1:
    # Sort TFBS clusters by their leftmost position
    sorted_tfbs_clusters = {
        k: v for k, v in sorted(tfbs_clusters.items(), key=lambda item: min(item[1]))
    }

    # Initialize binary matrix
    states = revels.index.tolist()
    tfbs_names = list(sorted_tfbs_clusters.keys())
    binary_matrix = pd.DataFrame(0, index=states, columns=tfbs_names)

    # Populate the binary matrix based on rectangle positions
    for state in states:
        for tfbs, positions in sorted_tfbs_clusters.items():
            # If any position in the TFBS is low entropy in the state, mark as present
            state_entropies = revels.loc[state, positions]
            if (state_entropies < entropy_threshold).any():  # Check for any overlap
                binary_matrix.at[state, tfbs] = 1

    # Create a heatmap for the binary matrix
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    # Use a discrete colormap: white for 0 (absent) and black for 1 (present)
    sns.heatmap(binary_matrix, cmap="binary", linewidths=0.25, linecolor='gray',
                cbar=False, square=True)  # Disable color bar for binary visualization

    # Add labels and title
    plt.xlabel("TFBS", fontsize=12)
    plt.ylabel("State", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Add a black border around the entire matrix
    ax = plt.gca()  # Get current axis
    n_states, n_tfbs = binary_matrix.shape  # Get dimensions of the matrix
    rect = patches.Rectangle((0, 0), n_tfbs, n_states, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Add a custom legend
    present_patch = patches.Patch(color='black', label='Present')
    absent_patch = patches.Patch(facecolor='white', edgecolor='black', label='Absent')
    plt.legend(handles=[present_patch, absent_patch], bbox_to_anchor=(1.05, 1), loc='upper left', title="Legend")

    # Show the plot
    plt.tight_layout()
    plt.show()


