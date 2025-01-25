# Standard libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Bioinformatics libraries
from Bio import motifs  # pip install biopython
import logomaker  # pip install logomaker
import squid.utils as squid_utils  # pip install squid-nn
import seaborn as sns  # pip install seaborn
from scipy.stats import entropy


class MetaExplainer:
    """
    MetaExplainer: A class for analyzing and visualizing attribution map clusters
    
    This class builds on the Clusterer class to provide detailed analysis and 
    visualization of attribution map clusters, including:
    
    Analysis Features:
    - Mechanism Summary Matrix (MSM) generation
    - Sequence logos and attribution logos
    - Cluster membership tracking
    - Bias removal and normalization
    
    Visualization Features:
    - DNN score distributions per cluster
    - Sequence logos (PWM and enrichment)
    - Attribution logos (fixed and adaptive scaling)
    - Mechanism Summary Matrices
    - Cluster profile plots
    
    Requirements:
    - All requirements from Clusterer class
    - Biopython
    - Logomaker
    - Seaborn
    - SQUID-NN
    """
    
    def __init__(self, clusterer, mave_df, ref_idx=0,
                bias_removal=False, mut_rate=0.10, aesthetic_lvl=1):
        """Initialize MetaExplainer with a Clusterer instance and data."""
        # Store inputs
        self.clusterer = clusterer  # clusterer.cluster_labels should contain labels_n
        self.mave = mave_df
        self.ref_idx = ref_idx
        self.bias_removal = bias_removal
        self.mut_rate = mut_rate
        self.aesthetic_lvl = aesthetic_lvl
        
        # Initialize other attributes
        self.alphabet = None
        self.msm = None
        self.cluster_bias = None
        self.consensus_df = None
        self.membership_df = None
        
        # Validate and process inputs
        self._validate_inputs()
        self._process_inputs()

    def _validate_inputs(self):
        """Validate input data and parameters."""
        # Ensure mave_df has required columns
        required_cols = {'Sequence', 'DNN'}
        if not required_cols.issubset(self.mave.columns):
            raise ValueError(f"mave_df must contain columns: {required_cols}")
        
        # Validate cluster labels exist
        if not hasattr(self.clusterer, 'cluster_labels') or self.clusterer.cluster_labels is None:
            raise ValueError("Clusterer must have valid cluster_labels. Did you run clustering?")
        
        # Get reference sequence from index
        self.ref_seq = self.mave['Sequence'].iloc[self.ref_idx]
                
        # Determine alphabet from sequences
        self.alphabet = sorted(list(set(self.mave['Sequence'][0:100].apply(list).sum())))
        
    def _process_inputs(self):
        """Process inputs and initialize derived data structures."""
        # Create membership tracking DataFrame
        self.membership_df = pd.DataFrame({
            'Cluster': self.clusterer.cluster_labels,
            'Original_Index': range(len(self.mave))
        })
        
        # Add cluster assignments to mave DataFrame
        self.mave = self.mave.copy()  # Create a copy to avoid modifying original
        self.mave['Cluster'] = self.clusterer.cluster_labels
        
        # Initialize cluster indices from unique cluster labels
        self.cluster_indices = np.unique(self.clusterer.cluster_labels)
        
    def get_cluster_order(self, sort_method='median', sort_indices=None):
        """Get cluster ordering based on specified method."""
        if sort_method is None:
            return self.cluster_indices  # Return actual indices instead of range
                
        if sort_method == 'predefined' and sort_indices is not None:
            return np.array(sort_indices)
                
        if sort_method == 'hierarchical':
            if not hasattr(self, 'msm') or self.msm is None:
                raise ValueError("MSM required for hierarchical sorting. Call generate_msm() first.")
            from scipy.cluster import hierarchy
            from scipy.spatial import distance
            matrix_data = self.msm.pivot(columns='Position', index='Cluster', values='Entropy')
            linkage = hierarchy.linkage(distance.pdist(matrix_data), method='ward')
            dendro = hierarchy.dendrogram(linkage, no_plot=True, color_threshold=-np.inf)
            return self.cluster_indices[dendro['leaves']]  # Map back to actual indices
                
        if sort_method == 'median':
            # Calculate median DNN score for each cluster
            cluster_medians = []
            for k in self.cluster_indices:
                k_idxs = self.mave.loc[self.mave['Cluster'] == k].index
                cluster_medians.append(self.mave.loc[k_idxs, 'DNN'].median())
            
            # Sort clusters by median DNN score
            sorted_order = np.argsort(cluster_medians)
            return self.cluster_indices[sorted_order]  # Map back to actual indices
        
        raise ValueError(f"Unknown sort_method: {sort_method}")
    
    def plot_cluster_stats(self, plot_type='box', metric='prediction', save_path=None, 
                        sort_by_median=True, show_ref=True, show_fliers=False, fontsize=8, dpi=200):
        """Plot cluster statistics with various visualization options.
        
        Parameters
        ----------
        plot_type : {'box', 'bar'}
            Type of visualization:
            - 'box': Show distribution as box plots (predictions only)
            - 'bar': Show bar plot of predictions or counts
        metric : {'prediction', 'counts'}
            What to visualize (only used for bar plots):
            - 'prediction': DNN prediction scores
            - 'counts': cluster occupancy/size
        save_path : str, optional
            Path to save figure. If None, display instead
        sort_by_median : bool
            If True, sort clusters by median prediction values
        show_ref : bool
            If True and reference sequence exists, highlight its cluster
        show_fliers : bool
            If True and plot_type='box', show outlier points
        fontsize : int
            Font size for tick labels
        dpi : int
            DPI for saved figure
        """        
        # Collect data for each cluster
        boxplot_data = []
        
        # Use actual clusters from data instead of cluster_indices
        actual_clusters = np.sort(self.mave['Cluster'].unique())
        cluster_to_idx = {k: i for i, k in enumerate(actual_clusters)}
        
        for k in actual_clusters:
            k_idxs = self.mave.loc[self.mave['Cluster'] == k].index
            if plot_type == 'box' or metric == 'prediction':
                data = self.mave.loc[k_idxs, 'DNN']
                boxplot_data.append(data)
            else:  # counts for bar plot
                boxplot_data.append([len(k_idxs)])
                
        # Sort if requested
        sorted_indices = None
        if sort_by_median:
            sorted_indices = self.get_cluster_order(sort_method='median')
            sorted_data = []
            for k in sorted_indices:
                idx = cluster_to_idx[k]
                sorted_data.append(boxplot_data[idx])
            boxplot_data = sorted_data
            
            # Update membership tracking
            mapping_dict = {old_k: new_k for new_k, old_k in 
                        enumerate(sorted_indices)}
            self.membership_df['Cluster_Sorted'] = self.membership_df['Cluster'].map(mapping_dict)

        if plot_type == 'box':
            # Calculate IQR
            iqr_values = [np.percentile(data, 75) - np.percentile(data, 25) 
                        for data in boxplot_data if len(data) > 0]
            average_iqr = np.mean(iqr_values) if iqr_values else 0
            
            plt.figure(figsize=(6.4, 4.8))
            plt.boxplot(boxplot_data[::-1], vert=False, 
                    showfliers=show_fliers, 
                    medianprops={'color': 'black'})
            plt.yticks(range(1, len(boxplot_data) + 1)[::10],
                    range(len(boxplot_data))[::-1][::10],
                    fontsize=fontsize)
            plt.ylabel('Clusters')
            plt.xlabel('DNN')
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.title(f'Average IQR: {average_iqr:.2f}')
            
            if show_ref and self.ref_seq is not None:
                ref_cluster = self.membership_df.loc[self.ref_idx, 'Cluster']
                if sort_by_median:
                    ref_cluster = mapping_dict[ref_cluster]
                ref_data = boxplot_data[ref_cluster]
                if len(ref_data) > 0:
                    plt.axvline(np.median(ref_data), c='red', 
                            label='Ref', zorder=-100)
                    plt.legend(loc='best')
        
        else:  # bar plot
            fig_width = 1.5 if metric == 'counts' else 1.0
            plt.figure(figsize=(fig_width, 5))
            
            y_positions = np.arange(len(boxplot_data))
            values = [np.median(data) if metric == 'prediction' else data[0] 
                    for data in boxplot_data]
            height = 0.8 if metric == 'prediction' else 1.0
            
            if show_ref and self.ref_seq is not None:
                ref_cluster = self.membership_df.loc[self.ref_idx, 'Cluster']
                if sort_by_median and sorted_indices is not None:
                    ref_cluster = np.where(sorted_indices == ref_cluster)[0][0]
                colors = ['red' if i == ref_cluster else 'C0' 
                        for i in range(len(values))]
                plt.barh(y_positions, values, height=height, color=colors)
            else:
                plt.barh(y_positions, values, height=height)
            
            plt.yticks(y_positions[::10], y_positions[::10], fontsize=fontsize)
            plt.ylabel('Cluster')
            plt.xlabel('DNN' if metric == 'prediction' else 'Count')
            plt.gca().invert_yaxis()
            plt.axvline(x=0, color='black', linewidth=0.5, zorder=100)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def generate_msm(self, gpu=False):
        """Generate Mechanism Summary Matrix (MSM) for all clusters.
        
        Parameters
        ----------
        gpu : bool
            If True, use GPU acceleration for computations (requires tensorflow)
        """
        # Get sequence length from first sequence
        seq_length = len(self.mave['Sequence'].iloc[0])
        print(f"Generating MSM for {len(self.cluster_indices)} clusters, {seq_length} positions...")
        
        if gpu:
            import tensorflow as tf
            device = '/GPU:0' if tf.test.is_built_with_cuda() else '/CPU:0'
            print(f"Using device: {device}")
        
        # Convert sequences to numpy array for faster processing
        sequences = np.array([list(seq) for seq in self.mave['Sequence']])
        
        # Initialize MSM DataFrame
        n_entries = len(self.cluster_indices) * seq_length
        self.msm = pd.DataFrame({
            'Cluster': np.repeat(self.cluster_indices, seq_length),
            'Position': np.tile(np.arange(seq_length), len(self.cluster_indices)),
            'Reference': np.nan,
            'Consensus': np.nan,
            'Entropy': np.nan
        })
        
        # Precompute one-hot encoding of reference sequence
        ref_oh = squid_utils.seq2oh(self.ref_seq, self.alphabet)
        
        # Process each cluster in parallel
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial
        
        def process_cluster(k, sequences, ref_oh):
            # Get sequences in current cluster
            k_mask = self.mave['Cluster'] == k
            seqs_k = sequences[k_mask]
            n_seqs = len(seqs_k)
            
            # Create position-wise counts matrix
            counts = np.zeros((len(self.alphabet), seq_length))
            for i, base in enumerate(self.alphabet):
                counts[i] = (seqs_k == base).sum(axis=0)
            
            # Calculate position-wise frequencies
            freqs = counts / n_seqs
            
            # Calculate entropy (vectorized)
            with np.errstate(divide='ignore', invalid='ignore'):
                pos_entropy = -np.sum(freqs * np.log2(freqs + 1e-10), axis=0)
                pos_entropy = np.nan_to_num(pos_entropy)
            
            # Get consensus sequence
            consensus_indices = np.argmax(counts, axis=0)
            consensus_seq = np.array(self.alphabet)[consensus_indices]
            
            # Calculate matches
            consensus_oh = squid_utils.seq2oh(consensus_seq, self.alphabet)
            consensus_matches = np.diagonal(consensus_oh.dot(counts)) / n_seqs * 100
            
            if self.ref_seq is not None:
                ref_matches = np.diagonal(ref_oh.dot(counts)) / n_seqs * 100
                ref_mismatches = 100 - ref_matches
            else:
                ref_mismatches = np.full(seq_length, np.nan)
            
            return k, pos_entropy, consensus_matches, ref_mismatches
        
        # Process clusters in parallel
        with ThreadPoolExecutor() as executor:
            process_fn = partial(process_cluster, sequences=sequences, ref_oh=ref_oh)
            results = list(tqdm(
                executor.map(process_fn, self.cluster_indices),
                total=len(self.cluster_indices),
                desc="Processing clusters"
            ))
        
        # Fill MSM with results
        for k, entropy, consensus, reference in results:
            mask = self.msm['Cluster'] == k
            self.msm.loc[mask, 'Entropy'] = np.tile(entropy, 1)
            self.msm.loc[mask, 'Consensus'] = np.tile(consensus, 1)
            if self.ref_seq is not None:
                self.msm.loc[mask, 'Reference'] = np.tile(reference, 1)
        
        return self.msm
    
    def plot_msm(self, column='Entropy', sort_method=None, delta_entropy=False, sort_indices=None, 
                square_cells=False, view_window=None, gui=False):
        """Visualize the Mechanism Summary Matrix (MSM) as a heatmap.
        
        Parameters
        ----------
        column : str
            Which MSM metric to visualize:
            - 'Entropy': Shannon entropy of characters at each position per cluster
            - 'Reference': Percentage of mismatches to reference sequence
            - 'Consensus': Percentage of matches to cluster consensus sequence
        sort_method : {None, 'hierarchical', 'median', 'predefined'}
            How to order the clusters:
            - None: Use original cluster ordering
            - 'hierarchical': Sort clusters by pattern similarity
            - 'median': Sort clusters by median DNN score
            - 'predefined': Use provided sort_indices
        delta_entropy : bool
            If True and column='Entropy', show change in entropy from background
            expectation (based on mutation rate)
        sort_indices : array-like, optional
            Custom cluster ordering to use when sort_method='predefined'
        square_cells : bool
            If True, set cells in MSM to be perfectly square
        view_window : list of [start, end], optional
            If provided, crop the x-axis to this window of positions
        gui : bool
            If True, return data for GUI processing without plotting
        """
        if not hasattr(self, 'msm') or self.msm is None:
            raise ValueError("MSM not generated. Call generate_msm() first.")

        # Validate inputs
        valid_columns = {'Entropy', 'Reference', 'Consensus'}
        if column not in valid_columns:
            raise ValueError(f"column must be one of: {valid_columns}")
            
        valid_sort_methods = {None, 'hierarchical', 'median', 'predefined'}
        if sort_method not in valid_sort_methods:
            raise ValueError(f"sort_method must be one of: {valid_sort_methods}")
            
        if view_window is not None:
            if not isinstance(view_window, (list, tuple)) or len(view_window) != 2:
                raise ValueError("view_window must be a list/tuple of [start, end]")
            start, end = view_window
            if start >= end:
                raise ValueError("view_window start must be less than end")
        
        # Prepare data matrix
        n_clusters = self.msm['Cluster'].max() + 1
        n_positions = self.msm['Position'].max() + 1
        matrix_data = self.msm.pivot(columns='Position', 
                                    index='Cluster', 
                                    values=column)
        
        # Apply view window if specified
        if view_window is not None:
            start, end = view_window
            matrix_data = matrix_data.iloc[:, start:end]
            n_positions = end - start
        
        # Sort clusters if requested
        cluster_order = self.get_cluster_order(sort_method=sort_method, 
                                            sort_indices=sort_indices)
        matrix_data = matrix_data.reindex(cluster_order)
        
        if gui:
            return None, None, cluster_order, matrix_data
        
        # Setup plot
        fig = plt.figure(figsize=(10, 6))
        main_ax = fig.add_subplot(111)
        
        # Get colormap settings
        cmap_settings = self._get_colormap_settings(column, delta_entropy, matrix_data)
        if delta_entropy and column == 'Entropy':
            matrix_data -= cmap_settings.pop('bg_entropy', 0)
        
        # Create heatmap
        heatmap = main_ax.pcolormesh(matrix_data, 
                                    cmap=cmap_settings['cmap'],
                                    norm=cmap_settings['norm'])
        
        # Set square cells if requested
        if square_cells:
            main_ax.set_aspect('equal')
        
        # Configure axes
        main_ax.set_xlabel('Position', fontsize=8)
        main_ax.set_ylabel('Cluster', fontsize=8)
        main_ax.invert_yaxis()
        
        # Set tick spacing based on data size
        self._configure_matrix_ticks(main_ax, n_positions, n_clusters, cluster_order)
        
        # Update x-axis ticks if using view window
        if view_window is not None:
            start, end = view_window
            x_ticks = main_ax.get_xticks()
            x_labels = [str(int(i + start)) for i in x_ticks]
            main_ax.set_xticklabels(x_labels)
        
        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(main_ax)
        cbar_ax = divider.append_axes('right', size='2%', pad=0.05)
        cbar = fig.colorbar(heatmap, cax=cbar_ax, orientation='vertical')
        
        # Set colorbar limits and label
        if column in ['Reference', 'Consensus']:
            heatmap.set_clim(0, 100)
        elif column == 'Entropy' and not delta_entropy:
            heatmap.set_clim(0, 2)
        cbar.ax.set_ylabel(cmap_settings['label'], rotation=270, fontsize=8, labelpad=10)
        
        plt.tight_layout()

    def _configure_matrix_ticks(self, ax, n_positions, n_clusters, cluster_order):
        """Configure tick marks and labels for the MSM visualization.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to configure
        n_positions : int
            Number of sequence positions
        n_clusters : int
            Number of clusters
        cluster_order : array-like
            Order of cluster indices
        """
        # Set position (x-axis) ticks
        x_skip = 10 if n_positions > 100 else 20 if n_positions > 1000 else 1
        x_ticks = np.arange(0.5, n_positions, x_skip)
        x_labels = [str(int(i-0.5)) for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=0)
        
        # Set cluster (y-axis) ticks
        y_skip = 10 if n_clusters > 10 else 1
        y_ticks = np.arange(0.5, n_clusters, y_skip)
        y_labels = [str(cluster_order[int(i-0.5)]) for i in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, rotation=0)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=6)

    def _get_colormap_settings(self, column, delta_entropy=False, matrix_data=None):
        """Get colormap settings for MSM visualization."""
        if column == 'Entropy':
            if delta_entropy:
                r = self.mut_rate
                p = np.array([1-r] + [r/(len(self.alphabet)-1)] * (len(self.alphabet)-1))
                bg_entropy = entropy(p, base=2)
                
                if matrix_data is not None:
                    return {
                        'cmap': 'seismic',
                        'norm': TwoSlopeNorm(
                            vmin=matrix_data.min().min(), 
                            vcenter=0,
                            vmax=matrix_data.max().max()
                        ),
                        'label': 'ΔH (bits)',
                        'bg_entropy': bg_entropy
                    }
                return {
                    'cmap': 'seismic',
                    'norm': None,
                    'label': 'ΔH (bits)',
                    'bg_entropy': bg_entropy
                }
                
            return {'cmap': 'rocket_r', 'norm': None, 'label': 'Entropy (bits)'}
        
        return {
            'cmap': 'Blues_r' if column == 'Reference' else 'rocket',
            'norm': None,
            'label': 'Percent mismatch' if column == 'Reference' else 'Percent match'
        }



# TODO:
# - remove squid dependency (replace with seam-nn once ready)