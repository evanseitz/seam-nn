from __future__ import division  # Should be first import

# Standard libraries
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as patches
import itertools

# Bioinformatics libraries
from Bio import motifs  # For PWM/enrichment logos

# BatchLogo package imports
from logomaker_batch.batch_logo import BatchLogo

# Local utilities
try:  # Try relative import first (for pip package)
    from . import utils
except ImportError:  # Fall back to direct import (for development/Colab)
    import utils

class MetaExplainer:
    """A class for analyzing and visualizing attribution map clusters.

    This class builds on the Clusterer class to provide detailed analysis and 
    visualization of attribution map clusters.

    Features
    --------
    Analysis
        - Mechanism Summary Matrix (MSM) generation
        - Sequence logos and attribution logos
        - Cluster membership tracking
        - Background separation and noise reduction of attribution maps

    Visualization
        - DNN score distributions per cluster
        - Sequence logos (PWM and enrichment)
        - Attribution logos (fixed and adaptive scaling)
        - Mechanism Summary Matrices
        - Cluster profile plots

    Requirements
    -----------
    - All requirements from Clusterer class
    - Biopython
    - Logomaker
    - Seaborn
    - SQUID-NN
    """
    
    def __init__(self, clusterer, mave_df, attributions, ref_idx=0,
                background_separation=False, mut_rate=0.10, sort_method='median',
                alphabet=None):
        """Initialize MetaExplainer with clusterer and data.

        Parameters
        ----------
        clusterer : Clusterer
            Initialized Clusterer object with clustering results.
        mave_df : pandas.DataFrame
            DataFrame containing sequences and their scores. Must have columns:
            - 'Sequence': DNA/RNA sequences
            - 'Score' or 'DNN': Model predictions
            - 'Cluster': Cluster assignments
        attributions : numpy.ndarray
            Attribution maps for sequences. Shape should be 
            (n_sequences, seq_length, n_characters).
        ref_idx : int, default=0
            Index of reference sequence in mave_df.
        background_separation : bool, default=False
            Whether to separate background signal from logos.
        mut_rate : float, default=0.10
            Mutation rate used for background sequence generation.
        sort_method : {'median', 'visual', None}, default='median'
            How to sort clusters in all visualizations and analyses.
            - 'median': Sort by median DNN score
            - 'visual': Sort based on hierarchical clustering of the MSM pattern
            - None: Use original cluster indices
        alphabet : list of str, optional
            List of characters to use in sequence logos.
            Default is ['A', 'C', 'G', 'T'].
        """
        # Store inputs
        self.clusterer = clusterer  # clusterer.cluster_labels should contain labels_n
        self.mave = mave_df
        self.attributions = attributions
        self.sort_method = sort_method
        self.ref_idx = ref_idx
        self.mut_rate = mut_rate
        self.alphabet = ['A', 'C', 'G', 'T']  # Default DNA alphabet
        self.background_logos = None
        
        # Initialize other attributes
        self.msm = None
        self.cluster_background = None
        self.consensus_df = None
        self.membership_df = None
        
        # Validate and process inputs
        self._validate_inputs()
        self._process_inputs()
        
        # Get the cluster ordering once at initialization
        if self.sort_method:
            self.cluster_order = self.get_cluster_order(sort_method=self.sort_method)
        else:
            self.cluster_order = None

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
                           show_ref=True, show_fliers=False, compact=False, fontsize=8, dpi=200, 
                           figsize=None, file_format='png'):
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
        show_ref : bool
            If True and reference sequence exists, highlight its cluster
        show_fliers : bool
            If True and plot_type='box', show outlier points
        compact: bool
            If False, shows full boxplots. (default: False)
            If True, uses a compact representation for boxplots with dots and IQR lines.
        fontsize : int
            Font size for tick labels
        dpi : int
            DPI for saved figure
        figsize : tuple, optional
            Figure size (width, height) in inches (default: None, uses matplotlib default)
        file_format : str, optional
            Format for saved figure (default: 'png'). Common formats: 'png', 'pdf', 'svg', 'eps'
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
                
        # Sort using class-level ordering if it exists
        if self.cluster_order is not None:
            sorted_data = []
            for k in self.cluster_order:
                idx = cluster_to_idx[k]
                sorted_data.append(boxplot_data[idx])
            boxplot_data = sorted_data
            
            # Update membership tracking
            mapping_dict = {old_k: new_k for new_k, old_k in 
                        enumerate(self.cluster_order)}
            self.membership_df['Cluster_Sorted'] = self.membership_df['Cluster'].map(mapping_dict)

        if plot_type == 'box':
            # Calculate IQR
            iqr_values = [np.percentile(data, 75) - np.percentile(data, 25) 
                        for data in boxplot_data if len(data) > 0]
            average_iqr = np.mean(iqr_values) if iqr_values else 0
            
            if figsize is not None:
                plt.figure(figsize=figsize)
            else:
                plt.figure(figsize=(6.4, 4.8))
            
            if not compact:
                plt.boxplot(boxplot_data[::-1], vert=False, 
                        showfliers=show_fliers, 
                        medianprops={'color': 'black'})
                plt.yticks(range(1, len(boxplot_data) + 1)[::10],
                        range(len(boxplot_data))[::-1][::10],
                        fontsize=fontsize)
            else:
                for pos, values in enumerate(boxplot_data[::-1]):
                    values = np.array(values)            
                    median = np.median(values)
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    plt.plot([q1, q3], [pos+1, pos+1], color='gray', lw=.5)  # plot the IQR line
                    plt.plot(median, pos+1, 'o', color='k', markersize=1, zorder=100)  # plot the median point
                plt.yticks(range(1, len(boxplot_data) + 1)[::10],
                        range(len(boxplot_data))[::-1][::10],
                        fontsize=fontsize)
            
            plt.ylabel('Clusters')
            plt.xlabel('DNN')
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.title(f'Average IQR: {average_iqr:.2f}')
            
            # Update reference cluster index if sorting is enabled
            if show_ref and self.ref_seq is not None:
                ref_cluster = self.membership_df.loc[self.ref_idx, 'Cluster']
                if self.cluster_order is not None:
                    ref_cluster = mapping_dict[ref_cluster]
                ref_data = boxplot_data[ref_cluster]
                if len(ref_data) > 0:
                    plt.axvline(np.median(ref_data), c='red', 
                            label='Ref', zorder=-100)
                    plt.legend(loc='best')

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path + '/cluster_%s_%s.%s' % (metric, plot_type, file_format), 
                           facecolor='w', dpi=dpi, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        else:  # bar plot
            if figsize is not None:
                plt.figure(figsize=figsize)
            else:
                plt.figure(figsize=(1.5, 5))
            
            y_positions = np.arange(len(boxplot_data))
            values = [np.median(data) if metric == 'prediction' else data[0] 
                    for data in boxplot_data]
            height = 1.0
            
            if show_ref and self.ref_seq is not None:
                ref_cluster = self.membership_df.loc[self.ref_idx, 'Cluster']
                if self.cluster_order is not None:
                    ref_cluster = mapping_dict[ref_cluster]
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
                plt.savefig(save_path + '/cluster_%s_%s.%s' % (metric, plot_type, file_format), 
                           facecolor='w', dpi=dpi, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def generate_msm(self, n_seqs=1000, batch_size=50, gpu=False):
        """Generate a Mechanism Summary Matrix (MSM) from cluster attribution maps.
        
        Parameters
        ----------
        n_seqs : int, default=1000
            Number of sequences to generate per cluster.
        batch_size : int, default=50
            Number of sequences to process in each batch.
        gpu : bool, default=False
            Whether to use GPU acceleration if available.
        
        Returns
        -------
        numpy.ndarray
            The Mechanism Summary Matrix with shape (n_clusters, n_clusters).
            Each entry [i,j] represents the average DNN score when applying
            cluster i's mechanism to sequences from cluster j.
        """
        # Get sequence length from first sequence
        seq_length = len(self.mave['Sequence'].iloc[0])
        
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
        ref_oh = utils.seq2oh(self.ref_seq, self.alphabet)
        
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
            consensus_oh = utils.seq2oh(consensus_seq, self.alphabet)
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
    
    def plot_msm(self, column='Entropy', delta_entropy=False, 
                square_cells=False, view_window=None, gui=False,
                show_tfbs_clusters=False, tfbs_clusters=None,
                entropy_multiplier=0.5, cov_matrix=None, row_order=None,
                revels=None, save_path=None, dpi=200, figsize=None,
                file_format='png'):
        """Visualize the Mechanism Summary Matrix (MSM) as a heatmap.
        
        Parameters
        ----------
        column : str
            Which MSM metric to visualize:
            - 'Entropy': Shannon entropy of characters at each position per cluster
            - 'Reference': Percentage of mismatches to reference sequence
            - 'Consensus': Percentage of matches to cluster consensus sequence
        delta_entropy : bool
            If True and column='Entropy', show change in entropy from background
            expectation (based on mutation rate)
        square_cells : bool
            If True, set cells in MSM to be perfectly square
        view_window : list of [start, end], optional
            If provided, crop the x-axis to this window of positions
        gui : bool
            If True, return data for GUI processing without plotting
        show_tfbs_clusters : bool
            Whether to show TFBS cluster rectangles (default: False)
        tfbs_clusters : dict, optional
            Dictionary mapping cluster IDs to lists of positions. Required if 
            show_tfbs_clusters is True.
        entropy_multiplier : float, optional
            Multiplier for entropy threshold when identifying background (default: 0.5)
        cov_matrix : numpy.ndarray, optional
            Covariance matrix for TFBS cluster plotting. Required if show_tfbs_clusters is True.
        row_order : list of int, optional
            Order of rows in cov_matrix. Required if show_tfbs_clusters is True.
        revels : pandas.DataFrame, optional
            Revels matrix for entropy calculations. Required if show_tfbs_clusters is True.
        save_path : str, optional
            Path to save figure. If None, display instead
        dpi : int
            DPI for saved figure
        figsize : tuple, optional
            Figure size (width, height) in inches (default: None, uses matplotlib default)
        file_format : str, optional
            Format for saved figure (default: 'png'). Common formats: 'png', 'pdf', 'svg', 'eps'
        """
        if show_tfbs_clusters:
            if not hasattr(self, 'msm') or self.msm is None:
                raise ValueError("MSM not generated. Call generate_msm() first.")
            if any(x is None for x in [cov_matrix, row_order, revels]):
                raise ValueError("cov_matrix, row_order, and revels required for TFBS cluster plotting")
        self.cov_matrix = cov_matrix
        self.row_order = row_order
        self.revels = revels
        
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
        
        cluster_order = self.cluster_order if self.cluster_order is not None else np.sort(self.mave['Cluster'].unique())
        matrix_data = matrix_data.reindex(cluster_order)
        
        if gui:
            return None, None, cluster_order, matrix_data
        
        # Setup plot
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
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
        
        # Add TFBS cluster rectangles if requested
        if show_tfbs_clusters and tfbs_clusters is not None:            
            # Define entropy threshold for active regions using instance mut_rate
            null_rate = 1 - self.mut_rate
            background_entropy = entropy([null_rate, (1-null_rate)/3, (1-null_rate)/3, (1-null_rate)/3], base=2)
            entropy_threshold = background_entropy * entropy_multiplier

            # Store active clusters by TFBS
            active_clusters_by_tfbs = {}
            
            for cluster, positions in tfbs_clusters.items():
                original_indices = self.cov_matrix.index.tolist()
                reordered_positions = [original_indices[self.row_order.index(pos)] 
                                     for pos in positions]
                
                if reordered_positions:
                    start = min(reordered_positions)
                    end = max(reordered_positions)
                    
                    # Find all clusters where this TFBS is active
                    active_clusters = []
                    for cluster_idx in range(len(matrix_data)):
                        # Get the original cluster index if sorting is used
                        if self.sort_method and self.cluster_order is not None:
                            original_cluster_idx = self.cluster_order[cluster_idx]
                        else:
                            original_cluster_idx = cluster_idx
                        
                        # Use revels for entropy calculation with original index
                        cluster_entropy = self.revels.iloc[original_cluster_idx, start:end + 1]
                        mean_entropy = cluster_entropy.mean()
                        if mean_entropy < entropy_threshold:
                            # Store the sorted index position for rectangle drawing
                            active_clusters.append(cluster_idx)

                    # Store active clusters for this TFBS
                    active_clusters_by_tfbs[cluster] = active_clusters
                    
                    # Group consecutive clusters
                    for k, g in itertools.groupby(enumerate(active_clusters), 
                                                lambda x: x[0] - x[1]):
                        group = list(map(lambda x: x[1], g))
                        if group:
                            rect_start = min(group)
                            rect_height = len(group)
                            
                            # If using view window, adjust start position
                            plot_start = start
                            plot_end = end
                            if view_window:
                                if end < view_window[0] or start > view_window[1]: # rectangle outside view window
                                    continue
                                plot_start = max(start, view_window[0])
                                plot_end = min(end, view_window[1])
                                plot_start -= view_window[0]
                                plot_end -= view_window[0]
                            
                            rect = patches.Rectangle(
                                (plot_start, rect_start), 
                                plot_end - plot_start + 1, rect_height,
                                linewidth=1, edgecolor='black', 
                                facecolor='none'
                            )
                            main_ax.add_patch(rect)
            
            # Store active clusters information
            self.active_clusters_by_tfbs = active_clusters_by_tfbs
        
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

        if save_path:
            if show_tfbs_clusters is False and tfbs_clusters is None:
                plt.savefig(save_path + f'/msm_{column.lower()}.{file_format}', 
                           facecolor='w', dpi=dpi, bbox_inches='tight')
            elif show_tfbs_clusters is not None and tfbs_clusters is not None and column == 'Entropy':
                plt.savefig(save_path + f'/msm_{column.lower()}_identified.{file_format}', 
                           facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig, main_ax

    def _configure_matrix_ticks(self, ax, n_positions, n_clusters, cluster_order):
        """Configure tick marks and labels for MSM visualization.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to configure.
        n_positions : int
            Number of sequence positions.
        n_clusters : int
            Number of clusters.
        cluster_order : array-like
            Order of clusters for y-axis labels.
        """
        cluster_order = self.cluster_order if self.cluster_order is not None else np.sort(self.mave['Cluster'].unique())
        
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
            'cmap': 'Blues' if column == 'Reference' else 'rocket', #Blues_r for dark blue background
            'norm': None,
            'label': 'Percent mismatch' if column == 'Reference' else 'Percent match'
        }

    def generate_logos(self, logo_type='average', background_separation=False,
                       mut_rate=0.01, entropy_multiplier=0.5, adaptive_background_scaling=False, 
                       figsize=(20, 2.5), batch_size=50, font_name='sans', stack_order='big_on_top',
                       center_values=True, color_scheme='classic', font_weight=None, fade_below=0.5,
                       shade_below=0.5, width=0.9):
        """Generate sequence or attribution logos for each cluster.
        
        This method creates visualization logos that represent either the average attribution
        patterns or sequence patterns within each cluster. It can optionally remove background
        signal to highlight cluster-specific patterns.
        
        Parameters
        ----------
        logo_type : {'average', 'pwm', 'enrichment'}, default='average'
            Type of logo to generate:
            - 'average': Shows average attribution values (based on attribution maps)
            - 'pwm': Shows position weight matrix of nucleotide frequencies (based on sequence statistics)
            - 'enrichment': Shows nucleotide enrichment relative to background (based on sequence statistics)
        background_separation : bool, default=False
            Whether to remove background signal from logos. Only applies to 'average' logos.
            When True, subtracts the background pattern computed by compute_background(),
            forcused on highly variable positions.
        mut_rate : float, default=0.01
            Mutation rate for background entropy calculation. Only used if 
            background_separation=True.
        entropy_multiplier : float, default=0.5
            Controls stringency of background position identification via a multiplier on the background entropy.
            Only used if background_separation=True.
        adaptive_background_scaling : bool, default=False
            If True and background_separation=True, uniformly scales the background pattern
            differently for each cluster based on the magnitude of its background signal.
            This is useful when clusters have similar background patterns but at
            different scales.
        figsize : tuple, default=(20, 2.5)
            Figure size in inches (width, height).
        batch_size : int, default=50
            Number of logos to process in each batch.
        font_name : str, default='sans'
            Font name for logo text.
        stack_order : {'big_on_top', 'small_on_top', 'fixed'}, default='big_on_top'
            How to order nucleotides in each stack:
            - 'big_on_top': Largest values on top
            - 'small_on_top': Smallest values on top
            - 'fixed': Fixed order (A, C, G, T)
        center_values : bool, default=True
            Whether to center values in each position. Only applies to 'average' logos.
        color_scheme : str or dict, default='classic'
            Color scheme for logo characters.
        font_weight : str or int, optional
            Font weight for logo text. Can be string ('normal', 'bold', etc.) or numeric (0-1000).
        fade_below : float, default=0.5
            Controls alpha transparency for negative values. Higher values make negative values more transparent.
        shade_below : float, default=0.5
            Controls color darkening for negative values. Higher values make negative values darker.
        width : float, default=0.9
            Controls the horizontal width of each character.
        """
        # Get sorted cluster order using class attribute
        cluster_order = self.get_cluster_order(sort_method=self.sort_method)
        
        # Compute background if needed
        if background_separation and logo_type == 'average':
            if not hasattr(self, 'background'):
                self.compute_background(mut_rate, entropy_multiplier, adaptive_background_scaling)
        
        # For enrichment logos, compute background PFM if not already done
        if logo_type == 'enrichment' and not hasattr(self, 'background_pfm'):
            # Calculate background PFM from all sequences
            all_seqs = self.mave['Sequence']
            seq_array = motifs.create(all_seqs, alphabet=self.alphabet)
            self.background_pfm = seq_array.counts
        
        # Get cluster matrices
        cluster_matrices = []
        for i, k in enumerate(tqdm(cluster_order, desc='Generating matrices')):
            k_idxs = self.mave['Cluster'] == k
            seqs_k = self.mave.loc[k_idxs, 'Sequence']
            
            if logo_type == 'average':
                maps_avg = np.mean(self.attributions[k_idxs], axis=0)
                if background_separation:
                    # Always use background_scaling (will be 1s if adaptive scaling is disabled)
                    maps_avg -= self.background_scaling[i] * self.background
                cluster_matrices.append(maps_avg)
                
            elif logo_type in ['pwm', 'enrichment']:
                center_values = False
                # Calculate position frequency matrix
                seq_array = motifs.create(seqs_k, alphabet=self.alphabet)
                pfm = seq_array.counts
                pseudocounts = 0.5
                
                if logo_type == 'pwm':
                    # Convert to PPM and calculate information content
                    ppm = pd.DataFrame(pfm.normalize(pseudocounts=pseudocounts))
                    background = getattr(self, 'background_freqs', 
                                       np.array([1.0 / len(self.alphabet)] * len(self.alphabet)))
                    ppm += 1e-6  # Avoid log(0)
                    info_content = np.sum(ppm * np.log2(ppm / background), axis=1)
                    cluster_matrices.append(np.array(ppm.multiply(info_content, axis=0)))
                    
                else:  # enrichment
                    # Calculate enrichment relative to background frequencies
                    enrichment = (pd.DataFrame(pfm) + pseudocounts) / \
                               (pd.DataFrame(self.background_pfm) + pseudocounts)
                    cluster_matrices.append(np.log2(enrichment))
        
        # Stack matrices into 3D array
        logo_array = np.stack(cluster_matrices)
        
        # Store both raw and background-separated maps
        if logo_type == 'average':
            if background_separation:
                # Store both versions - the raw maps and the background-separated maps
                self.cluster_maps = np.stack([np.mean(self.attributions[self.mave['Cluster'] == k], axis=0) 
                                            for k in cluster_order])
                self.cluster_maps_no_bg = logo_array.copy()  # These are already background-separated
            else:
                # If no background separation, just store the raw maps
                self.cluster_maps = logo_array.copy()

        # Always compute global y-limits for attribution logos
        y_min_max = None
        if logo_type == 'average':
            y_mins = []
            y_maxs = []
            # Make a copy of logo_array to avoid modifying the original when centering
            matrices = logo_array.copy()
            if center_values:
                # Center all matrices if center_values is True
                for i, matrix in enumerate(matrices):
                    matrices[i] = matrix - np.expand_dims(np.mean(matrix, axis=1), axis=1)
            
            # Calculate y-limits from either centered or uncentered matrices
            for matrix in matrices:
                positive_mask = matrix > 0
                positive_matrix = matrix * positive_mask
                positive_sums = positive_matrix.sum(axis=1)
                
                negative_mask = matrix < 0
                negative_matrix = matrix * negative_mask
                negative_sums = negative_matrix.sum(axis=1)
                
                y_mins.append(negative_sums.min())
                y_maxs.append(positive_sums.max())
            
            y_min_max = [min(y_mins), max(y_maxs)]
        
        batch_logos = BatchLogo(
            logo_array,
            alphabet=self.alphabet,
            fig_size=figsize,
            batch_size=batch_size,
            font_name=font_name,
            stack_order=stack_order,
            center_values=center_values,
            y_min_max=y_min_max,
            color_scheme=color_scheme,
            font_weight=font_weight,
            fade_below=fade_below,
            shade_below=shade_below,
            width=width
        )
        batch_logos.process_all()
        
        self.batch_logos = batch_logos
        
        return batch_logos

    def show_sequences(self, cluster_idx):
        """Show sequences belonging to a specific cluster.
        
        Parameters
        ----------
        cluster_idx : int
            Index of cluster to show sequences for. If sorting was specified
            during initialization, this index refers to the sorted order
            (e.g., 0 is the first cluster after sorting).
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing sequences and scores for the specified cluster.
        """
        # Get original cluster index using class-level sorting if available
        if self.cluster_order is not None:
            original_idx = self.cluster_order[cluster_idx]
        else:
            original_idx = cluster_idx
        
        # Get sequences from the specified cluster
        cluster_seqs = self.mave[self.mave['Cluster'] == original_idx]
        
        return cluster_seqs[['Sequence', 'DNN']]

    def plot_cluster_profiles(self, profiles, save_dir=None, dpi=200, figsize=None, file_format='png'):
        """Plot overlay of profiles associated with each cluster.
        
        Parameters
        ----------
        profiles : np.ndarray
            Array of profile data corresponding to sequences in mave_df
        save_dir : str, optional
            Directory to save profile plots. If None, displays instead.
        dpi : int
            DPI for saved figures
        figsize : tuple, optional
            Figure size (width, height) in inches (default: None, uses matplotlib default)
        file_format : str, optional
            Format for saved figure (default: 'png'). Common formats: 'png', 'pdf', 'svg', 'eps'
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for k in self.cluster_indices:
            k_idxs = self.mave.loc[self.mave['Cluster'] == k].index
            cluster_profiles = profiles[k_idxs]
            
            if figsize is not None:
                plt.figure(figsize=figsize)
            else:
                plt.figure(figsize=(10, 5))
                
            for profile in cluster_profiles:
                plt.plot(profile, alpha=0.1, color='gray')
            plt.plot(cluster_profiles.mean(axis=0), color='red', linewidth=2)
            
            plt.title(f'Cluster {k} Profiles')
            plt.xlabel('Position')
            plt.ylabel('Value')
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'cluster_{k}_profiles.{file_format}'), 
                           dpi=dpi, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

    def compute_background(self, mut_rate=0.01, entropy_multiplier=0.5, adaptive_background_scaling=False):
        """Compute background signal based on entropic positions.
        
        This method identifies and computes background signal patterns for each cluster
        based on positions with high entropy (high variability). The background can be
        computed either uniformly across all clusters or with cluster-specific scaling.
        
        Parameters
        ----------
        mut_rate : float, default=0.01
            Mutation rate used to calculate background entropy threshold. Higher values
            will identify more positions as entropic.
        entropy_multiplier : float, default=0.5
            Factor to multiply background entropy by for threshold. Lower values make
            the threshold more stringent (fewer positions identified as entropic).
        adaptive_background_scaling : bool, default=False
            If True, computes a scaling factor for each cluster that best matches the
            magnitude of that cluster's background signal. This is useful when different
            clusters have similar background patterns but at different scales. If False,
            uses the same background scale for all clusters.
            
        Notes
        -----
        The background computation process:
        1. Identifies entropic (highly variable) positions in each cluster
        2. Computes the average attribution pattern at these positions
        3. If adaptive_background_scaling is True, computes a scaling factor for each
           cluster based on positions that are entropic in both that cluster and the
           global background
        """
        # Calculate background entropy threshold
        null_rate = 1 - mut_rate
        background_entropy = entropy(
            np.array([null_rate, (1-null_rate)/3, (1-null_rate)/3, (1-null_rate)/3]),
            base=2
        )
        entropy_threshold = background_entropy * entropy_multiplier
        
        # Initialize cluster background matrix
        n_clusters = len(self.mave['Cluster'].unique())
        seq_length = self.attributions.shape[1]
        n_chars = len(self.alphabet)
        cluster_background = np.zeros(shape=(n_clusters, seq_length, n_chars))
        
        # Compute background for each cluster
        for idx, k in enumerate(self.get_cluster_order()):
            k_idxs = self.mave['Cluster'] == k
            child_maps = self.attributions[k_idxs]
            
            # Get entropic positions for this cluster
            entropic_positions = self._get_entropic_positions(k, entropy_threshold)
            
            if len(entropic_positions) == 0:
                continue
            
            # Compute background for entropic positions
            for ep in entropic_positions:
                for child in child_maps:
                    cluster_background[idx, ep, :] += child[ep, :]
            cluster_background[idx] /= len(child_maps)
        
        # Store cluster backgrounds and compute global background
        self.cluster_backgrounds = cluster_background
        self.background = np.mean(cluster_background, axis=0)
        
        # Initialize background scaling factors
        background_scaling = np.ones(len(self.cluster_backgrounds))
        
        # Compute cluster-specific scaling factors if requested
        if adaptive_background_scaling:
            for i in range(len(self.cluster_backgrounds)):
                cluster_bg = self.cluster_backgrounds[i]
                # Get positions that have signal in both cluster and global background
                entropic_mask = np.any(cluster_bg != 0, axis=1)  # True for positions that were entropic in this cluster
                global_mask = np.any(self.background != 0, axis=1)  # True for positions that were entropic in any cluster
                valid_positions = entropic_mask & global_mask  # Only use positions that were entropic in both
                
                # Compute scaling using only valid positions
                if np.any(valid_positions):
                    alpha = np.sum(np.abs(cluster_bg[valid_positions])) / np.sum(np.abs(self.background[valid_positions]))
                    background_scaling[i] = alpha
        
        self.background_scaling = background_scaling
        
        # Create BatchLogo instance for backgrounds
        self.background_logos = BatchLogo(
            cluster_background,  # Pass the cluster_background directly
            alphabet=self.alphabet,
            fig_size=[20, 2.5],
            batch_size=50,
            font_name='sans'
        )
        self.background_logos.process_all()
        
        return self.background_logos

    def _get_entropic_positions(self, cluster, threshold):
        """Get positions with entropy above threshold for given cluster.
        
        Parameters
        ----------
        cluster : int
            Cluster index
        threshold : float
            Entropy threshold value
        
        Returns
        -------
        np.ndarray
            Array of positions with entropy above threshold
        """
        # Get sequences for this cluster
        k_idxs = self.mave['Cluster'] == cluster
        seqs = self.mave.loc[k_idxs, 'Sequence']
        
        # Convert sequences to position-specific frequency matrix
        seq_length = len(seqs.iloc[0])
        char_counts = np.zeros((seq_length, len(self.alphabet)))
        
        for seq in seqs:
            for pos, char in enumerate(seq):
                char_idx = self.alphabet.index(char)
                char_counts[pos, char_idx] += 1
        
        # Convert to frequencies
        freqs = char_counts / len(seqs)
        
        # Calculate entropy at each position
        entropies = np.zeros(seq_length)
        for pos in range(seq_length):
            pos_freqs = freqs[pos]
            # Avoid log(0) by only considering non-zero frequencies
            valid_freqs = pos_freqs[pos_freqs > 0]
            entropies[pos] = -np.sum(valid_freqs * np.log2(valid_freqs))
        
        # Return positions above threshold
        return np.where(entropies > threshold)[0]

    def get_cluster_maps(self, cluster_idx):
        """Get attribution maps belonging to a specific cluster.
        
        Parameters
        ----------
        cluster_idx : int
            Index of cluster to get maps for. If sorting was specified
            during initialization, this index refers to the sorted order
            (e.g., 0 is the first cluster after sorting).
        
        Returns
        -------
        numpy.ndarray
            Attribution maps for the specified cluster.
        """
        # Get original cluster index using class-level sorting if available
        if self.cluster_order is not None:
            original_idx = self.cluster_order[cluster_idx]
        else:
            original_idx = cluster_idx
        
        # Get maps from the specified cluster
        cluster_maps = self.attributions[self.mave['Cluster'] == original_idx]
        
        return cluster_maps

    def plot_attribution_variation(self, scope='all', metric='std', save_path=None, view_window=None, 
                                 figsize=None, dpi=600, colors=None, xtick_spacing=5, file_format='png'):
        """Visualize the variation in attribution values across attribution maps for each nucleotide position.
        
        Parameters
        ----------
        scope : {'all', 'clusters'}, default='all'
            Scope of variation calculation:
            - 'all': Use all individual attribution maps
            - 'clusters': Use cluster-averaged attribution maps
        metric : {'std', 'var'}, default='std'
            Metric to use for variation calculation:
            - 'std': Standard deviation
            - 'var': Variance
        save_path : str, optional
            Path to save figure. If None, display instead.
        view_window : list of [start, end], optional
            If provided, crop the x-axis to this window of positions.
        figsize : tuple, optional
            Figure size (width, height) in inches (default: None, uses matplotlib default)
        dpi : int, default=600
            DPI for saved figure.
        colors : dict, optional
            Dictionary mapping nucleotide indices to RGB colors.
            Default: {0: [0, .5, 0], 1: [0, 0, 1], 2: [1, .65, 0], 3: [1, 0, 0]}
            for A, C, G, T respectively.
        xtick_spacing : int, default=5
            Show x-axis labels every nth position. Set to 1 to show all positions.
        file_format : str, optional
            Format for saved figure (default: 'png'). Common formats: 'png', 'pdf', 'svg', 'eps'
            
        Returns
        -------
        numpy.ndarray
            Array of variation values (std or var) for each position and nucleotide
        """
        if not hasattr(self, 'attributions') or self.attributions is None:
            raise ValueError("No attribution maps found. Run compute() first.")
            
        if scope not in ['all', 'clusters']:
            raise ValueError("scope must be one of: 'all', 'clusters'")
            
        if metric not in ['std', 'var']:
            raise ValueError("metric must be one of: 'std', 'var'")
            
        # Get appropriate attribution maps based on scope
        if scope == 'all':
            maps_to_analyze = self.attributions
        else:  # clusters
            # Get cluster-averaged attribution maps
            cluster_maps = []
            cluster_order = self.cluster_order if self.cluster_order is not None else np.sort(self.mave['Cluster'].unique())
            for k in cluster_order:
                k_maps = self.get_cluster_maps(k)
                cluster_maps.append(np.mean(k_maps, axis=0))
            maps_to_analyze = np.stack(cluster_maps, axis=0)
            
        # Calculate variation metric across maps
        if metric == 'std':
            variation = np.std(maps_to_analyze, axis=0)  # shape: (L, A)
        else:  # var
            variation = np.var(maps_to_analyze, axis=0)  # shape: (L, A)
        
        # Set default colors if not provided
        if colors is None:
            colors = {
                0: [0, .5, 0],   # A: green
                1: [0, 0, 1],    # C: blue
                2: [1, .65, 0],  # G: orange
                3: [1, 0, 0]     # T: red
            }
        
        # Create position indices and apply view window
        if view_window is not None:
            start, end = view_window
            variation = variation[start:end]
            plot_positions = np.arange(len(variation))
            seq_positions = np.arange(start, end)
        else:
            plot_positions = np.arange(len(variation))
            seq_positions = plot_positions
        
        # Create plot
        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure(figsize=(20, 1.5))
        
        # Plot bars for each nucleotide
        bar_width = 0.2
        for i, nuc in enumerate(self.alphabet):
            plt.bar(plot_positions + i * bar_width, variation[:, i], 
                   width=bar_width, color=colors[i], label=nuc)
        
        # Customize plot with spaced ticks
        tick_mask = np.zeros_like(plot_positions, dtype=bool)
        tick_mask[::xtick_spacing] = True
        plt.xticks(plot_positions[tick_mask] + 1.5 * bar_width, 
                  seq_positions[tick_mask],
                  rotation=90)
        
        # Set x-axis limits to show full range with small padding
        plt.xlim(plot_positions[0] - 0.25, plot_positions[-1] + 0.85)
        
        # Add y-axis label based on metric
        plt.ylabel('Std Dev' if metric == 'std' else 'Variance')
        
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path + '/attribution_variation_%s_%s.%s' % (metric, scope, file_format), 
                       dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        return variation