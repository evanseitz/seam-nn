# Standard libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Bioinformatics libraries
from Bio import motifs  # pip install biopython
from scipy.stats import entropy
from logomaker_batch.batch_logo import BatchLogo
from matplotlib.colors import TwoSlopeNorm

# Local imports
# Try relative import first (for pip package)
#try:
import utils
# Fall back to direct import (for Colab/direct usage)
#except ImportError:
    # Add the directory containing utils.py to the Python path
#    module_dir = os.path.dirname(os.path.abspath(__file__))
#    if module_dir not in sys.path:
#        sys.path.append(module_dir)
#   import utils

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
        - Bias removal and normalization

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
                bias_removal=False, mut_rate=0.10, sort_method='median',
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
        bias_removal : bool, default=False
            Whether to remove background signal from logos.
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
        self.ref_idx = ref_idx
        self.bias_removal = bias_removal
        self.mut_rate = mut_rate
        self.sort_method = sort_method
        self.alphabet = alphabet or ['A', 'C', 'G', 'T']
        
        # Initialize other attributes
        self.msm = None
        self.cluster_bias = None
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
                           show_ref=True, show_fliers=False, fontsize=8, dpi=200):
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
        
        else:  # bar plot
            fig_width = 1.5 if metric == 'counts' else 1.0
            plt.figure(figsize=(fig_width, 5))
            
            y_positions = np.arange(len(boxplot_data))
            values = [np.median(data) if metric == 'prediction' else data[0] 
                    for data in boxplot_data]
            height = 0.8 if metric == 'prediction' else 1.0
            
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
            plt.savefig(save_path, facecolor='w', dpi=dpi, bbox_inches='tight')
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
                square_cells=False, view_window=None, gui=False):
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
        """
        if not hasattr(self, 'msm') or self.msm is None:
            raise ValueError("MSM not generated. Call generate_msm() first.")

        # Validate inputs
        valid_columns = {'Entropy', 'Reference', 'Consensus'}
        if column not in valid_columns:
            raise ValueError(f"column must be one of: {valid_columns}")
            
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
        
        cluster_order = self.cluster_order if self.cluster_order is not None else np.sort(self.mave['Cluster'].unique())
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
            'cmap': 'Blues_r' if column == 'Reference' else 'rocket',
            'norm': None,
            'label': 'Percent mismatch' if column == 'Reference' else 'Percent match'
        }

    def generate_logos(self, logo_type='attribution', background_removal=False, 
                      figsize=(20, 2.5), batch_size=50, font_name='sans',
                      stack_order='big_on_top', center_values=True, 
                      fixed_ylim=False, color_scheme='classic'):
        """Generate sequence or attribution logos for each cluster.

        Parameters
        ----------
        logo_type : {'attribution', 'pwm', 'enrichment'}, default='attribution'
            Type of logo to generate.
        background_removal : bool, default=False
            Whether to remove background signal from logos.
        figsize : tuple of float, default=(20, 2.5)
            Figure size (width, height) in inches.
        batch_size : int, default=50
            Number of logos to process in each batch.
        font_name : str, default='sans'
            Font family to use for logo characters.
        stack_order : {'big_on_top', 'small_on_top', 'fixed'}, default='big_on_top'
            How to stack characters in logos.
        center_values : bool, default=True
            Whether to center values around zero for each position.
        fixed_ylim : bool, default=False
            Whether to use same y-axis limits across all logos.
        color_scheme : str, default='classic'
            Color scheme for sequence logos.

        Returns
        -------
        BatchLogo
            Object containing the generated logos.
        """
        # Get sorted cluster order using class attribute
        cluster_order = self.get_cluster_order(sort_method=self.sort_method)
        
        # Get cluster matrices
        cluster_matrices = []
        for k in tqdm(cluster_order, desc='Generating matrices'):
            k_idxs = self.mave['Cluster'] == k
            seqs_k = self.mave.loc[k_idxs, 'Sequence']
            
            if logo_type == 'attribution':
                # Average attribution maps for this cluster to create noise-reduced meta-attribution map
                maps_avg = np.mean(self.attributions[k_idxs], axis=0)
                if background_removal:
                    maps_avg -= self.background
                cluster_matrices.append(maps_avg)
                
            elif logo_type in ['pwm', 'enrichment']:
                # Calculate position frequency matrix
                seq_array = motifs.create(seqs_k, alphabet=self.alphabet)
                pfm = seq_array.counts
                pseudocounts = 0.5
                
                if logo_type == 'pwm':
                    # Convert to PPM and calculate information content
                    ppm = pd.DataFrame(pfm.normalize(pseudocounts=pseudocounts))
                    background = np.array([1.0 / len(self.alphabet)] * len(self.alphabet))
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
        
        # Only compute global y-limits if fixed_ylim is True
        y_min_max = None
        if fixed_ylim and logo_type == 'attribution':
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
            color_scheme=color_scheme
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

    def generate_variability_logo(self, logo_type='attribution', **logo_kwargs):
        """Generate a variability logo by overlaying all cluster logos.
        
        Parameters
        ----------
        logo_type : {'attribution', 'pwm', 'enrichment'}, default='attribution'
            Type of logo to generate.
        **logo_kwargs
            Additional arguments passed to generate_logos().
        
        Returns
        -------
        matplotlib.figure.Figure
            The combined variability logo figure.
        
        Raises
        ------
        ValueError
            If logos haven't been generated yet. Call generate_logos() first.
        """
        if not hasattr(self, 'batch_logos'):
            raise ValueError("Logos not generated yet. Call generate_logos() first.")
        
        # Use existing BatchLogo object
        batch_logos = self.batch_logos
        
        # Create a new figure for the variability logo
        fig, ax = plt.subplots(figsize=logo_kwargs.get('figsize', (20, 2.5)))
        
        # For each position and character, combine heights across all clusters
        for pos in range(batch_logos.seq_length):
            for char_idx, char in enumerate(self.alphabet):
                # Get all heights for this character at this position across clusters
                heights = batch_logos.get_heights(pos, char_idx)
                
                # Draw character with alpha proportional to frequency of appearance
                if any(heights):  # Only draw if character appears in any cluster
                    alpha = len([h for h in heights if h != 0]) / len(heights)
                    batch_logos.draw_character(ax, char, pos, heights.mean(), alpha=alpha)
        
        return fig

"""
# TODO:
# - line up the MSM with the logos, and with the bar plot
# - save directory parameter
# - make sure all imports and __init__.py are correctly set up for pip package


1. Implement logo generation and plotting functionality (lines 755-823 from meta_explainer_orig.py)
   - Handle bias removal
   - Save to appropriate directories

2. Implement variability logo rendering (lines 833-854 from meta_explainer_orig.py)
   - Use PIL for image processing
   - Implement darken_blend function
   - Process all PNGs in directory
   - Save combined variability logo

3. Add file/directory management (lines 414-420 from meta_explainer_orig.py)
   - Save cluster consensus info
   - Handle bias removal directories
   - Save average matrices

4. Add profile plotting functionality (lines 741-753 from meta_explainer_orig.py)
   - Plot individual profiles for each cluster
   - Handle alpha blending for overlays
   - Save to profiles directory

5. Add configuration parameters from meta_explainer_orig.py:
   - embedding options (lines 157-176)
   - logo generation flags (lines 179-189)
   - sequence handling (lines 194-202)

6. Background removal module needs to be implemented:
   - Add background computation and storage in MetaExplainer.__init__
   - Rename all "bias" references to "background"
   - Add documentation about background separation
   - Fix AttributeError in generate_logos for self.background

7. Class Storage Standardization:
   - All key outputs should be both returned and stored as instance variables
   - Examples to update:
     - compiler.py: store mave internally
     - clusterer.py: store clustering results
     - meta_explainer.py: store msm results
   - Follow pattern established by attributer.attributions

8. Package Structure Updates:
   - Update all relative imports in logomaker_batch:
     - batch_logo.py needs relative imports
     - gpu_utils.py needs relative imports
     - colors.py needs relative imports
     - Any matplotlib utilities need relative imports
   - Ensure all imports use the pattern: from logomaker_batch.xxx import XXX
   - Test package imports work both in development and after pip install

NOTE: Several components from meta_explainer_orig.py have already been modernized:

1. generate_regulatory_df has been upgraded to generate_msm() with:
   - Parallel processing
   - Vectorized operations
   - Better memory management
   - GPU support option
   - Progress tracking
   Reference: meta_explainer.py lines 248-318

2. Boxplot functionality and median sorting are now handled by:
   - plot_cluster_stats() for visualization (lines 130-213)
   - get_cluster_order() for cluster sorting
   Both include improved error handling and more configuration options

3. aesthetic_lvl parameter has been deprecated:
   - Originally used to trade off logo quality for speed
   - No longer needed due to BatchLogo's optimized rendering
   - All logos now render at highest quality with minimal performance impact
"""