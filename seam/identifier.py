import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.patches as patches
import sys
import seaborn as sns
import matplotlib.colors as mpl
import itertools
from scipy.stats import entropy
import matplotlib.colors as colors
import os

class Identifier:
    """Class for identifying and analyzing transcription factor binding sites (TFBSs) from attribution maps.
    
    The Identifier class takes attribution maps from a MetaExplainer and identifies distinct TFBSs
    by analyzing patterns of activity across clusters. It uses a multi-step process:
    
    1. Covariance Analysis:
       - Analyzes the covariance between positions in the attribution maps
       - Identifies regions that show coordinated activity across clusters
       - Uses hierarchical clustering to group positions into potential TFBSs
    
    2. TFBS Identification:
       - Defines TFBS regions based on clustered covariance patterns
       - Determines which clusters are active for each TFBS using entropy-based thresholds
       - Creates a binary or continuous binding configuration matrix showing TFBS activity levels in each cluster
    
    3. Binding Configuration Assignment:
       - Assigns clusters to specific TFBS binding configurations (e.g., A only, A+B, background)
       - Uses a distance-based scoring system to find the best cluster for each configuration
       - For background configuration, finds clusters with minimal TFBS activity across all TFBSs
    
    Key Concepts:
    - TFBS Activity: Measured as 1 - (normalized entropy), where higher values indicate
      stronger TFBS activity in a cluster
    - Binding Configuration Matrix: Shows binary or continuous activity levels (0-1) for each TFBS in each cluster
    - Binding Configuration Assignments: Maps each possible TFBS combination to its optimal cluster
    
    Parameters
    ----------
    msm_df : pandas.DataFrame
        Mechanism Summary Matrix (MSM) data from MetaExplainer, containing entropy
        or other activity measures for each position in each cluster
    meta_explainer : MetaExplainer
        Instance of MetaExplainer class that generated the attribution maps
    column : str, optional
        Column from MSM to use for analysis (default: 'Entropy')
    
    Attributes
    ----------
    revels : pandas.DataFrame
        Pivoted MSM data with clusters as rows and positions as columns
    cov_matrix : pandas.DataFrame
        Covariance matrix between positions, used for TFBS identification
    tfbs_clusters : dict
        Dictionary mapping TFBS labels to their constituent positions
    entropy_multiplier : float
        Threshold multiplier for determining active clusters
    active_clusters_by_tfbs : dict
        Dictionary mapping TFBS labels to their active clusters
    
    Notes
    -----
    The class uses entropy-based measures to identify TFBS activity, where:
    - Lower entropy indicates more specific, TFBS-like activity
    - Higher entropy indicates more background-like activity
    - Activity is normalized relative to background entropy to account for
      mutation rate and sequence composition
    """
    
    def __init__(self, msm_df, meta_explainer, column='Entropy'):
        """Initialize Identifier with MSM data and MetaExplainer instance.
        
        Parameters
        ----------
        msm_df : pandas.DataFrame
            MSM data from MetaExplainer
        meta_explainer : MetaExplainer
            Instance of MetaExplainer class
        column : str, optional
            Column to use for analysis (default: 'Entropy')
        """
        self.df = msm_df
        self.meta_explainer = meta_explainer
        self.column = column
        self.nC = self.df['Cluster'].max() + 1
        self.nP = self.df['Position'].max() + 1
        
        # Create pivot table for analysis
        self.revels = self.df.pivot(
            columns='Position', 
            index='Cluster', 
            values=self.column
        )
        
        # Calculate covariance matrix
        self.cov_matrix = self.revels.cov()
        
    def cluster_msm_covariance(self, method='average', n_clusters=None, cut_height=None):
        """
        Cluster the covariance matrix using hierarchical clustering.
        
        Parameters
        ----------
        method : str, optional
            Linkage method for hierarchical clustering (default: 'average')
        n_clusters : int, optional
            Number of clusters to form. If None, will use cut_height or automatic detection.
            Note: This is the number of clusters BEFORE removing the largest cluster.
        cut_height : float, optional
            Height at which to cut the dendrogram. If None and n_clusters is None,
            will use automatic gap detection.
        
        Returns
        -------
        dict
            Dictionary mapping cluster labels to positions
        """
        # Validate inputs
        if n_clusters is not None and cut_height is not None:
            raise ValueError("Cannot specify both n_clusters and cut_height. Please provide only one.")
        
        # Store clustering method
        self.cluster_method = 'n_clusters' if n_clusters is not None else 'cut_height'
        self.n_clusters = n_clusters
        
        # Compute linkage for rows
        row_linkage = hierarchy.linkage(distance.pdist(self.cov_matrix), 
                                      method=method)
        row_dendrogram = hierarchy.dendrogram(row_linkage, no_plot=True, 
                                              color_threshold=-np.inf)
        self.row_order = row_dendrogram['leaves']
        
        # Reorder covariance matrix
        self.reordered_cov_matrix = self.cov_matrix.iloc[self.row_order, :].iloc[:, self.row_order]
        
        # Determine clustering criterion
        if n_clusters is not None:
            # Adjust n_clusters to account for removal of largest cluster
            n_clusters_adjusted = n_clusters + 1
            
            # Use specified number of clusters
            cluster_labels = hierarchy.fcluster(row_linkage, n_clusters_adjusted, 
                                              criterion='maxclust')
            # Get the cut height that corresponds to n_clusters
            distances = row_linkage[:, 2]
            sorted_distances = np.sort(distances)
            self.cut_height = sorted_distances[-n_clusters_adjusted + 1] * 0.99  # Slight adjustment
        else:
            if cut_height is None:
                # Find optimal cut using gap statistic
                distances = row_linkage[:, 2]
                gaps = np.diff(np.sort(distances))
                n_significant_gaps = sum(gaps > np.mean(gaps) + np.std(gaps))
                
                if n_significant_gaps > 0:
                    sorted_distances = np.sort(distances)
                    significant_gaps = gaps > np.mean(gaps) + np.std(gaps)
                    last_significant_gap = np.where(significant_gaps)[0][-1]
                    cut_height = sorted_distances[last_significant_gap + 1]
                else:
                    return self.cluster_covariance(method=method, n_clusters=3)
            
            self.cut_height = cut_height
            cluster_labels = hierarchy.fcluster(row_linkage, cut_height, 
                                              criterion='distance')
            self.n_clusters = len(set(cluster_labels))
        
        # Map clusters to original positions
        self.tfbs_clusters = {f"TFBS {label}": [] for label in set(cluster_labels)}
        for idx, cluster in enumerate(cluster_labels):
            original_position = self.cov_matrix.index[self.row_order[idx]]
            self.tfbs_clusters[f"TFBS {cluster}"].append(original_position)
        
        # Remove largest cluster (usually background)
        largest_cluster = max(self.tfbs_clusters, key=lambda k: len(self.tfbs_clusters[k]))
        self.tfbs_clusters = {k: v for k, v in self.tfbs_clusters.items() 
                            if k != largest_cluster}
        
        # Store clustering info
        self.linkage = row_linkage
        self.cluster_labels = cluster_labels
        
        return self.tfbs_clusters

    def _get_45deg_mesh(self, mat):
        """Create X and Y grids rotated -45 degrees."""
        theta = -np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

        K = len(mat) + 1
        grid1d = np.arange(0, K) - .5
        X = np.tile(np.reshape(grid1d, [K, 1]), [1, K])
        Y = np.tile(np.reshape(grid1d, [1, K]), [K, 1])
        xy = np.array([X.ravel(), Y.ravel()])

        xy_rot = R @ xy
        X_rot = xy_rot[0, :].reshape(K, K)
        Y_rot = xy_rot[1, :].reshape(K, K).T

        return X_rot, Y_rot

    def plot_pairwise_matrix(self, theta_lclc, view_window=None, threshold=None, cbar_title='Pairwise',
                             gridlines=True, xtick_spacing=1, figsize=None, save_path=None, dpi=200,
                             file_format='png'):
        """Plot pairwise matrix visualization.
        Adapted from https://github.com/jbkinney/mavenn/blob/master/mavenn/src/visualization.py
        Original authors: Tareen, A. and Kinney, J.
        
        Parameters
        ----------
        theta_lclc : np.ndarray
            Pairwise matrix parameters (shape: (L,C,L,C))
        view_window : tuple, optional
            (start, end) positions to view
        threshold : float, optional
            Threshold for matrix values
        cbar_title : str, optional
            Title for colorbar
        gridlines : bool, optional
            Whether to show gridlines
        xtick_spacing : int, optional
            Show every nth x-tick label (default: 1)
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            Path to save the figure
        dpi : int, optional
            DPI for saved figure (default: 200)
        file_format : str, optional
            Format for saved figure (default: 'png')
        """
        if threshold is not None:
            temp = theta_lclc.flatten()
            temp[(temp >= -1.*threshold) & (temp <= threshold)] = 0
            theta_lclc = temp.reshape(theta_lclc.shape)

        # Set up gridlines
        if gridlines:
            show_seplines = True
            sepline_kwargs = {'linestyle': '-',
                            'linewidth': .3,
                            'color':'lightgray'}
        else:
            show_seplines = False
            sepline_kwargs = {'linestyle': '-',
                            'linewidth': .5,
                            'color':'gray'}

        # Create figure
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

        # Get matrix dimensions
        L = theta_lclc.shape[0]
        C = theta_lclc.shape[1]
        
        # Create position grids
        ls = np.arange(L)
        cs = np.arange(C)
        l1_grid = np.tile(np.reshape(ls, (L, 1, 1, 1)), (1, C, L, C))
        c1_grid = np.tile(np.reshape(cs, (1, C, 1, 1)), (L, 1, L, C))
        l2_grid = np.tile(np.reshape(ls, (1, 1, L, 1)), (L, C, 1, C))

        # Set up pairwise matrix
        nan_ix = ~(l2_grid - l1_grid >= 1)
        values = theta_lclc.copy()
        values[nan_ix] = np.nan

        # Reshape into matrix
        mat = values.reshape((L*C, L*C))
        mat = mat[:-C, :]
        mat = mat[:, C:]
        K = (L - 1) * C

        # Get finite elements
        ix = np.isfinite(mat)
        
        # Set color limits
        clim = [np.min(mat[ix]), np.max(mat[ix])]
        ccenter = 0
        
        # Set up normalization
        if ccenter is not None:
            if (clim[0] > ccenter) or (clim[1] < ccenter):
                ccenter = 0.5 * (clim[0] + clim[1])
            norm = TwoSlopeNorm(vmin=clim[0], vcenter=ccenter, vmax=clim[1])
        else:
            norm = Normalize(vmin=clim[0], vmax=clim[1])

        # Get rotated mesh using instance method
        X_rot, Y_rot = self._get_45deg_mesh(mat)
        
        # Normalize coordinates
        half_pixel_diag = 1 / (2*C)
        pixel_side = 1 / (C * np.sqrt(2))
        X_rot = X_rot * pixel_side + half_pixel_diag
        Y_rot = Y_rot * pixel_side
        Y_rot = -Y_rot

        # Set up plot limits
        xlim_pad = 0.1
        ylim_pad = 0.1
        xlim = [-xlim_pad, L - 1 + xlim_pad]
        ylim = [-0.5 - ylim_pad, (L - 1) / 2 + ylim_pad]

        # Create heatmap
        im = ax.pcolormesh(X_rot, Y_rot, mat, cmap='seismic', norm=norm)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add gridlines if requested
        if show_seplines:
            ysep_min = -0.5 - .001 * half_pixel_diag
            ysep_max = L / 2 + .001 * half_pixel_diag
            
            for n in range(0, K+1, C):
                x = X_rot[n, :]
                y = Y_rot[n, :]
                ks = (y >= ysep_min) & (y <= ysep_max)
                ax.plot(x[ks], y[ks], **sepline_kwargs)

                x = X_rot[:, n]
                y = Y_rot[:, n]
                ks = (y >= ysep_min) & (y <= ysep_max)
                ax.plot(x[ks], y[ks], **sepline_kwargs)

        # Add triangle boundary
        boundary_kwargs = {'linestyle': '-', 'linewidth': .7, 'color':'k'}
        
        # Top edge
        top_x = X_rot[0, :]
        top_y = Y_rot[0, :]
        ax.plot(top_x, top_y, **boundary_kwargs)
        
        # Right edge
        right_x = [X_rot[0, -1], X_rot[-1, -1]]
        right_y = [Y_rot[0, -1], Y_rot[-1, -1]]
        ax.plot(right_x, right_y, **boundary_kwargs)
        
        # Bottom edge
        bottom_x = []
        bottom_y = []
        for i in range(len(X_rot) - 1):
            bottom_x.extend([X_rot[i + 1, i], X_rot[i + 1, i + 1]])
            bottom_y.extend([Y_rot[i + 1, i], Y_rot[i + 1, i + 1]])
        ax.plot(bottom_x, bottom_y, **boundary_kwargs)
        
        # Left edge completion
        last_x = [top_x[0], bottom_x[0]]
        last_y = [top_y[0], bottom_y[0]]
        ax.plot(last_x, last_y, **boundary_kwargs)

        # Set plot properties
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_yticks([])
        
        # Handle x-tick spacing
        all_ticks = np.arange(L).astype(int)
        shown_ticks = all_ticks[::xtick_spacing]
        ax.set_xticks(shown_ticks)
        if view_window:
            ax.set_xticklabels(np.arange(view_window[0], view_window[1])[::xtick_spacing])
        else:
            ax.set_xticklabels(shown_ticks)
        
        ax.set_xlabel('Nucleotide position')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(cbar_title, labelpad=8, rotation=-90)
        cb.outline.set_visible(False)
        cb.ax.tick_params(direction='in', size=20, color='white')

        # Set symmetric colorbar limits
        theta_max = max(abs(np.min(theta_lclc)), abs(np.max(theta_lclc)))
        cb.mappable.set_clim(vmin=-theta_max, vmax=theta_max)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/{cbar_title.lower()}_matrix.{file_format}", 
                    facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
            
        return fig, ax

    def plot_msm_covariance_triangular(self, view_window=None, xtick_spacing=5, show_clusters=False,
                                       figsize=None, save_path=None, dpi=200, file_format='png'):
        """
        Plot the covariance matrix.
        
        Parameters
        ----------
        view_window : tuple, optional
            (start, end) positions to view
        xtick_spacing : int, optional
            Show every nth x-tick label (default: 5)
        show_clusters : bool, optional
            Whether to show TFBS cluster rectangles (default: False)
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            Directory to save the plot
        dpi : int, optional
            DPI for saved figure (default: 200)
        file_format : str, optional
            Format for saved figure (default: 'png')
        """
        matrix = self.cov_matrix.to_numpy()
        
        if view_window:
            matrix = matrix[view_window[0]:view_window[1], 
                          view_window[0]:view_window[1]]
            
        matrix = matrix.reshape(matrix.shape[0], 1, matrix.shape[0], 1)
        
        fig, ax = self.plot_pairwise_matrix(
            matrix, 
            view_window=view_window, 
            cbar_title='Covariance',
            gridlines=False,
            save_path=None,
            xtick_spacing=xtick_spacing,
            figsize=figsize
        )
        
        if show_clusters and hasattr(self, 'tfbs_clusters'):
            for cluster, positions in self.tfbs_clusters.items():
                # Convert positions to indices in the current view
                if view_window:
                    positions = [p for p in positions 
                               if view_window[0] <= p <= view_window[1]]
                    if not positions:
                        continue
                
                # Get rectangle coordinates
                start = min(positions)
                end = max(positions)
                width = end - start + 1
                
                # Create rotated rectangle for upper triangular plot
                rect = patches.Rectangle(
                    (start, -start/2), 
                    width, 
                    width/2,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='none',
                    transform=ax.transData + 
                             plt.matplotlib.transforms.Affine2D().rotate_deg(-45)
                )
                ax.add_patch(rect)

        if save_path:
            plt.savefig(save_path + f'/msm_covariance_triangular.{file_format}', facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig, ax
    
    def plot_msm_covariance_dendrogram(self, figsize=(15, 10), leaf_rotation=90, leaf_font_size=8,
                        save_path=None, dpi=200, file_format='png'):
        """
        Plot the dendrogram from hierarchical clustering.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) in inches
        leaf_rotation : float, optional
            Rotation angle for leaf labels (default: 90)
        leaf_font_size : int, optional
            Font size for leaf labels (default: 8)
        save_path : str, optional
            Path to save figure (if None, displays plot)
        dpi : int, optional
            DPI for saved figure (default: 200)
        file_format : str, optional
            Format for saved figure (default: 'png')
        """
        if not hasattr(self, 'linkage'):
            raise ValueError("Must run cluster_covariance() before plotting dendrogram")
        
        sys.setrecursionlimit(100000)  # Fix for large dendrograms
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        
        # Plot dendrogram with enhanced styling
        with plt.rc_context({'lines.linewidth': 2}):
            hierarchy.dendrogram(
                self.linkage,
                leaf_rotation=leaf_rotation,
                leaf_font_size=leaf_font_size,
            )
        
        # Add cut height line and appropriate label
        if hasattr(self, 'cut_height'):
            plt.axhline(y=self.cut_height, color='r', linestyle='--')
            if self.cluster_method == 'n_clusters':
                label = f'Cut height: {self.cut_height:.3f}\n(n_clusters={self.n_clusters})'
            else:
                label = f'Cut height: {self.cut_height:.3f}\n(resulted in {self.n_clusters} clusters)'
            plt.legend([label], frameon=False, loc='best')
        
        # Clean up plot styling
        plt.xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Store return values
        ret_fig, ret_ax = fig, ax
        
        if save_path:
            fig.savefig(save_path + f'/msm_covariance_dendrogram.{file_format}', facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
            
        return ret_fig, ret_ax
    
    def plot_msm_covariance_square(self, view_window=None, show_clusters=True, view_linkage_space=False, 
                                   figsize=None, save_path=None, dpi=200, file_format='png'):
        """
        Plot covariance matrix in square format using seaborn heatmap.
        
        Parameters
        ----------
        view_window : tuple, optional
            (start, end) positions to view in nucleotide position space.
            Note: Disabled when view_linkage_space is True.
        show_clusters : bool, optional
            Whether to show TFBS cluster rectangles. Only available in nucleotide position space.
        view_linkage_space : bool, optional
            If True, shows matrix reordered by hierarchical clustering linkage.
            If False (default), shows matrix in original nucleotide position space.
            Note: cluster visualization and view_window are disabled in linkage space.
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            Path to save figure
        dpi : int, optional
            DPI for saved figure (default: 200)
        file_format : str, optional
            Format for saved figure (default: 'png')
        """
        # Choose matrix based on space parameter
        matrix = self.reordered_cov_matrix if view_linkage_space else self.cov_matrix
        original_indices = self.cov_matrix.index.tolist()
        
        # Handle view window only in nucleotide position space
        if view_window and not view_linkage_space:
            matrix = matrix.iloc[view_window[0]:view_window[1], 
                               view_window[0]:view_window[1]]
        elif view_window and view_linkage_space:
            print("Note: view_window is disabled in linkage space")
        
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()

        sns.heatmap(matrix, cmap='seismic', center=0, 
                    cbar_kws={'label': 'Covariance'}, ax=ax)
        
        # Only show clusters in nucleotide position space
        if show_clusters and not view_linkage_space and hasattr(self, 'tfbs_clusters'):
            for cluster, positions in self.tfbs_clusters.items():
                # Map from reordered positions back to original positions
                plot_positions = [original_indices[self.row_order.index(pos)] 
                                for pos in positions]
                
                # Filter positions based on view window
                if view_window:
                    plot_positions = [p - view_window[0] for p in plot_positions 
                                    if view_window[0] <= p <= view_window[1]]
                    if not plot_positions:
                        continue
                
                # Get rectangle coordinates
                start = min(plot_positions)
                end = max(plot_positions)
                width = end - start + 1
                height = width
                
                # Create rectangle
                rect = patches.Rectangle(
                    (start, start),  # Lower left corner
                    width,           # Width
                    height,         # Height
                    linewidth=1,
                    edgecolor='black',
                    facecolor='none'
                )
                ax.add_patch(rect)
        elif show_clusters and view_linkage_space:
            print("Note: Cluster visualization is disabled in linkage space")
        
        space_label = "Linkage Space" if view_linkage_space else "Nucleotide Position Space"
        plt.title(f"Covariance Matrix ({space_label})")
        
        # Update axis labels based on space
        ax.set_xlabel("Position" if not view_linkage_space else "Linkage Order")
        ax.set_ylabel("Position" if not view_linkage_space else "Linkage Order")
        
        # Add coordinate formatter
        '''def format_coord(x, y):
            try:
                col, row = int(x), int(y)
                if 0 <= col < matrix.shape[1] and 0 <= row < matrix.shape[0]:
                    if view_window and not view_linkage_space:
                        # Adjust coordinates to account for view window
                        display_x = col + view_window[0]
                        display_y = row + view_window[0]
                    else:
                        display_x = col
                        display_y = row
                    value = matrix.iloc[row, col]
                    return f"x={display_x}, y={display_y}, z={value:.2f}"
                return f"x={x:.2f}, y={y:.2f}"
            except (ValueError, IndexError):
                return f"x={x:.2f}, y={y:.2f}"
        
        ax.format_coord = format_coord'''
        
        # Store return values
        ret_fig, ret_ax = fig, ax
        
        if save_path:
            fig.savefig(save_path + f'/msm_covariance_square.{file_format}', facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
        
        return ret_fig, ret_ax
    
    def set_entropy_multiplier(self, entropy_multiplier):
        """Set the entropy multiplier for TFBS activity detection.
        
        This value is used to determine which clusters are considered active
        for each TFBS region based on their entropy values.
        
        Parameters
        ----------
        entropy_multiplier : float
            Multiplier for background entropy threshold. Lower values result
            in more clusters being considered active.
        """
        self.entropy_multiplier = entropy_multiplier
        
        # Update active clusters if we have access to MetaExplainer's results
        if hasattr(self.meta_explainer, 'active_clusters_by_tfbs'):
            self.active_clusters_by_tfbs = self.meta_explainer.active_clusters_by_tfbs

    def get_tfbs_positions(self, active_clusters):
        """
        Get the start and stop positions for each TFBS cluster.
        
        Parameters
        ----------
        active_clusters : dict
            Dictionary mapping TFBS labels to active clusters
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing start, stop, length, positions, and active clusters 
            for each TFBS, sorted by start position and labeled alphabetically (A, B, C, etc.)
        """
        if not hasattr(self, 'tfbs_clusters'):
            raise ValueError("Must run cluster_covariance() before getting TFBS positions")
        
        # Initialize lists to store data
        clusters = []
        starts = []
        stops = []
        lengths = []
        positions_list = []
        active_clusters_list = []
        
        # Get original indices for mapping
        original_indices = self.cov_matrix.index.tolist()
        
        for cluster, positions in self.tfbs_clusters.items():
            # Map from reordered positions back to original positions
            original_positions = [original_indices[self.row_order.index(pos)] 
                                for pos in positions]
            
            # Get start and stop positions
            start = min(original_positions)
            stop = max(original_positions)
            length = stop - start + 1
            
            clusters.append(cluster)
            starts.append(start)
            stops.append(stop)
            lengths.append(length)
            positions_list.append(sorted(original_positions))  # Store actual positions
            active_clusters_list.append(sorted(active_clusters[cluster]))
        
        # Create DataFrame with basic info
        tfbs_df = pd.DataFrame({
            'TFBS': clusters,
            'Start': starts,
            'Stop': stops,
            'Length': lengths,
            'Positions': positions_list,
            'N_Positions': [len(pos) for pos in positions_list],
            'Active_Clusters': active_clusters_list
        })
        
        # Sort and rename
        tfbs_df = tfbs_df.sort_values('Start').reset_index(drop=True)
        tfbs_df['TFBS'] = [chr(65 + i) for i in range(len(tfbs_df))]
        
        return tfbs_df
    
    def get_binding_config_matrix(self, active_clusters, mode='binary'):
        """Create a binding configuration matrix showing TFBS activity in each cluster.
        
        Parameters
        ----------
        active_clusters : dict
            Dictionary mapping TFBS labels to active clusters
        mode : str
            'binary': 0/1 for inactive/active
            'continuous': normalized activity values (1 - normalized entropy),
                         where higher values indicate more activity
        
        Returns
        -------
        pd.DataFrame
            Binding configuration matrix with clusters as rows and TFBSs as columns
        """
        if not hasattr(self, 'tfbs_clusters'):
            raise ValueError("Must run cluster_covariance() before getting binding configuration matrix")
        
        # Get cluster order from meta_explainer
        cluster_order = range(self.nC)  # Default order
        if hasattr(self.meta_explainer, 'cluster_order') and self.meta_explainer.cluster_order is not None:
            cluster_order = np.arange(self.nC)  # Use positional indices
        
        # Get TFBS positions DataFrame to use its ordering
        tfbs_df = self.get_tfbs_positions(active_clusters)
        tfbs_order = tfbs_df['TFBS'].tolist()
        
        # Initialize binding configuration matrix with ordered clusters and TFBSs
        binding_config_matrix = pd.DataFrame(0, 
                                  index=cluster_order,
                                  columns=tfbs_order)
        
        # Calculate background entropy
        null_rate = 1 - self.meta_explainer.mut_rate
        background_entropy = entropy([null_rate, (1-null_rate)/3, (1-null_rate)/3, (1-null_rate)/3], base=2)
        
        if mode == 'binary':
            # Binary mode (0/1)
            for _, row in tfbs_df.iterrows():
                tfbs = row['TFBS']
                active_cluster_indices = row['Active_Clusters']
                binding_config_matrix.iloc[active_cluster_indices, binding_config_matrix.columns.get_loc(tfbs)] = 1
        
        elif mode == 'continuous':
            # For each cluster
            for cluster in cluster_order:
                # Get the original cluster index if sorting is used
                if self.meta_explainer.sort_method and self.meta_explainer.cluster_order is not None:
                    original_cluster_idx = self.meta_explainer.cluster_order[cluster]
                else:
                    original_cluster_idx = cluster
                    
                cluster_entropy = self.meta_explainer.revels.iloc[original_cluster_idx]
                
                # For each TFBS
                for _, tfbs_row in tfbs_df.iterrows():
                    positions = tfbs_row['Positions']
                    tfbs_entropy = cluster_entropy[positions]
                    
                    # Convert entropy to activity (1 - normalized entropy)
                    mean_entropy = tfbs_entropy.mean()
                    normalized_activity = 1 - (mean_entropy / background_entropy)
                    normalized_activity = max(0, min(1, normalized_activity))  # Clip to [0,1]
                    
                    binding_config_matrix.iloc[cluster, binding_config_matrix.columns.get_loc(tfbs_row['TFBS'])] = normalized_activity
        
        else:
            raise ValueError("mode must be 'binary' or 'continuous'")
        
        return binding_config_matrix
    
    def plot_binding_config_matrix(self, active_clusters, mode='binary', orientation='vertical',
                          figsize=None, save_path=None, dpi=200, file_format='png'):
        """Plot binding configuration matrix showing TFBS activity in each cluster.
        
        Parameters
        ----------
        active_clusters : dict
            Dictionary mapping TFBS labels to active clusters
        mode : str
            'binary': dark gray/white for active/inactive
            'continuous': grayscale for activity level
        orientation : str
            'vertical': Clusters on y-axis, TFBS on x-axis (default)
            'horizontal': TFBS on y-axis, Clusters on x-axis
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            Path to save the figure. If None, displays plot.
        dpi : int, optional
            DPI for saved figure (default: 200)
        file_format : str, optional
            Format for saved figure (default: 'png')
        """
        def _add_colorbar_border(cbar_ax):
            """Add black border to all sides of colorbar."""
            for spine in cbar_ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
        
        def _get_colorbar_params(orientation):
            """Get colorbar parameters based on orientation."""
            is_vertical = orientation == 'vertical'
            return {
                'orientation': 'vertical' if is_vertical else 'horizontal',
                'location': 'right' if is_vertical else 'bottom',
                'label': 'TFBS Binding Activity',
                'fraction': 0.046,
                'aspect': 40,
                'pad': 0.04 if is_vertical else 0.08,
            }
        
        def _get_axis_labels(orientation):
            """Get axis labels based on orientation."""
            return ('TFBS', 'Cluster') if orientation == 'vertical' else ('Cluster', 'TFBS')
        
        # Get and prepare binding configuration matrix
        binding_config_matrix = self.get_binding_config_matrix(active_clusters=active_clusters, mode=mode)
        if orientation == 'horizontal':
            binding_config_matrix = binding_config_matrix.T
        
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # Plot heatmap
        if mode == 'continuous':
            cbar_ax = sns.heatmap(
                binding_config_matrix,
                cmap='Greys',
                cbar=True,
                cbar_kws=_get_colorbar_params(orientation),
                xticklabels=True,
                yticklabels=True,
                square=True,
                linewidths=1,
                linecolor='black',
                ax=ax
            ).collections[0].colorbar.ax
            _add_colorbar_border(cbar_ax)
        else:  # binary mode
            sns.heatmap(
                binding_config_matrix,
                cmap=colors.ListedColormap(['white', '#404040']),  # Use dark gray instead of black
                cbar=False,
                xticklabels=True,
                yticklabels=True,
                square=True,
                linewidths=1,
                linecolor='black',
                ax=ax
            )
            # Update legend to match new color
            legend_elements = [
                patches.Patch(facecolor='#404040', label='Active'),
                patches.Patch(facecolor='white', edgecolor='black', label='Inactive')
            ]
            legend_params = {
                'handles': legend_elements,
                'title': "TFBS Binding Activity",
                'bbox_to_anchor': (1.05, 1) if orientation == 'vertical' else (1.0, -0.8),
                'loc': 'upper left' if orientation == 'vertical' else 'lower right',
                'ncol': 1 if orientation == 'vertical' else 2
            }
            ax.legend(**legend_params)
        
        # Add border to matrix
        ax.add_patch(patches.Rectangle(
            (0, 0),
            binding_config_matrix.shape[1],
            binding_config_matrix.shape[0],
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        ))
        
        # Set labels
        xlabel, ylabel = _get_axis_labels(orientation)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('TFBS Activity Matrix')
        
        # Handle layout and display
        if save_path:
            plt.savefig(save_path + f'/binding_config_matrix_{mode}.{file_format}', 
                       facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()
            plt.close(fig)
        
        return fig, ax

    def get_binding_config_assignments(self, tfbs_positions, mode='auto', print_template=False):
        """Assign clusters to specific TFBS binding configurations based on their activity patterns.
        
        This function analyzes the continuous activity levels of TFBSs across clusters to find
        the optimal cluster for each possible TFBS binding configuration. For example, it will find:
        - Which cluster best represents TFBS A alone
        - Which cluster best represents TFBS B alone
        - Which cluster best represents the combined presence of TFBSs A and B
        - Which cluster best represents the background configuration (no TFBSs active)
        
        The scoring system works by:
        1. For each binding configuration, defining an "ideal" activity pattern where:
           - Desired TFBS(s) have maximum observed activity
           - Other TFBSs have minimum observed activity
        2. Calculating how far each cluster's activity pattern is from this ideal
        3. Selecting the cluster that minimizes this distance
        
        For example, when finding a cluster for TFBS A:
        - The ideal configuration would have maximum activity for A and minimum for others
        - Each cluster's score is based on how close it comes to this ideal
        - The cluster with the highest score (smallest distance from ideal) is selected
        
        Parameters
        ----------
        tfbs_positions : pd.DataFrame
            DataFrame from get_tfbs_positions containing TFBS information.
            Must have columns: 'TFBS', 'Start', 'Stop', 'Positions', 'Active_Clusters'
        mode : str, optional
            'auto' : Automatically assign clusters based on activity patterns
            'template' : Print a template for manual assignment
        print_template : bool, optional
            If True and mode='template', prints a formatted template showing all possible
            TFBS combinations and their current cluster assignments

        Returns
        -------
        dict or None
            If mode='auto': Dictionary mapping TFBS binding configurations to cluster indices.
            For example:
            {
                (): 5,           # Background configuration (no TFBSs)
                ('A',): 1,       # TFBS A alone
                ('B',): 3,       # TFBS B alone
                ('A', 'B'): 7,   # Interaction of TFBSs A and B
                ...
            }
            If mode='template': None, but prints template for manual assignment

        Notes
        -----
        The function internally uses the continuous binding configuration matrix (normalized entropy-based
        activity levels) to make assignments, ensuring consistent scoring across all
        binding configurations. This means:
        - Activity levels are normalized relative to background entropy
        - Higher values (closer to 1) indicate stronger TFBS activity
        - Lower values (closer to 0) indicate weaker or no TFBS activity
        
        The scoring system prioritizes finding clusters that:
        1. Have high activity for the desired TFBS(s)
        2. Have low activity for other TFBSs
        3. Show balanced activity when multiple TFBSs are desired
        """
        # Get continuous binding configuration matrix internally
        binding_config_matrix = self.get_binding_config_matrix(
            active_clusters=self.meta_explainer.active_clusters_by_tfbs,
            mode='continuous'
        )
        
        # Get list of TFBS IDs
        tfbs_ids = tfbs_positions['TFBS'].tolist()
        n_tfbs = len(tfbs_ids)
        
        # Generate all possible combinations (including empty set)
        all_combinations = [()]  # Start with empty set
        for r in range(1, n_tfbs + 1):
            all_combinations.extend(list(itertools.combinations(tfbs_ids, r)))
        
        if mode == 'template':
            # Print template for manual assignment
            print("\nBinding Configuration Assignment Template:")
            print("----------------------------------------")
            print("# Copy this template and replace None with cluster indices")
            print("binding_config_assignments = {")
            
            # Print each combination with a descriptive comment
            for combo in all_combinations:
                if len(combo) == 0:
                    comment = "# Cluster for background configuration (no TFBSs active)"
                elif len(combo) == n_tfbs:
                    comment = "# Cluster for all TFBSs active (highest minimum activity)"
                elif len(combo) == 1:
                    comment = f"# Cluster for TFBS {combo[0]} alone (highest activity for {combo[0]}, lowest for others)"
                else:
                    comment = f"# Cluster for interaction between TFBSs {', '.join(combo)} (highest activity for these, lowest for others)"
                print(f"    {combo}: None,  {comment}")
            
            print("}")
            return None
            
        elif mode == 'auto':
            # Initialize assignments dictionary
            assignments = {}
            
            # For each combination
            for combo in all_combinations:
                if len(combo) == 0:
                    # For background configuration: find cluster with lowest maximum activity
                    max_activities = binding_config_matrix.max(axis=1)
                    assignments[combo] = max_activities.idxmin()
                    
                elif len(combo) == n_tfbs:
                    # For all-TFBSs configuration: find cluster with highest minimum activity
                    min_activities = binding_config_matrix.min(axis=1)
                    assignments[combo] = min_activities.idxmax()
                    
                else:
                    # Get the columns for this combination and others
                    combo_cols = list(combo)
                    other_cols = [col for col in binding_config_matrix.columns if col not in combo_cols]
                    
                    # Define ideal configuration using actual observed values
                    ideal_config = pd.Series(0, index=binding_config_matrix.columns)
                    # For desired TFBSs, use maximum observed value
                    ideal_config[combo_cols] = binding_config_matrix[combo_cols].max().max()
                    # For other TFBSs, use minimum observed value
                    ideal_config[other_cols] = binding_config_matrix[other_cols].min().min()
                    
                    # Calculate distance from ideal configuration for each cluster
                    # Using negative distance so higher scores are better
                    scores = -(binding_config_matrix - ideal_config).abs().mean(axis=1)
                    
                    # Assign the cluster with highest score (smallest distance from ideal)
                    assignments[combo] = scores.idxmax()
            
            return assignments
            
        else:
            raise ValueError("mode must be either 'auto' or 'template'")

    def get_additive_params(self, tfbs_positions, specific_clusters=None, zero_out_inactive=False, separate_background=True):
        """Extract additive parameters for each TFBS by cropping from meta-attribution maps.
        
        Parameters
        ----------
        tfbs_positions : pd.DataFrame
            DataFrame containing TFBS information (from get_tfbs_positions)
        specific_clusters : list of int, optional
            List of one cluster per TFBS to use for cropping (e.g., [5, 17, 20, 23] for TFBSs A, B, C, D).
            If None, uses the average of all active clusters for each TFBS.
        zero_out_inactive : bool, optional
            Controls how to handle positions within the cropped region:
            - False (default): Return the full cropped region (start to stop) with all positions
            - True: Return the full cropped region (start to stop), with inactive positions set to zero
        separate_background : bool, optional
            Whether to use background-separated cluster maps (default: True).
            If True, uses meta_explainer.cluster_maps_no_bg if available.
            If False or if background-separated maps aren't available, uses regular cluster maps.
            
        Returns
        -------
        dict
            Dictionary mapping TFBS IDs (A, B, C, etc.) to their 4xL parameter matrices.
            For each TFBS, the matrix is cropped from either:
            - The cluster-averaged attribution map for the specified cluster, or
            - The average of cluster-averaged attribution maps from all active clusters
            The matrix always spans the full region (start to stop), with L = stop - start + 1.
            If zero_out_inactive=True, positions not in the TFBS's Positions list are set to zero.
        """
        if not hasattr(self, 'tfbs_clusters'):
            raise ValueError("Must run cluster_covariance() before getting additive parameters")
            
        # Validate specific_clusters if provided
        if specific_clusters is not None:
            if len(specific_clusters) != len(tfbs_positions):
                raise ValueError(f"Length of specific_clusters ({len(specific_clusters)}) must match number of TFBSs ({len(tfbs_positions)})")
        
        # Get cluster maps from meta_explainer
        if separate_background and hasattr(self.meta_explainer, 'cluster_maps_no_bg'):
            print("\nUsing background-separated maps")
            cluster_maps = self.meta_explainer.cluster_maps_no_bg
            # Debug: Compare raw and background-separated maps for first cluster
            if hasattr(self.meta_explainer, 'cluster_maps'):
                raw_maps = self.meta_explainer.cluster_maps
        elif hasattr(self.meta_explainer, 'cluster_maps'):
            print("\nUsing raw maps (background-separated maps not available)")
            cluster_maps = self.meta_explainer.cluster_maps
        else:
            print("\nGenerating new maps with background_separation=", separate_background)
            # Generate logos if maps aren't available
            self.meta_explainer.generate_logos(
                logo_type='average',
                background_separation=separate_background
            )
            cluster_maps = (self.meta_explainer.cluster_maps_no_bg if separate_background 
                          else self.meta_explainer.cluster_maps)
        
        # Initialize dictionary to store parameters for each TFBS
        params_dict = {}
        
        # Process each TFBS
        for i, (_, tfbs_row) in enumerate(tfbs_positions.iterrows()):
            tfbs_id = tfbs_row['TFBS']
            active_positions = tfbs_row['Positions']  # List of active positions for this TFBS
            start_pos = tfbs_row['Start']
            stop_pos = tfbs_row['Stop']
            
            # Get the cluster-averaged attribution map to crop from
            if specific_clusters is not None:
                # Use the specified cluster's averaged map for this TFBS
                cluster = specific_clusters[i]
                attribution_map = cluster_maps[cluster]  # Already averaged
            else:
                # Average the cluster-averaged maps from all active clusters
                active_clusters = tfbs_row['Active_Clusters']
                attribution_map = np.mean(cluster_maps[active_clusters], axis=0)  # Average over cluster maps
            
            # Create matrix spanning the full cropped region
            length = stop_pos - start_pos + 1
            params = np.zeros((length, 4))  # Changed from (4, length) to (length, 4)
            
            # Fill in values for all positions in the region
            for pos in range(start_pos, stop_pos + 1):
                rel_pos = pos - start_pos  # Position relative to start of cropped region
                if pos in active_positions:
                    params[rel_pos] = attribution_map[pos]
                elif zero_out_inactive:
                    # Position is already zero from initialization
                    pass
                else:
                    params[rel_pos] = attribution_map[pos]
                
            params_dict[tfbs_id] = params  # (L_cropped,4) matrix
            
        return params_dict

    def get_epistatic_params(self, tfbs_positions, binding_config_assignments=None):
        """Calculate epistatic interactions between TFBSs using Möbius inversion.
        
        For each combination of TFBSs, calculates the interaction using the inclusion-exclusion principle.
        For example, for a 3-way interaction ABC:
        I_ABC = y_ABC - y_AB - y_AC - y_BC + y_A + y_B + y_C - y_bg
        
        Parameters
        ----------
        tfbs_positions : pd.DataFrame
            DataFrame containing TFBS information (from get_tfbs_positions)
        binding_config_assignments : dict, optional
            Dictionary mapping TFBS binding configurations to cluster indices.
            If None, will use get_binding_config_assignments() with mode='auto' to get assignments.
            
        Returns
        -------
        dict
            Dictionary mapping TFBS combinations to their epistatic interaction values.
            Keys are tuples of TFBS IDs (e.g., ('A', 'B') for 2-way, ('A', 'B', 'C') for 3-way).
            Values are the calculated interaction terms using Möbius inversion.
            
        Notes
        -----
        The epistatic interactions are calculated using Möbius inversion, where each term's
        coefficient is (-1)^k for a subset of size k. This ensures that:
        
        1. The interaction term captures the deviation from additivity
        2. Higher-order interactions are properly decomposed into their constituent terms
        3. The background configuration (empty set) is properly accounted for
        
        For example:
        - 2-way: I_AB = y_AB - y_A - y_B + y₀
        - 3-way: I_ABC = y_ABC - y_AB - y_AC - y_BC + y_A + y_B + y_C - y₀
        - 4-way: I_ABCD = y_ABCD - y_ABC - y_ABD - y_ACD - y_BCD + 
                       y_AB + y_AC + y_AD + y_BC + y_BD + y_CD - 
                       y_A - y_B - y_C - y_D + y₀
        
        A positive interaction indicates synergy (combined effect > sum of individual effects),
        while a negative interaction indicates antagonism (combined effect < sum of individual effects).

        Saving and Loading
        -----------------
        If the epistatic parameters are saved as a NumPy file (.npy), these parameters can be loaded as follows:

        ```python
        import numpy as np

        # Load the saved parameters
        epistatic_params = np.load('path/to/identified_parameters/epistatic_params.npy', 
                                 allow_pickle=True).item()

        # The loaded data will be a dictionary where:
        # - Keys are tuples of TFBS IDs (e.g., ('A', 'B') for pairwise interactions)
        # - Values are the interaction terms (float values)
        
        # Example usage:
        # Get a pairwise interaction
        ab_interaction = epistatic_params[('A', 'B')]
        
        # Get a higher-order interaction
        abc_interaction = epistatic_params[('A', 'B', 'C')]
        ```

        Note: The allow_pickle=True parameter is required because the data is stored as a dictionary,
        and .item() is needed to convert the NumPy array back into a dictionary format.
        """
        if not hasattr(self, 'tfbs_clusters'):
            raise ValueError("Must run cluster_covariance() before getting epistatic parameters")
        
        # Get binding configuration assignments if not provided
        if binding_config_assignments is None:
            binding_config_assignments = self.get_binding_config_assignments(tfbs_positions, mode='auto')
        
        # Get list of TFBS IDs
        tfbs_ids = tfbs_positions['TFBS'].tolist()
        n_tfbs = len(tfbs_ids)
        
        # Get median DNN scores for each cluster from the actual sequences
        mave_df = self.meta_explainer.mave
        
        # Calculate cluster medians using original cluster indices
        cluster_medians = {}
        for cluster in np.unique(mave_df['Cluster']):
            cluster_data = mave_df[mave_df['Cluster'] == cluster]['DNN']
            cluster_medians[cluster] = cluster_data.median()
        
        # Initialize dictionary to store epistatic interactions
        epistatic_params = {}
        
        # Generate all possible combinations (excluding empty set and single TFBSs)
        all_combinations = []
        for r in range(2, n_tfbs + 1):  # Start from 2-way interactions
            all_combinations.extend(list(itertools.combinations(tfbs_ids, r)))
        
        # Calculate interactions for each combination
        for combo in all_combinations:
            # Get all possible subsets of the combination
            subsets = []
            for r in range(len(combo) + 1):
                subsets.extend(list(itertools.combinations(combo, r)))
            
            # Calculate interaction using inclusion-exclusion principle
            interaction = 0
            
            for subset in subsets:
                # Get the cluster for this subset
                sorted_cluster = binding_config_assignments.get(subset)
                if sorted_cluster is None:
                    raise ValueError(f"No cluster assigned for TFBS combination {subset}")
                
                # Convert sorted cluster index to original cluster index if needed
                if self.meta_explainer.cluster_order is not None:
                    original_cluster = self.meta_explainer.cluster_order[sorted_cluster]
                else:
                    original_cluster = sorted_cluster
                
                # Get median DNN score for this subset
                y = cluster_medians[original_cluster]
                
                # Add or subtract based on subset size parity
                # Even-sized subsets (including empty set) get +1, odd-sized get -1
                sign = 1 if len(subset) % 2 == 0 else -1
                term = sign * y
                interaction += term
            
            # Store the interaction
            epistatic_params[combo] = interaction
        
        return epistatic_params

    def plot_epistatic_interactions(self, epistatic_params, tfbs_positions=None, 
                                  pairwise_only=False, annotate=True, cmap='RdBu_r',
                                  figsize=(10, 8), save_path=None, dpi=200, file_format='png'):
        """Plot epistatic interactions between TFBSs.
        
        Creates two visualizations:
        1. A lower triangular heatmap showing pairwise interactions (excluding diagonal)
        2. A bar plot showing higher-order interactions (if any exist)
        
        Parameters
        ----------
        epistatic_params : dict
            Dictionary mapping TFBS combinations to their interaction values
        tfbs_positions : pandas.DataFrame, optional
            DataFrame containing TFBS positions, used for consistent ordering
        pairwise_only : bool, default=False
            If True, only plot pairwise interactions
        annotate : bool, default=True
            Whether to show interaction values on the heatmap
        cmap : str, default='RdBu_r'
            Colormap for the heatmap
        figsize : tuple, default=(10, 8)
            Figure size for the heatmap
        save_path : str, optional
            Directory to save the plots
        dpi : int, default=200
            DPI for saved figures
        file_format : str, default='png'
            Format for saved figures
            
        Returns
        -------
        tuple
            (fig_heatmap, ax_heatmap) if pairwise_only=True
            ((fig_heatmap, ax_heatmap), (fig_bar, ax_bar)) if pairwise_only=False
        """
        # Get TFBS IDs in consistent order
        if tfbs_positions is not None:
            tfbs_ids = sorted(tfbs_positions['TFBS'])
        else:
            # Extract unique TFBS IDs from epistatic_params
            tfbs_ids = sorted(set(tfbs for combo in epistatic_params.keys() for tfbs in combo))
        
        # Create pairwise interaction matrix
        n = len(tfbs_ids)
        interaction_matrix = np.zeros((n, n))
        
        # Only process pairwise interactions for the heatmap
        for combo, value in epistatic_params.items():
            if len(combo) == 2:  # Only process pairs
                i = tfbs_ids.index(combo[0])
                j = tfbs_ids.index(combo[1])
                interaction_matrix[i, j] = value
                interaction_matrix[j, i] = value  # Make symmetric
        
        # Create mask for upper triangular and diagonal
        mask = np.triu(np.ones_like(interaction_matrix, dtype=bool))
        
        # Plot pairwise interactions
        fig_heatmap, ax_heatmap = plt.subplots(figsize=figsize)
        
        # Find the maximum absolute value for symmetric colormap
        vmax = np.max(np.abs(interaction_matrix))
        vmin = -vmax
        
        # Create heatmap with centered colormap, masking upper triangle and diagonal
        sns.heatmap(interaction_matrix, 
                   xticklabels=tfbs_ids,
                   yticklabels=tfbs_ids,
                   cmap=cmap,
                   center=0,  # Center colormap at 0
                   vmin=vmin,  # Set symmetric limits
                   vmax=vmax,
                   annot=annotate,
                   fmt='.2f',
                   square=True,
                   mask=mask,  # Mask upper triangle and diagonal
                   ax=ax_heatmap)
        
        ax_heatmap.set_title('Pairwise Epistatic Interactions')
        
        if save_path:
            fig_heatmap.savefig(os.path.join(save_path, 'epistatic_interactions_pairwise.png'),
                              dpi=dpi, bbox_inches='tight', format=file_format)
        
        if pairwise_only:
            return fig_heatmap, ax_heatmap
            
        # Check if there are any higher-order interactions
        higher_order = {combo: value for combo, value in epistatic_params.items() 
                       if len(combo) > 2}
        
        if not higher_order:
            plt.close(fig_heatmap)
            raise ValueError("No higher-order interactions found")
            
        # Plot higher-order interactions
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        
        # Sort interactions by order first
        sorted_interactions = sorted(higher_order.items(), 
                                  key=lambda x: len(x[0]))  # Sort by interaction order
        
        combos = [''.join(combo) for combo, _ in sorted_interactions]
        values = [value for _, value in sorted_interactions]
        
        # Create bar plot with color based on sign
        bars = ax_bar.bar(range(len(combos)), values)
        for bar, value in zip(bars, values):
            bar.set_color('red' if value < 0 else 'blue')
            
        ax_bar.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax_bar.set_title('Higher-Order Epistatic Interactions')
        ax_bar.set_xticks(range(len(combos)))
        ax_bar.set_xticklabels(combos, rotation=45, ha='right')
        
        if save_path:
            fig_bar.savefig(os.path.join(save_path, 'epistatic_interactions_higher_order.png'),
                          dpi=dpi, bbox_inches='tight', format=file_format)
            
        return (fig_heatmap, ax_heatmap), (fig_bar, ax_bar)