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

class Identifier:
    """Class for identifying and analyzing patterns in MSM data."""
    
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
        
    def cluster_covariance(self, method='average', n_clusters=None, cut_height=None):
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

    def plot_pairwise_matrix(self, theta_lclc, view_window=None, 
                            threshold=None, save_dir=None, cbar_title='Pairwise', 
                            gridlines=True, xtick_spacing=1):
        """Plot pairwise matrix visualization.
        Adapted from https://github.com/jbkinney/mavenn/blob/master/mavenn/src/visualization.py
        Original authors: Tareen, A. and Kinney, J.
        
        Parameters
        ----------
        theta_lclc : np.ndarray
            Pairwise matrix parameters (shape: (L,C,L,C))
        view_window : tuple, optional
            (start, end) positions to view
        xtick_spacing : int, optional
            Show every nth x-tick label (default: 1)
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
        fig, ax = plt.subplots(figsize=[10,5])

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
        
        if save_dir is not None:
            plt.savefig(f"{save_dir}/{cbar_title.lower()}_matrix.pdf", 
                    facecolor='w', dpi=600)
            plt.close()
            
        return fig, ax

    def plot_covariance_triangular(self, view_window=None, save_dir=None, xtick_spacing=5, 
                       show_clusters=False):
        """
        Plot the covariance matrix.
        
        Parameters
        ----------
        view_window : tuple, optional
            (start, end) positions to view
        save_dir : str, optional
            Directory to save the plot
        xtick_spacing : int, optional
            Show every nth x-tick label (default: 5)
        show_clusters : bool, optional
            Whether to show TFBS cluster rectangles (default: False)
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
            save_dir=save_dir,
            xtick_spacing=xtick_spacing
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
        
        return fig, ax
    
    def plot_dendrogram(self, figsize=(15, 10), leaf_rotation=90, 
                    leaf_font_size=8, save_path=None, dpi=200):
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
        """
        if not hasattr(self, 'linkage'):
            raise ValueError("Must run cluster_covariance() before plotting dendrogram")
        
        sys.setrecursionlimit(100000)  # Fix for large dendrograms
        
        plt.figure(figsize=figsize)
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
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if save_path:
            plt.savefig(save_path, facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
            return None
        else:
            return plt.gcf(), ax
    
    def plot_covariance_square(self, view_window=None, show_clusters=True, view_linkage_space=False):
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
        
        fig, ax = plt.subplots(figsize=(10, 8))
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
        def format_coord(x, y):
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
        
        ax.format_coord = format_coord
        
        return fig, ax
    
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
    
    def get_state_matrix(self, active_clusters, mode='binary'):
        """Create a state matrix showing TFBS activity in each cluster.
        
        Parameters
        ----------
        active_clusters : dict
            Dictionary mapping TFBS labels to active clusters
        mode : str
            'binary': 0/1 for inactive/active
            'continuous': raw entropy values from MSM for active positions,
                         0 for inactive positions
        
        Returns
        -------
        pd.DataFrame
            State matrix with clusters as rows and TFBSs as columns
        """
        if not hasattr(self, 'tfbs_clusters'):
            raise ValueError("Must run cluster_covariance() before getting state matrix")
        
        # Get cluster order from meta_explainer
        cluster_order = range(self.nC)  # Default order
        if hasattr(self.meta_explainer, 'cluster_order') and self.meta_explainer.cluster_order is not None:
            cluster_order = np.arange(self.nC)  # Use positional indices
        
        # Get TFBS order from active_clusters keys
        tfbs_order = list(active_clusters.keys())
        
        # Initialize state matrix with ordered clusters and TFBSs
        state_matrix = pd.DataFrame(0, 
                                  index=cluster_order,
                                  columns=tfbs_order)
        
        if mode == 'binary':
            # Binary mode (0/1)
            for tfbs, active_cluster_indices in active_clusters.items():
                # active_cluster_indices are already in the sorted order from meta_explainer
                state_matrix.iloc[active_cluster_indices, state_matrix.columns.get_loc(tfbs)] = 1
        
        elif mode == 'continuous':
            # Get MSM data
            msm = self.meta_explainer.msm
            
            # For each TFBS
            for tfbs, active_cluster_indices in active_clusters.items():
                positions = self.tfbs_clusters[tfbs]
                
                # For each cluster
                for cluster_idx in cluster_order:
                    if cluster_idx in active_cluster_indices:
                        # Get mean entropy for this TFBS in this cluster
                        cluster_data = msm[(msm['Cluster'] == cluster_idx) & 
                                        (msm['Position'].isin(positions))]
                        mean_entropy = cluster_data['Entropy'].mean()
                        state_matrix.iloc[cluster_idx, state_matrix.columns.get_loc(tfbs)] = mean_entropy
        
        else:
            raise ValueError("mode must be 'binary' or 'continuous'")
        
        return state_matrix
    
    def plot_state_matrix(self, active_clusters, mode='binary', orientation='vertical', save_path=None):
        """
        Plot state matrix showing TFBS activity in each cluster.
        
        Parameters
        ----------
        active_clusters : dict
            Dictionary mapping TFBS labels to active clusters
        mode : str
            'binary': black/white for active/inactive
            'continuous': grayscale for activity level
        orientation : str
            'vertical': Clusters on y-axis, TFBS on x-axis (default)
            'horizontal': TFBS on y-axis, Clusters on x-axis
        save_path : str, optional
            Path to save the figure. If None, displays plot.
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
                'label': 'TFBS State Activity',
                'fraction': 0.046,
                'aspect': 40,
                'pad': 0.04 if is_vertical else 0.08,
            }
        
        def _get_axis_labels(orientation):
            """Get axis labels based on orientation."""
            return ('TFBS', 'Cluster') if orientation == 'vertical' else ('Cluster', 'TFBS')
        
        # Get and prepare state matrix
        state_matrix = self.get_state_matrix(active_clusters=active_clusters, mode=mode)
        if orientation == 'horizontal':
            state_matrix = state_matrix.T
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot heatmap
        if mode == 'continuous':
            cbar_ax = sns.heatmap(
                state_matrix,
                cmap='Greys',
                cbar=True,
                cbar_kws=_get_colorbar_params(orientation),
                xticklabels=True,
                yticklabels=True,
                square=True,
                linewidths=1,
                linecolor='black'
            ).collections[0].colorbar.ax
            _add_colorbar_border(cbar_ax)
        else:  # binary mode
            sns.heatmap(
                state_matrix,
                cmap='binary',
                cbar=False,
                xticklabels=True,
                yticklabels=True,
                square=True,
                linewidths=1,
                linecolor='black'
            )
            # Add legend with orientation-dependent position
            legend_elements = [
                patches.Patch(color='black', label='Active'),
                patches.Patch(facecolor='white', edgecolor='black', label='Inactive')
            ]
            legend_params = {
                'handles': legend_elements,
                'title': "TFBS State Activity",
                'bbox_to_anchor': (1.05, 1) if orientation == 'vertical' else (1.0, -0.8),
                'loc': 'upper left' if orientation == 'vertical' else 'lower right',
                'ncol': 1 if orientation == 'vertical' else 2
            }
            plt.legend(**legend_params)
        
        # Add border to matrix
        ax.add_patch(patches.Rectangle(
            (0, 0),
            state_matrix.shape[1],
            state_matrix.shape[0],
            linewidth=2,
            edgecolor='black',
            facecolor='none'
        ))
        
        # Set labels
        xlabel, ylabel = _get_axis_labels(orientation)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('TFBS Activity Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, facecolor='w', dpi=600, bbox_inches='tight')
            plt.close()
            return None
        
        return fig, ax