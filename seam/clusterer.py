import os
import sys
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

def _check_umap_available(gpu=False):
    """Check if UMAP is available, optionally checking for GPU support.
    
    Args:
        gpu: Whether to check for GPU-accelerated UMAP (default: False)
    
    Returns:
        UMAP class from either cuml (if gpu=True) or umap-learn
    """
    if gpu:
        try:
            from cuml.manifold import UMAP
            return UMAP
        except ImportError:
            print("cuml not available. Falling back to CPU implementation.")
            print("For GPU acceleration, install cuml via conda:")
            print("conda install -c rapidsai -c conda-forge -c nvidia cuml cuda-version=11.8")
            gpu = False
    
    if not gpu:
        try:
            import umap
            return umap.UMAP
        except ImportError:
            raise ImportError(
                "UMAP is required for this functionality. "
                "Install it with: pip install umap-learn"
            )

def _check_phate_available():
    try:
        import phate
    except ImportError:
        raise ImportError(
            "PHATE is required for this functionality. "
            "Install it with: pip install phate"
        )
    return phate

def _check_tsne_available(gpu=False):
    """Check if t-SNE is available, optionally checking for GPU support.
    
    Args:
        gpu: Whether to check for GPU-accelerated t-SNE (default: False)
    
    Returns:
        TSNE class from either cuml.manifold (if gpu=True) or openTSNE.sklearn
    """
    if gpu:
        try:
            from cuml.manifold import TSNE
            return TSNE
        except ImportError:
            print("cuml not available. Falling back to CPU implementation.")
            print("For GPU acceleration, install cuml via conda:")
            print("conda install -c rapidsai -c conda-forge -c nvidia cuml cuda-version=11.8")
            gpu = False
    
    if not gpu:
        try:
            from openTSNE.sklearn import TSNE
            return TSNE
        except ImportError:
            raise ImportError(
                "t-SNE is required for this functionality. "
                "Install it with: pip install openTSNE"
            )

def _check_sklearn_available():
    try:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError(
            "scikit-learn is required for this functionality. "
            "Install it with: pip install scikit-learn"
        )
    return KMeans, DBSCAN, PCA

def _check_cuml_pca_available():
    """Check if cuML PCA is available for GPU acceleration.
    
    Returns:
        PCA class from cuml.decomposition if available, None otherwise
    """
    try:
        from cuml.decomposition import PCA as cuPCA
        return cuPCA
    except ImportError:
        return None

def _check_kmeanstf_available():
    """Check if KMeansTF is available for GPU acceleration.
    
    Returns:
        KMeansTF class from kmeanstf if available, None otherwise
    """
    try:
        from kmeanstf import KMeansTF
        return KMeansTF
    except ImportError:
        return None

class Clusterer:
    """
    Clusterer: A unified interface for embedding and clustering attribution maps
    
    This implementation provides implementations of common embedding 
    and clustering methods for attribution maps:
    
    Embedding Methods:
    - UMAP (requires umap-learn)
    - PHATE (requires phate)
    - t-SNE (requires openTSNE)
    - PCA (GPU-accelerated with cuML, CPU fallback with scikit-learn)
    - Diffusion Maps (not yet implemented)

    Clustering Methods:
    - Hierarchical (GPU-optimized available)
    - K-means (GPU-accelerated with kmeanstf, CPU fallback with scikit-learn)
    - DBSCAN (requires scikit-learn)
    
    Requirements:
    - numpy
    - scipy
    - scikit-learn (for PCA, K-means, DBSCAN)
    
    Optional Requirements:
    - tensorflow (for GPU-accelerated hierarchical clustering)
    - cuml (for GPU-accelerated PCA)
    - kmeanstf (for GPU-accelerated K-means clustering)
    - umap-learn (for UMAP)
    - phate (for PHATE)
    - openTSNE (for t-SNE)

    Additional Requirements:
    - scikit-learn (for clustering)
    - matplotlib (for visualization)
    
    Example usage:
        # Initialize clusterer with attribution maps
        clusterer = Clusterer(
            maps,
            method='umap',
            n_components=2
        )
        
        # Compute embedding
        embedding = clusterer.embed()
        
        # For K-means or DBSCAN:
        clusters = clusterer.cluster(embedding, method='kmeans', n_clusters=10)
        
        # For hierarchical clustering:
        linkage = clusterer.cluster(method='hierarchical')
        # Then get cluster labels using different criteria:
        labels = clusterer.get_cluster_labels(linkage, criterion='distance', max_distance=8)
        # or
        labels, cut_level = clusterer.get_cluster_labels(linkage, criterion='maxclust', n_clusters=100)
    """
    
    SUPPORTED_METHODS = {'umap', 'phate', 'tsne', 'pca', 'diffmap'}
    SUPPORTED_CLUSTERERS = {'hierarchical', 'kmeans', 'dbscan'}

    def __init__(self, attribution_maps, method='umap', gpu=True):
        """Initialize the Clusterer.
        
        Args:
            attribution_maps: numpy array of shape (N, L, A) containing attribution maps
            method: Embedding method (default: 'umap')
            gpu: Whether to use GPU acceleration when available (default: True)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
        
        self.embedding = None
        self.cluster_labels = None
        self.maps = attribution_maps
        self.method = method
        self.gpu = gpu
        
        # Reshape maps if needed
        if len(self.maps.shape) == 3:
            N, L, A = self.maps.shape
            self.maps = self.maps.reshape((N, L*A))
            
    def embed(self, **kwargs):
        """Compute embedding using specified method.
        
        Args:
            **kwargs: Method-specific parameters. Can be passed directly or as a 'kwargs' dictionary.
            
        Returns:
            numpy.ndarray: Embedded coordinates
        """
        t0 = time.time()
        
        # Handle case where parameters are passed in a 'kwargs' dictionary
        if 'kwargs' in kwargs:
            params = kwargs['kwargs']
        else:
            params = kwargs
            
        if self.method == 'umap':
            embedding = self._embed_umap(**params)
        elif self.method == 'phate':
            embedding = self._embed_phate(**params)
        elif self.method == 'tsne':
            embedding = self._embed_tsne(**params)
        elif self.method == 'pca':
            embedding = self._embed_pca(**params)
            
        print(f'Embedding time: {time.time() - t0:.2f}s')
        return embedding
    
    def _embed_umap(self, **kwargs):
        """Compute UMAP embedding with optional GPU acceleration."""
        UMAP = _check_umap_available(self.gpu)
        print(f"Using {'GPU' if self.gpu else 'CPU'} UMAP")
        fit = UMAP(**kwargs)
        return fit.fit_transform(self.maps)
    
    def _embed_phate(self, **kwargs):
        """Compute PHATE embedding."""
        phate = _check_phate_available()
        phate_op = phate.PHATE(**kwargs)
        return phate_op.fit_transform(self.maps)
    
    def _embed_tsne(self, **kwargs):
        """Compute t-SNE embedding with optional GPU acceleration.
        
        Args:
            **kwargs: Additional parameters for t-SNE
                For GPU (cuML) version:
                    method: str, 'fft', 'barnes_hut', or 'exact' (default: 'fft')
                    learning_rate_method: str, 'adaptive', 'none', or None (default: 'adaptive')
                    n_neighbors: int (default: 90)
                    perplexity_max_iter: int (default: 100)
                    exaggeration_iter: int (default: 250)
                    pre_momentum: float (default: 0.5)
                    post_momentum: float (default: 0.8)
                    square_distances: bool (default: True)
                For CPU (openTSNE) version:
                    perplexity: float (default: 30)
                    learning_rate: float (default: 'auto')
                    early_exaggeration: float (default: 'auto')
                    n_iter: int (default: 500)
        """
        TSNE = _check_tsne_available(self.gpu)
        print(f"Using {'GPU' if self.gpu else 'CPU'} t-SNE")
        
        # Set default parameters based on GPU/CPU version
        if self.gpu:
            # cuML defaults
            default_params = {
                'n_components': 2,
                'perplexity': 30.0,
                'early_exaggeration': 12.0,
                'late_exaggeration': 1.0,
                'learning_rate': 200.0,
                'n_iter': 1000,
                'method': 'fft',
                'learning_rate_method': 'adaptive',
                'n_neighbors': 90,
                'perplexity_max_iter': 100,
                'exaggeration_iter': 250,
                'pre_momentum': 0.5,
                'post_momentum': 0.8,
                'square_distances': True
            }
        else:
            # openTSNE defaults
            default_params = {
                'n_components': 2,
                'perplexity': 30,
                'learning_rate': 'auto',
                'early_exaggeration': 'auto',
                'n_iter': 500
            }
        
        # Update defaults with user-provided parameters
        params = {**default_params, **kwargs}
        
        tsne = TSNE(**params)
        return tsne.fit_transform(self.maps)
    
    def _embed_pca(self, plot_eigenvalues=False, view_dims=None, xtick_spacing=5, figsize=None, save_path=None, dpi=200, file_format='png', **kwargs):
        """Compute PCA embedding.
        
        Args:
            plot_eigenvalues: If True, plot the eigenvalue spectrum (default: False)
            view_dims: Number of dimensions to show in eigenvalue plot (default: None, shows all)
            xtick_spacing: int, default=5
                Show x-axis labels every nth position. Set to 1 to show all positions.
            figsize: Figure size (width, height) in inches (default: None, uses matplotlib default)
            save_path: Path to save figure (if None, displays plot)
            dpi: DPI for saved figure (default: 200)
            file_format: Format for saved figure (default: 'png'). Common formats: 'png', 'pdf', 'svg', 'eps'
            **kwargs: Additional arguments passed to sklearn.decomposition.PCA
        
        Returns:
            numpy.ndarray: PCA embedding
        """
        # Try GPU PCA first if gpu=True
        if self.gpu:
            cuPCA = _check_cuml_pca_available()
            if cuPCA is not None:
                print("Using GPU-accelerated PCA (cuML)")
                try:
                    pca = cuPCA(**kwargs)
                    # Use separate fit and transform steps as per cuML documentation
                    pca.fit(self.maps.astype(np.float32))
                    embedding = pca.transform(self.maps.astype(np.float32))
                    
                    return embedding
                except Exception as e:
                    print(f"GPU PCA failed: {e}")
                    print("Falling back to CPU PCA.")
                    # Continue to CPU implementation below
            else:
                print("cuML not available. Falling back to CPU PCA.")
                print("For GPU acceleration, install cuml via:")
                print("pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com")
        
        # CPU PCA (fallback or when gpu=False)
        _, _, PCA = _check_sklearn_available()
        pca = PCA(**kwargs)
        embedding = pca.fit_transform(self.maps)
        
        if plot_eigenvalues:
            if figsize is not None:
                plt.figure(figsize=figsize)
            else:
                plt.figure(figsize=(10, 4))
            
            # Get number of dimensions to plot
            n_dims = view_dims if view_dims is not None else len(pca.explained_variance_ratio_)
            
            # Create x-axis values (1-based indexing for PCs)
            x = np.arange(1, n_dims + 1)
            
            # Plot individual explained variance ratios
            plt.plot(x, pca.explained_variance_ratio_[:n_dims], 'o-', label='Individual')
            
            # Add cumulative variance line
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            plt.plot(x, cumulative[:n_dims], 'o-', label='Cumulative')
            
            # Simple elbow detection using cumulative variance thresholds
            dim_80 = np.where(cumulative >= 0.8)[0][0] + 1
            dim_90 = np.where(cumulative >= 0.9)[0][0] + 1
            
            print(f"Suggested dimensions:")
            print(f"- {dim_80} PCs explain 80% of variance")
            print(f"- {dim_90} PCs explain 90% of variance")
            
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('PCA Eigenvalue Spectrum')
            plt.grid(True)
            plt.legend()
            
            # Set x-axis ticks with specified spacing
            tick_positions = np.arange(1, n_dims + 1, xtick_spacing)
            plt.xticks(tick_positions)
            
            # Force integer ticks and set limits
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.xlim(1, n_dims)
            
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path + f'/pca_eigenvalues.{file_format}', 
                           facecolor='w', dpi=dpi, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        return embedding
    
    def _embed_diffusion_maps(self, epsilon=None, batch_size=10000, dist_fname='distances.dat', **kwargs):
        """Compute Diffusion Maps embedding."""
        '''D = self._compute_distance_matrix(batch_size, dist_fname)
        import diffusion_maps
        vals, embedding = diffusion_maps.op(D, epsilon, **kwargs)
        return embedding[:, :self.n_components]'''
        raise NotImplementedError(
        "Diffusion Maps embedding is not yet implemented. "
        "This feature will be available in a future release."
    )
    
    
    def _compute_distance_matrix(self, batch_size=10000, dist_fname='distances.dat', return_flat=False):
        """Compute pairwise distance matrix with GPU acceleration if available.
        
        Args:
            batch_size: Batch size for GPU computation
            dist_fname: Temporary file for distance matrix memmap
            return_flat: If True, returns flattened distance matrix for hierarchical clustering
        
        Returns:
            numpy.ndarray: Distance matrix (flattened if return_flat=True)
        """
        t1 = time.time()
        
        if self.gpu:
            try:
                import tensorflow as tf
                D = self._compute_distances_gpu(batch_size, dist_fname)
                D = np.array(D).astype(np.float32)
                np.fill_diagonal(D, 0)
                D = (D + D.T) / 2  # matrix may be antisymmetric due to precision errors
                if return_flat:
                    D = squareform(D)
                
                # Clean up temporary file
                try:
                    os.remove(dist_fname)  
                except FileNotFoundError:
                    pass
                
            except ImportError:
                print("TensorFlow not available. Falling back to CPU implementation.")
                D_flat = self._compute_distances_cpu()
                D = squareform(D_flat) if not return_flat else D_flat
        else:
            D_flat = self._compute_distances_cpu()
            D = squareform(D_flat) if not return_flat else D_flat
                
        print(f'Distances time: {time.time() - t1:.2f}s')
        return D
    
    def cluster(self, embedding=None, method='kmeans', n_clusters=10, **kwargs):
        """Cluster the embedded data.
        
        Args:
            embedding: Optional pre-computed embedding. If None, uses stored embedding
            method: Clustering method ('kmeans', 'dbscan', or 'hierarchical')
            n_clusters: Number of clusters for kmeans
            **kwargs: Additional clustering parameters
                For DBSCAN:
                    eps: Maximum distance between samples (default: 0.01)
                    min_samples: Minimum samples per cluster (default: 10)
                For KMeans:
                    random_state: Random seed (default: 0)
                    n_init: Number of initializations (default: 10)
                    max_iter: Maximum iterations (default: 300 for GPU, sklearn default for CPU)
                For Hierarchical:
                    batch_size: Batch size for GPU computation (default: 10000)
                    link_method: Linkage method (default: 'ward')
                    dist_fname: Temporary file for distance matrix
                    store_distances: Whether to return distances (default: False)
        
        Returns:
            For kmeans/dbscan:
                numpy.ndarray: Cluster labels for each sample
            For hierarchical:
                scipy.cluster.hierarchy.linkage: Linkage matrix for hierarchical clustering
                (use get_cluster_labels() to obtain cluster assignments)
            If store_distances=True with hierarchical:
                tuple: (linkage_matrix, distance_matrix)
        """
        if method not in self.SUPPORTED_CLUSTERERS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_CLUSTERERS}")
            
        # Special case for hierarchical clustering - works directly on maps
        if method == 'hierarchical':
            return self._cluster_hierarchical(**kwargs)
            
        # For other methods, need embedding
        if embedding is None:
            if self.embedding is None:
                raise ValueError("No embedding provided or computed. Run embed() first.")
            embedding = self.embedding

        if method in ['kmeans', 'dbscan']:
            if method == 'kmeans':
                # Try GPU k-means first if gpu=True
                if self.gpu:
                    KMeansTF = _check_kmeanstf_available()
                    if KMeansTF is not None:
                        clusterer = KMeansTF(
                            n_clusters=n_clusters,
                            init='k-means++',
                            random_state=kwargs.get('random_state', 0),
                            n_init=kwargs.get('n_init', 10),
                            max_iter=kwargs.get('max_iter', 300),
                            tol=kwargs.get('tol', 0.0001),
                            verbose=kwargs.get('verbose', 0)
                        )
                        # Convert to numpy array to avoid TensorFlow tensor issues
                        embedding_np = np.array(embedding.astype(np.float32))
                        clusterer.fit(embedding_np)
                        self.cluster_labels = clusterer.labels_

                        # Convert TensorFlow tensor to numpy array
                        if hasattr(self.cluster_labels, 'numpy'):
                            self.cluster_labels = self.cluster_labels.numpy()
                        return self.cluster_labels
                    else:
                        print("KMeansTF not available. Falling back to CPU K-means.")
                        print("For GPU acceleration, install kmeanstf:")
                        print("pip install kmeanstf")
                
                # CPU k-means (fallback or when gpu=False)
                KMeans, _, _ = _check_sklearn_available()
                clusterer = KMeans(
                    n_clusters=n_clusters,
                    init='k-means++',
                    random_state=kwargs.get('random_state', 0),
                    n_init=kwargs.get('n_init', 10)
                )
            else:  # dbscan
                _, DBSCAN, _ = _check_sklearn_available()
                clusterer = DBSCAN(
                    eps=kwargs.get('eps', 0.01),
                    min_samples=kwargs.get('min_samples', 10)
                )
            
        self.cluster_labels = clusterer.fit_predict(embedding)
        return self.cluster_labels
    
    def _cluster_hierarchical(self, batch_size=10000, link_method='ward', 
                            dist_fname='distances.dat', store_distances=False):
        """Perform hierarchical clustering with optional GPU acceleration."""
        D_flat = self._compute_distance_matrix(batch_size, dist_fname, return_flat=True)
        
        print('Computing hierarchical clusters...')
        t1 = time.time()
        linkage = hierarchy.linkage(D_flat, method=link_method, metric='euclidean')
        print(f'Linkage time: {time.time() - t1:.2f}s')
        
        if store_distances:
            return linkage, D_flat
        return linkage

    def _compute_distances_gpu(self, batch_size, dist_fname):
        """Compute pairwise distances on GPU using TensorFlow with memory mapping."""
        import tensorflow as tf
        
        A = tf.cast(self.maps, tf.float32)
        B = tf.cast(self.maps, tf.float32)
        num_A = A.shape[0]
        num_B = B.shape[0]
        
        # Create memory-mapped file for distance matrix
        distance_matrix = np.memmap(dist_fname, dtype=np.float32, mode='w+', 
                                shape=(num_A, num_B))
        
        def pairwise_distance(A_batch, B_batch):
            v_A = tf.expand_dims(tf.reduce_sum(tf.square(A_batch), 1), 1)
            v_B = tf.expand_dims(tf.reduce_sum(tf.square(B_batch), 1), 1)
            p1 = tf.reshape(v_A, (-1, 1))
            p2 = tf.reshape(v_B, (1, -1))
            dist_squared = tf.add(p1, p2) - 2 * tf.matmul(A_batch, B_batch, transpose_b=True)
            dist_squared = tf.maximum(dist_squared, 0.0)  # ensure non-negative values
            return tf.sqrt(dist_squared)
        
        # Check TensorFlow execution mode to handle different attribution methods:
        # - DeepSHAP requires non-eager execution (tf.compat.v1.disable_eager_execution())
        # - Other methods (saliency, smoothgrad, etc.) typically use eager execution
        # This check ensures compatibility with both modes when computing distances
        is_eager = tf.executing_eagerly()
        
        for i in tqdm(range(0, num_A, batch_size), desc='Distance batch'):
            A_batch = A[i:i + batch_size]
            for j in range(0, num_B, batch_size):
                B_batch = B[j:j + batch_size]
                distances = pairwise_distance(A_batch, B_batch)
                
                if is_eager:
                    # In eager mode, we can call .numpy() directly
                    distances = distances.numpy()
                else:
                    # In non-eager mode, we need to use a session
                    with tf.compat.v1.Session() as sess:
                        distances = sess.run(distances)
                
                distance_matrix[i:i + batch_size, j:j + batch_size] = distances
        
        distance_matrix.flush()  # Ensure all data is written to disk
        return distance_matrix

    def _compute_distances_cpu(self):
        """Compute pairwise distances on CPU."""
        if 0:
            nS = self.maps.shape[0]
            D_upper = np.zeros(shape=(nS, nS))
            
            for i in tqdm(range(nS), desc='Computing distances'):
                for j in range(i + 1, nS):
                    D_upper[i,j] = np.linalg.norm(self.maps[i,:] - self.maps[j,:])
            
            D = D_upper + D_upper.T - np.diag(np.diag(D_upper))  # Match original code
            return squareform(D)
        else:
            """Compute pairwise distances on CPU using vectorized operations."""
            return pdist(self.maps.reshape(self.maps.shape[0], -1))
    
    def normalize(self, embedding, to_sum=False, copy=True):
        """Normalize embedding vectors to [0,1] range.
        
        Args:
            embedding: Array of shape (n_samples, n_dimensions)
            to_sum: If True, normalize to sum=1. If False, normalize to range [0,1]
            copy: If True, operate on a copy of the data
        
        Returns:
            numpy.ndarray: Normalized embedding
        """
        d = embedding if not copy else np.copy(embedding)
        d -= np.min(d, axis=0)
        d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0))  # normalize to [0,1]
        return d
    
    def plot_embedding(self, embedding, labels=None, dims=[0,1], 
                    normalize=False, cmap='jet', s=2.5, alpha=1.0, 
                    linewidth=0.1, colorbar_label=None, sort_order=None, 
                    ref_index=None, legend_loc='upper left', figsize=None,
                    save_path=None, dpi=200, file_format='png'):
        """Plot embedding and optionally color by labels/values.
        
        Args:
            embedding: Array of shape (n_samples, n_dimensions)
            labels: Values for coloring points. Can be:
                - numpy array of shape (N,) or (N,1)
                - pandas Series/DataFrame column (e.g., mave['DNN'])
                - None (points will be single color)
            dims: Which dimensions to plot [dim1, dim2]
            normalize: Whether to normalize embedding to [0,1] range
            cmap: Colormap for points (e.g., 'viridis', 'jet', 'tab10')
                - Use 'viridis'/'jet' for continuous values
                - Use 'tab10'/'Set3' for discrete clusters
            s: Point size (default: 2.5)
            alpha: Point transparency (default: 1.0)
            linewidth: Width of point edges (default: 0.1)
            colorbar_label: Label for colorbar (if None, no colorbar shown)
            sort_order: Order to plot points ('ascending', 'descending', or None)
                - Useful for ensuring important points are plotted on top
                - Points are sorted based on their label values
                - Works with both numpy arrays and pandas Series/DataFrames
            ref_index: Index of reference/wild-type sequence to highlight (default: None)
                - Will be shown as a black star on the plot
            legend_loc: Location of legend for reference sequence ('best', 'top left', 'upper right', etc.)
            figsize: Figure size (width, height) in inches (default: None, uses matplotlib default)
            save_path: Path to save figure (if None, displays plot)
            dpi: DPI for saved figure (default: 200)
            file_format: Format for saved figure (default: 'png'). Common formats: 'png', 'pdf', 'svg', 'eps'
        
        Example usage:
            # Basic plot with reference sequence
            clusterer.plot_embedding(embedding, ref_index=0)
            
            # Color by DNN predictions with colorbar and reference
            clusterer.plot_embedding(
                embedding,
                labels=mave['DNN'],  # or y_mut numpy array
                colorbar_label='DNN prediction',
                sort_order='descending',  # high predictions on top
                ref_index=ref_idx
            )
        """
        try:
            plt.close()
        except:
            pass

        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()

        # Normalize embedding if requested  
        if normalize:
            embedding = self.normalize(embedding)
        
        # Ensure embedding has enough dimensions
        if embedding.shape[1] <= max(dims):
            raise ValueError(f"Embedding has {embedding.shape[1]} dimensions, but dims={dims} was requested")
        
        # Convert labels to numpy array if needed and flatten
        if labels is not None:
            if hasattr(labels, 'values'):  # pandas Series/DataFrame
                labels = labels.values
            labels = labels.flatten()  # Handle both (N,1) and (N,) arrays
        
        # Sort points if requested
        if labels is not None and sort_order is not None:
            if sort_order not in ['ascending', 'descending']:
                raise ValueError("sort_order must be 'ascending', 'descending', or None")
            sort_idx = np.argsort(labels)
            if sort_order == 'descending':
                sort_idx = sort_idx[::-1]
        else:
            sort_idx = slice(None)  # equivalent to ':'
        
        scatter = plt.scatter(
            embedding[:, dims[0]][sort_idx], 
            embedding[:, dims[1]][sort_idx], 
            c=None if labels is None else labels[sort_idx],
            cmap=cmap,
            s=s,
            alpha=alpha,
            linewidth=linewidth,
            edgecolors='k',
            zorder=0
        )
        
        # Plot reference sequence if specified
        if ref_index is not None:
            plt.scatter(
                embedding[ref_index, dims[0]], 
                embedding[ref_index, dims[1]], 
                marker='*', 
                c='k', 
                s=200,  # make star larger than regular points
                label='Reference',
                zorder=100
            )
            plt.legend(markerscale=1, fontsize='small', frameon=False, loc=legend_loc)

        # Create colorbar using the main scatter object
        if colorbar_label is not None and labels is not None:
            plt.colorbar(scatter, label=colorbar_label)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel(f'Ψ{dims[0]+1}')
        plt.ylabel(f'Ψ{dims[1]+1}')
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path + f'/attributions_embedding.{file_format}', 
                       facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_histogram(self, embedding, dims=[0,1], bins=101, cmap='viridis', 
                    colorbar_label='Count', figsize=None, save_path=None, dpi=200,
                    file_format='png'):
        """Plot 2D histogram of embedding points.
        
        Args:
            embedding: Array of shape (n_samples, n_dimensions)
            dims: Which dimensions to plot [dim1, dim2]
            bins: Number of bins for histogram (default: 101)
            cmap: Colormap for histogram (default: 'viridis')
            colorbar_label: Label for colorbar (if None, shows 'Count')
            figsize: Figure size (width, height) in inches (default: None, uses matplotlib default)
            save_path: Path to save figure (if None, displays plot)
            dpi: DPI for saved figure (default: 200)
            file_format: Format for saved figure (default: 'png'). Common formats: 'png', 'pdf', 'svg', 'eps'
        """
        try:
            plt.close()
        except:
            pass
        
        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
        
        # Ensure embedding has enough dimensions
        if embedding.shape[1] <= max(dims):
            raise ValueError(f"Embedding has {embedding.shape[1]} dimensions, but dims={dims} was requested")
        
        # Always normalize the embedding
        embedding = self.normalize(embedding)
        
        # Create 2D histogram
        H, edges = np.histogramdd(
            np.vstack((embedding[:, dims[0]], embedding[:, dims[1]])).T,
            bins=bins,
            range=((0,1), (0,1)),
            density=False,
            weights=None
        )
        
        # Plot histogram
        mesh = plt.pcolormesh(H.T, cmap=cmap)
        
        if colorbar_label is not None:
            plt.colorbar(mesh, label=colorbar_label)
        else:
            plt.colorbar(mesh, label='Count')
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel(f'Ψ{dims[0]+1}')
        plt.ylabel(f'Ψ{dims[1]+1}')
        plt.gca().set_aspect('equal')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path + f'/attributions_embedding_histogram.{file_format}', 
                       facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_dendrogram(self, linkage, figsize=(15, 10), leaf_rotation=90, 
                        leaf_font_size=8, cut_level=None, save_path=None, dpi=200,
                        file_format='png', ax=None, truncate=True, cut_level_truncate=None, 
                        criterion=None, n_clusters=None, gui=False):
        """Plot dendrogram from hierarchical clustering linkage matrix.
        
        Args:
            linkage: Hierarchical clustering linkage matrix
            figsize: Figure size (width, height)
            leaf_rotation: Rotation of leaf labels
            leaf_font_size: Font size for leaf labels
            cut_level: Optional height at which to draw horizontal cut line
            save_path: Path to save figure (if None, displays plot)
            dpi: DPI for saved figure (default: 200)
            file_format: Format for saved figure (default: 'png'). Common formats: 'png', 'pdf', 'svg', 'eps'
            ax: Matplotlib axis to plot on (for GUI use). If provided, plots on existing axis instead of creating new figure
            truncate: Whether to truncate dendrogram for large datasets (for GUI use). Only used when ax is provided
            cut_level_truncate: Height at which to truncate dendrogram (for GUI use). Used with truncate=True
            criterion: Clustering criterion ('maxclust' or 'distance') for truncation calculation. Used with truncate=True
            n_clusters: Number of clusters (for maxclust criterion) for truncation calculation. Used with truncate=True and criterion='maxclust'
            gui: Whether to apply GUI-specific styling (smaller fonts, removed spines, etc.) (default: False)
        """
        sys.setrecursionlimit(100000)  # Fix for large dendrograms
        
        # GUI mode: plot on existing axis
        if ax is not None:
            ax.clear()
            dendro_params = {
                'leaf_rotation': leaf_rotation,
                'leaf_font_size': leaf_font_size,
                'ax': ax
            }
            if truncate and cut_level_truncate is not None:
                # Calculate p based on the user's clustering parameters
                if criterion == 'maxclust':
                    p = n_clusters
                else:
                    # For distance criterion, calculate how many clusters are formed at this distance
                    clusters_at_distance = hierarchy.fcluster(linkage, cut_level_truncate, criterion='distance')
                    p = len(np.unique(clusters_at_distance))
                
                dendro_params.update({
                    'truncate_mode': 'lastp',
                    'p': p,
                    'show_leaf_counts': False
                })
                ax.set_xlabel('')
            else:
                # Don't truncate - show full dendrogram
                dendro_params['no_labels'] = True
                ax.set_xlabel('')
                ax.set_xticks([])

            hierarchy.dendrogram(linkage, **dendro_params)

            if cut_level is not None:
                # Make reference line thinner and add legend
                ax.axhline(y=cut_level, color='r', linestyle='--', linewidth=0.5, label='Reference')
                if gui:
                    ax.legend(loc='best', fontsize=6, frameon=False)
                else:
                    ax.legend(loc='best', frameon=False)

            # Apply GUI-specific styling only when gui=True
            if gui:
                # Make boundary lines thinner to match embedding figure
                for spine in ax.spines.values():
                    spine.set_linewidth(0.1)
                    spine.set_visible(False)  # Remove black rectangle border for GUI mode

                ax.set_title('Hierarchical Clustering Dendrogram', fontsize=8)
                ax.set_ylabel('Distance', fontsize=6)
                ax.tick_params(axis='both', which='major', labelsize=4)
            else:
                # Regular styling for non-GUI mode
                ax.set_title('Hierarchical Clustering Dendrogram')
                ax.set_ylabel('Distance')
            
            return
        
        # Original functionality
        plt.figure(figsize=figsize)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('')  # Remove x-label
        plt.ylabel('Distance')
        
        with plt.rc_context({'lines.linewidth': 2}):
            hierarchy.dendrogram(
                linkage,
                leaf_rotation=leaf_rotation,
                leaf_font_size=leaf_font_size,
            )

        if cut_level is not None:
            # Make reference line thinner and add legend
            plt.axhline(y=cut_level, color='r', linestyle='-', linewidth=0.5, label='Reference')
            if gui:
                plt.legend(loc='best', fontsize=4, frameon=False)
            else:
                plt.legend(loc='best', frameon=False)
        
        # Add padding around the dendrogram
        if gui:
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()
            x_pad = (xlim[1] - xlim[0]) * 0.1
            y_pad = (ylim[1] - ylim[0]) * 0.1
            plt.gca().set_xlim(xlim[0] - x_pad, xlim[1] + x_pad)
            plt.gca().set_ylim(ylim[0] - y_pad, ylim[1] + y_pad)
            plt.gca().tick_params(axis='both', which='major', labelsize=4)

        plt.xticks([])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        if save_path:
            plt.savefig(save_path + f'/attributions_dendrogram.{file_format}', 
                       facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def get_cluster_labels(self, linkage, criterion='maxclust', max_distance=10, n_clusters=200):
        """Get cluster labels from a linkage matrix.
        
        Args:
            linkage: Linkage matrix from scipy.cluster.hierarchy.linkage
            criterion: How to form flat clusters ('distance' or 'maxclust')
                'distance': Cut tree at specified height 
                'maxclust': Produce specified number of clusters
            max_distance: Maximum cophenetic distance within clusters 
                        (only used if criterion='distance')
            n_clusters: Desired number of clusters to produce
                    (only used if criterion='maxclust')
        
        Returns:
            numpy.ndarray: Cluster labels (zero-indexed)
            float: Cut level (max_distance if criterion='distance', or computed level if criterion='maxclust')
        """
        if criterion == 'distance':
            clusters = hierarchy.fcluster(linkage, max_distance, criterion='distance')
            clusters = clusters - 1  # Zero-index clusters
            self.cluster_labels = clusters  # Store labels in the instance
            return clusters, max_distance
            
        elif criterion == 'maxclust':
            # Check if n_clusters is greater than number of sequences
            n_sequences = len(linkage) + 1  # Number of sequences is rows in linkage + 1
            if n_clusters > n_sequences:
                raise ValueError(f"Requested {n_clusters} clusters but only have {n_sequences} sequences. "
                               f"Number of clusters must be less than or equal to number of sequences.")
            
            clusters = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')
            clusters = clusters - 1  # Zero-index clusters
            
            # Find the cut level that gives the desired number of clusters
            sorted_heights = np.sort(linkage[:, 2])[::-1]  # Heights from highest to lowest
            for height in sorted_heights:
                temp_clusters = hierarchy.fcluster(linkage, height, criterion='distance')
                if len(np.unique(temp_clusters)) == n_clusters:
                    max_d = height
                    break
                    
            print(f"Cut level for {n_clusters} clusters: {max_d:.3f}")
            self.cluster_labels = clusters  # Store labels in the instance
            return clusters, max_d
            
        else:
            raise ValueError("criterion must be either 'distance' or 'maxclust'")