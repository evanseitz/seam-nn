import os
import sys
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy

class Clusterer:
    """
    Clusterer: A unified interface for embedding and clustering attribution maps
    
    This implementation provides implementations of common embedding 
    and clustering methods for attribution maps:
    
    Embedding Methods:
    - UMAP (requires umap-learn)
    - PHATE (requires phate)
    - t-SNE (requires openTSNE)
    - PCA (requires scikit-learn)
    - Diffusion Maps

    Clustering Methods:
    - Hierarchical (GPU-optimized available)
    - K-means
    - DBSCAN
    
    Requirements:
    - numpy
    - scipy
    - scikit-learn (for PCA, K-means, DBSCAN)
    
    Optional Requirements:
    - tensorflow (for GPU-accelerated hierarchical clustering)
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

    def __init__(self, attribution_maps, method='umap', n_components=2, gpu=True):
        """Initialize the Clusterer.
        
        Args:
            attribution_maps: numpy array of shape (N, L, A) containing attribution maps
            method: Embedding method (default: 'umap')
            n_components: Number of dimensions for embedding (default: 2)
            gpu: Whether to use GPU acceleration when available (default: True)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method must be one of {self.SUPPORTED_METHODS}")
        
        self.embedding = None
        self.cluster_labels = None
        self.maps = attribution_maps
        self.method = method
        self.n_components = n_components
        self.gpu = gpu
        
        # Reshape maps if needed
        if len(self.maps.shape) == 3:
            N, L, A = self.maps.shape
            self.maps = self.maps.reshape((N, L*A))
            
    def embed(self, **kwargs):
        """Compute embedding using specified method.
        
        Args:
            **kwargs: Method-specific parameters
            
        Returns:
            numpy.ndarray: Embedded coordinates
        """
        t0 = time.time()
        
        if self.method == 'umap':
            embedding = self._embed_umap(**kwargs)
        elif self.method == 'phate':
            embedding = self._embed_phate(**kwargs)
        elif self.method == 'tsne':
            embedding = self._embed_tsne(**kwargs)
        elif self.method == 'pca':
            embedding = self._embed_pca(**kwargs)
        elif self.method == 'diffmap':
            embedding = self._embed_diffusion_maps(**kwargs)
            
        print(f'Embedding time: {time.time() - t0:.2f}s')
        return embedding
    
    def _embed_umap(self, **kwargs):
        """Compute UMAP embedding."""
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP requires the 'umap-learn' package. Install with: conda install -c conda-forge umap-learn. May also need: pip install pynndescent==0.5.8")
            
        fit = umap.UMAP(n_components=self.n_components, **kwargs)
        return fit.fit_transform(self.maps)
    
    def _embed_phate(self, **kwargs):
        """Compute PHATE embedding."""
        try:
            import phate
        except ImportError:
            raise ImportError("PHATE requires the 'phate' package. Install with: pip install --user phate")
            
        phate_op = phate.PHATE(n_components=self.n_components, **kwargs)
        return phate_op.fit_transform(self.maps)
    
    def _embed_tsne(self, perplexity=30, n_jobs=8, random_state=42, **kwargs):
        """Compute t-SNE embedding."""
        try:
            from openTSNE import TSNE
        except ImportError:
            raise ImportError("t-SNE requires the 'openTSNE' package. Install with: pip install openTSNE")
            
        tsne = TSNE(
            perplexity=perplexity,
            metric="euclidean",
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        return tsne.fit(self.maps)
    
    def _embed_pca(self, **kwargs):
        """Compute PCA embedding."""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("PCA requires the 'scikit-learn' package. Install with: pip install scikit-learn")
            
        return PCA(n_components=self.n_components, **kwargs).fit_transform(self.maps)
    
    def _embed_diffusion_maps(self, epsilon=None, batch_size=10000, dist_fname='distances.dat', **kwargs):
        """Compute Diffusion Maps embedding."""
        D = self._compute_distance_matrix(batch_size, dist_fname)
        import diffusion_maps
        vals, embedding = diffusion_maps.op(D, epsilon, **kwargs)
        return embedding[:, :self.n_components]
    
    def _compute_distance_matrix(self, batch_size=10000, dist_fname='distances.dat', return_flat=False):
        """Compute pairwise distance matrix with GPU acceleration if available.
        
        Args:
            batch_size: Batch size for GPU computation
            dist_fname: Temporary file for distance matrix memmap
            return_flat: If True, returns flattened distance matrix for hierarchical clustering
        
        Returns:
            numpy.ndarray: Distance matrix (flattened if return_flat=True)
        """
        print('Computing distances...')
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
                os.remove(dist_fname)  # Clean up temporary file
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
            
        if embedding is None:
            if self.embedding is None:
                raise ValueError("No embedding provided or computed. Run embed() first.")
            embedding = self.embedding

        if method == 'hierarchical':
            return self._cluster_hierarchical(**kwargs)
        elif method == 'kmeans':
            clusterer = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                random_state=kwargs.get('random_state', 0),
                n_init=kwargs.get('n_init', 10)
            )
        elif method == 'dbscan':
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
        
        for i in tqdm(range(0, num_A, batch_size), desc='Distance batch'):
            A_batch = A[i:i + batch_size]
            for j in range(0, num_B, batch_size):
                B_batch = B[j:j + batch_size]
                distances = pairwise_distance(A_batch, B_batch).numpy()
                distance_matrix[i:i + batch_size, j:j + batch_size] = distances
        
        distance_matrix.flush()  # Ensure all data is written to disk
        return distance_matrix

    def _compute_distances_cpu(self):
        """Compute pairwise distances on CPU."""
        nS = self.maps.shape[0]
        D_upper = np.zeros(shape=(nS, nS))
        
        for i in tqdm(range(nS), desc='Computing distances'):
            for j in range(i + 1, nS):
                D_upper[i,j] = np.linalg.norm(self.maps[i,:] - self.maps[j,:])
        
        D = D_upper + D_upper.T - np.diag(np.diag(D_upper))  # Match original code
        return squareform(D)
    
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
                    normalize=False, cmap='tab10', s=2.5, alpha=1.0, 
                    save_path=None, dpi=200):
        """Plot embedding and optionally color by clusters.
        
        Args:
            embedding: Array of shape (n_samples, n_dimensions)
            labels: Optional cluster labels (if None, uses stored labels)
            dims: Which dimensions to plot [dim1, dim2]
            normalize: Whether to normalize embedding to [0,1] range
            cmap: Colormap for clusters
            s: Point size
            alpha: Point transparency
            save_path: Optional path to save figure
            dpi: DPI for saved figure (default: 200)
        """
        if normalize:
            embedding = self.normalize(embedding)
                    
        plt.figure(figsize=(8, 8))
        plt.scatter(
            embedding[:, dims[0]], 
            embedding[:, dims[1]], 
            c=labels,
            cmap=cmap,
            s=s,
            alpha=alpha,
            linewidth=.1,
            edgecolors='k'
        )
        plt.xlabel(f'Dimension {dims[0]+1}')
        plt.ylabel(f'Dimension {dims[1]+1}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_dendrogram(self, linkage, figsize=(15, 10), leaf_rotation=90, 
                        leaf_font_size=8, save_path=None, dpi=200):
        """Plot dendrogram from hierarchical clustering linkage matrix.
        
        Args:
            linkage: Hierarchical clustering linkage matrix
            figsize: Figure size (width, height)
            leaf_rotation: Rotation of leaf labels
            leaf_font_size: Font size for leaf labels
            save_path: Path to save figure (if None, displays plot)
            dpi: DPI for saved figure (default: 200)
        """
        sys.setrecursionlimit(100000)  # Fix for large dendrograms
        
        plt.figure(figsize=figsize)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        
        with plt.rc_context({'lines.linewidth': 2}):
            hierarchy.dendrogram(
                linkage,
                leaf_rotation=leaf_rotation,
                leaf_font_size=leaf_font_size,
            )
        
        plt.xticks([])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel('Clusters')
        
        if save_path:
            plt.savefig(save_path, facecolor='w', dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def get_cluster_labels(self, linkage, criterion='maxclust', max_distance=10, n_clusters=200):
        """Get cluster labels from hierarchical clustering linkage matrix.
        
        Args:
            linkage: Hierarchical clustering linkage matrix
            criterion: How to form flat clusters ('distance' or 'maxclust')
                'distance': Cut tree at specified height 
                'maxclust': Produce specified number of clusters
            max_distance: Maximum cophenetic distance within clusters 
                        (only used if criterion='distance')
            n_clusters: Desired number of clusters to produce
                    (only used if criterion='maxclust')
        
        Returns:
            numpy.ndarray: Cluster labels (zero-indexed)
            float: Cut level (only if criterion='maxclust')
        """
        if criterion == 'distance':
            clusters = hierarchy.fcluster(linkage, max_distance, criterion='distance')
        elif criterion == 'maxclust':
            clusters = hierarchy.fcluster(linkage, n_clusters, criterion='maxclust')
        else:
            raise ValueError("criterion must be either 'distance' or 'maxclust'")
            
        clusters = clusters - 1  # Zero-index clusters
        
        if criterion == 'maxclust':
            # Find the cut level that gives the desired number of clusters
            max_d = 0
            for i in range(1, len(linkage) + 1):
                if len(np.unique(hierarchy.fcluster(linkage, i, criterion='maxclust'))) == n_clusters:
                    max_d = linkage[i-1, 2]
                    break
            print(f"Cut level for {n_clusters} clusters: {max_d:.3f}")
            return clusters, max_d
        
        return clusters


# TO DO:
# - plot points colored by DNN prediction; Hamming distance; GIA score, etc.
#   - code from view_embedding.py, update_rerun_umap.py, GUI, etc.
# - need to compile mave.csv with DNN scores, Hamming distance, GIA score, etc.