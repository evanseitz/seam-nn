"""
SEAM example of local library using DeepSTARR enhancers - GPU-ONLY PERSISTENT WORKER WITH MEMORY POOLING

ARROW FILE LOADING:
At the end of this script, there is minimalistic code to load saved Arrow files and extract
the essential data (reference cluster averages, background sequences, MSM data, etc.) for
downstream genome-wide SEAM-based annotation workflows.

This script implements a persistent TensorFlow worker that loads the model once and keeps it in GPU memory
for processing multiple sequences efficiently. This eliminates the overhead of repeated model loading.

REQUIREMENTS:
- GPU REQUIRED: This script will fail immediately if no GPU is detected
- TensorFlow with GPU support
- Additional requirements: 
    - pip install squid-nn
    - pip install seam-nn
    - pip install pyarrow

OPTIONAL REQUIREMENTS:
- cuml: For GPU-accelerated PCA (use correct cuda version, e.g. for cuda 11: pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com)
- kmeanstf: For GPU-accelerated k-means clustering (pip install kmeanstf)
- cupy: For enhanced GPU memory management (pip install cupy-cuda11x -- replace with your CUDA version)

MEMORY POOLING OPTIMIZATION:
- Pre-allocates arrays with optimized data types for maximum memory efficiency
- Eliminates repeated np.zeros() calls and reduces memory fragmentation
- Improves GPU memory locality and reduces allocation overhead
- Automatically manages array lifecycle with get_arrays() and release_arrays()

ESSENTIAL DATA ONLY:
- Focuses on core computation pipeline without optional features
- Saves only essential data for downstream genome-wide SEAM-based annotation
- Removes figure generation, logo rendering, and verification features
- Streamlined for maximum performance in batch processing

For batch processing across multiple GPUs:
    Single sequence:
        python example_deepstarr_local_batch_persistent.py 463513 1  # sequence 463513, Hk task
    Range of sequences (persistent mode):
        CUDA_VISIBLE_DEVICES=3 python example_deepstarr_local_batch_persistent.py 0 99999 0
        CUDA_VISIBLE_DEVICES=4 python example_deepstarr_local_batch_persistent.py 100000 199999 0
        CUDA_VISIBLE_DEVICES=5 python example_deepstarr_local_batch_persistent.py 200000 299999 0
        CUDA_VISIBLE_DEVICES=7 python example_deepstarr_local_batch_persistent.py 300000 399999 0

Model:
    - DeepSTARR

Batch parameters:
    - start_index: First sequence index to process (inclusive)
    - stop_index: Last sequence index to process (inclusive)
    - task_index: Index of DeepSTARR task to analyze (0 = Dev, 1 = Hk)
    
    Single sequence mode (legacy):
    - seq_index: Index of DeepSTARR sequence to analyze from test set
    - task_index: Index of DeepSTARR task to analyze (0 = Dev, 1 = Hk)

Default parameters:
    - 10k sequences
    - 10% mutation rate
    - Integrated gradients
    - PCA + K-means clustering with 20 clusters
    - Adaptive background scaling enabled
    - Hardcoded sequence length: 249bp (DeepSTARR standard)

PERSISTENT WORKER FEATURES:
    - Model loaded once and kept in GPU memory for entire batch
    - Eliminates repeated tf.keras.models.load_model() calls
    - Maintains warm GPU state between sequences
    - Processes range of sequences efficiently (start_index to stop_index)
    - Fault-tolerant: continues processing if individual sequences fail
    - Manual multi-GPU support via CUDA_VISIBLE_DEVICES environment variable
    - Same output format and functionality as original script

CLUSTERING APPROACH:
    - Users can choose between two clustering methods via if/else switch:
      - PCA + K-means clustering (default): Uses PCA embedding to reduce dimensionality before k-means clustering
      - Hierarchical clustering: Direct hierarchical clustering with Ward linkage on attribution maps
    - PCA approach: Computes PCA with 10 components to explain variance while reducing noise
    - Performs clustering using SEAM's Clusterer class
    - GPU-accelerated when cuML is available, falls back to CPU if not

DATA TYPE OPTIMIZATIONS:
    - x_mut: int8 one-hot sequences
    - y_mut: float16 predictions
    - attributions: float16 attribution maps
    - PCA embeddings: float16 dimensionality reduction
    - Saved Arrow files: float16 for attribution data, int8 for sequence data
"""

import os, sys
import time
import random
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Suppress TensorRT warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import squid
from urllib.request import urlretrieve
import pandas as pd
import gc
import pyarrow as pa
import pyarrow.feather as feather
import h5py
from keras.models import model_from_json

# Optional cupy import for GPU memory management
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available for GPU memory management")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - GPU memory management will be limited")
    print("For better GPU memory management, install cupy:")
    print("pip install cupy-cuda11x  # replace with your CUDA version")

# Global persistent model and data storage
PERSISTENT_MODEL = None
PERSISTENT_X_DATASET = None
PERSISTENT_ASSETS_DIR = None
PERSISTENT_ALPHABET = ['A','C','G','T']

# Global processing parameters
PERSISTENT_MUT_RATE = 0.1  # mutation rate for in silico mutagenesis
PERSISTENT_NUM_SEQS = 10000  # number of sequences to generate
PERSISTENT_N_CLUSTERS = 20  # number of clusters for PCA + k-means clustering
PERSISTENT_N_PCA_COMPONENTS = 10  # number of PCA components for dimensionality reduction
PERSISTENT_SEQ_LEN = 249  # hardcoded sequence length for DeepSTARR (all sequences are 249bp)
PERSISTENT_ATTRIBUTION_METHOD = 'intgrad'  # {saliency, smoothgrad, intgrad, ism}
PERSISTENT_ADAPTIVE_BACKGROUND_SCALING = True  # Whether to use cluster-specific background scaling


# Global GPU detection
PERSISTENT_GPU = len(tf.config.list_physical_devices('GPU')) > 0
if not PERSISTENT_GPU:
    print("="*80)
    print("ERROR: GPU REQUIRED")
    print("="*80)
    print("This script requires a GPU to run. No GPU devices were detected.")
    print("Please ensure:")
    print("  - CUDA is properly installed")
    print("  - TensorFlow can access GPU devices")
    print("  - You are running on a machine with GPU hardware")
    print("="*80)
    raise RuntimeError("GPU is required for persistent worker. No GPU devices found.")

class MemoryPool:
    """Memory pool for pre-allocating frequently used arrays to avoid repeated allocations."""
    
    def __init__(self, max_seqs=10000, max_seq_length=200, max_pca_components=20):
        """
        Initialize memory pool with pre-allocated arrays.
        
        Args:
            max_seqs: Maximum number of sequences to support
            max_seq_length: Maximum sequence length to support
            max_pca_components: Maximum number of PCA components to support
        """
        print(f"Initializing memory pool: {max_seqs} sequences × {max_seq_length} positions")
        
        # Pre-allocate arrays for mutagenesis data (optimized data types)
        self.x_mut_pool = np.zeros((max_seqs, max_seq_length, 4), dtype=np.int8)  # One-hot sequences
        self.y_mut_pool = np.zeros((max_seqs, 1), dtype=np.float16)  # Predictions (float16 precision sufficient)
        
        # Pre-allocate arrays for attribution maps (float16 for memory efficiency)
        self.attributions_pool = np.zeros((max_seqs, max_seq_length, 4), dtype=np.float16)
        
        # Pre-allocate array for PCA embedding (float16 for memory efficiency)
        self.pca_embedding_pool = np.zeros((max_seqs, max_pca_components), dtype=np.float16)
        
        # Track available slots
        self.available_slots = list(range(max_seqs))
        self.max_seqs = max_seqs
        self.max_seq_length = max_seq_length
        self.max_pca_components = max_pca_components
        
        print(f"Memory pool initialized with {len(self.available_slots)} available slots")
        print(f"PCA embedding pool: {max_seqs} sequences × {max_pca_components} components")
    
    def get_arrays(self, num_seqs, seq_length):
        """
        Get pre-allocated arrays for the specified number of sequences.
        
        Args:
            num_seqs: Number of sequences needed
            seq_length: Length of each sequence
            
        Returns:
            Tuple of (x_mut, y_mut, attributions, attributions_flat) views
        """
        if num_seqs > len(self.available_slots):
            raise ValueError(f"Requested {num_seqs} sequences but only {len(self.available_slots)} available")
        
        if seq_length > self.max_seq_length:
            raise ValueError(f"Requested sequence length {seq_length} exceeds max {self.max_seq_length}")
        
        # Get slots for this request
        slots = self.available_slots[:num_seqs]
        self.available_slots = self.available_slots[num_seqs:]
        
        # Return views into pre-allocated memory
        x_mut = self.x_mut_pool[slots, :seq_length, :]
        y_mut = self.y_mut_pool[slots, :num_seqs]
        attributions = self.attributions_pool[slots, :seq_length, :]
        
        return x_mut, y_mut, attributions, slots
    
    def get_pca_embedding(self, num_seqs, n_components):
        """
        Get pre-allocated PCA embedding array.
        
        Args:
            num_seqs: Number of sequences needed
            n_components: Number of PCA components needed
            
        Returns:
            View into pre-allocated PCA embedding array
        """
        if n_components > self.max_pca_components:
            raise ValueError(f"Requested {n_components} PCA components but only {self.max_pca_components} available")
        
        # Return view into pre-allocated PCA embedding memory
        return self.pca_embedding_pool[:num_seqs, :n_components]
    
    def release_arrays(self, slots):
        """
        Release arrays back to the pool.
        
        Args:
            slots: List of slot indices to release
        """
        # No need to zero arrays - they'll be overwritten on next use
        # Mark slots as available again
        self.available_slots.extend(slots)
        
    def get_pool_status(self):
        """Get current pool status for debugging."""
        return {
            'total_slots': self.max_seqs,
            'available_slots': len(self.available_slots),
            'used_slots': self.max_seqs - len(self.available_slots),
            'max_seq_length': self.max_seq_length
        }
    
    def get_gpu_memory_info(self):
        """Get GPU memory usage information."""
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                return {
                    'used': mempool.used_bytes(),
                    'total': mempool.total_bytes(),
                    'free': mempool.free_bytes(),
                    'pinned_used': pinned_mempool.used_bytes(),
                    'pinned_total': pinned_mempool.total_bytes()
                }
            except Exception as e:
                return {'error': str(e)}
        else:
            return {'error': 'CuPy not available'}

# Global memory pool
PERSISTENT_MEMORY_POOL = None

def initialize_persistent_resources():
    """Initialize persistent resources (model, data) that will be reused across sequences."""
    global PERSISTENT_MODEL, PERSISTENT_X_DATASET, PERSISTENT_ASSETS_DIR, PERSISTENT_MEMORY_POOL
    
    if PERSISTENT_MODEL is not None:
        print("Persistent resources already initialized, skipping...")
        return
    
    print("Initializing persistent resources (model, data, and memory pool)...")
    start_time = time.time()
    
    # Create assets_deepstarr folder if it doesn't exist
    py_dir = os.path.dirname(os.path.abspath(__file__))
    PERSISTENT_ASSETS_DIR = os.path.join(py_dir, 'assets_deepstarr')
    if not os.path.exists(PERSISTENT_ASSETS_DIR):
        os.makedirs(PERSISTENT_ASSETS_DIR)

    def download_if_not_exists(url, filename):
        """Download a file if it doesn't exist locally."""
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urlretrieve(url, filename)
        else:
            print(f"Using existing {filename}")
        
        # Validate file is not empty
        if os.path.getsize(filename) == 0:
            print(f"ERROR: {filename} is empty! Re-downloading...")
            urlretrieve(url, filename)
            if os.path.getsize(filename) == 0:
                raise RuntimeError(f"Failed to download {filename} - file is still empty after retry")
        
        print(f"✓ {filename} validated ({os.path.getsize(filename)} bytes)")

    # Define URLs and filenames
    files = {
        'deepstarr.model.json': 'https://www.dropbox.com/scl/fi/y1mwsqpv2e514md9t68jz/deepstarr.model.json?rlkey=cdwhstqpv2e514md9t68jz&st=9a0c5skz&dl=1',
        'deepstarr.model.h5': 'https://www.dropbox.com/scl/fi/6nl6e2hofyw70lh99h3uk/deepstarr.model.h5?rlkey=hqfnivn199xa54bjh8dn2jpaf&st=l4jig4ky&dl=1',
        'deepstarr_data.h5': 'https://www.dropbox.com/scl/fi/cya4ntqk2o8yftxql52lu/deepstarr_data.h5?rlkey=5ly363vqjb3vaw2euw2dhsjo3&st=6eod6fg8&dl=1'
    }

    # Download files to assets_deepstarr folder
    print("Downloading required files...")
    for filename, url in files.items():
        filepath = os.path.join(PERSISTENT_ASSETS_DIR, filename)
        download_if_not_exists(url, filepath)

    keras_model_weights = os.path.join(PERSISTENT_ASSETS_DIR, 'deepstarr.model.h5')
    keras_model_json = os.path.join(PERSISTENT_ASSETS_DIR, 'deepstarr.model.json')

    # Load all sequences from all splits (persistent)
    print("Loading DeepSTARR dataset...")
    with h5py.File(os.path.join(PERSISTENT_ASSETS_DIR, 'deepstarr_data.h5'), 'r') as dataset:
        X_train = np.array(dataset['x_train']).astype(np.float32)
        X_valid = np.array(dataset['x_valid']).astype(np.float32)
        X_test = np.array(dataset['x_test']).astype(np.float32)
        
        # Combine all sequences into one dataset
        PERSISTENT_X_DATASET = np.concatenate([X_train, X_valid, X_test], axis=0)
        print(f"Loaded {len(PERSISTENT_X_DATASET)} total sequences")
        print(f"Sequence shape: {PERSISTENT_X_DATASET.shape}")
        
        # Validate hardcoded sequence length matches dataset
        actual_seq_len = PERSISTENT_X_DATASET.shape[1]
        if actual_seq_len != PERSISTENT_SEQ_LEN:
            raise ValueError(f"Hardcoded sequence length {PERSISTENT_SEQ_LEN} doesn't match dataset sequence length {actual_seq_len}")
        print(f"✓ Sequence length validation passed: {PERSISTENT_SEQ_LEN}bp")

    # Load the DeepSTARR model (persistent)
    print("Loading DeepSTARR model...")
    
    # Validate JSON file before loading
    try:
        with open(keras_model_json, 'r') as f:
            json_content = f.read()
            if not json_content.strip():
                raise RuntimeError(f"JSON file {keras_model_json} is empty")
            # Test JSON parsing
            import json
            json.loads(json_content)
        print(f"✓ JSON file validated")
    except Exception as e:
        print(f"ERROR: Invalid JSON file {keras_model_json}: {e}")
        print(f"File size: {os.path.getsize(keras_model_json)} bytes")
        raise RuntimeError(f"JSON file validation failed: {e}")
    
    # Set random seeds BEFORE loading model (matching original script order)
    np.random.seed(113)
    random.seed(0)
    PERSISTENT_MODEL = model_from_json(open(keras_model_json).read(), custom_objects={'Functional': tf.keras.Model})
    PERSISTENT_MODEL.load_weights(keras_model_weights)
    
    # Initialize memory pool with hardcoded sequence length for maximum efficiency
    print("Initializing memory pool...")
    PERSISTENT_MEMORY_POOL = MemoryPool(max_seqs=PERSISTENT_NUM_SEQS, max_seq_length=PERSISTENT_SEQ_LEN, max_pca_components=PERSISTENT_N_PCA_COMPONENTS)
    
    init_time = time.time() - start_time
    print(f"Persistent resources initialized in {init_time:.2f} seconds")
    print("Model, data, and memory pool will be reused for all subsequent sequences")

def process_sequence(seq_index, task_index=0):
    """Main processing function for a single sequence using persistent resources"""

    # Start total timer
    total_start_time = time.time()
    
    from seam import Compiler, Attributer, Clusterer, MetaExplainer

    # In persistent mode, we DON'T clear the session to maintain model state
    # Only run garbage collection to free unused memory
    gc.collect()
    
    # Ensure persistent resources are available
    initialize_persistent_resources()

    # =============================================================================
    # Use global processing parameters
    # =============================================================================
    mut_rate = PERSISTENT_MUT_RATE
    num_seqs = PERSISTENT_NUM_SEQS
    n_clusters = PERSISTENT_N_CLUSTERS
    attribution_method = PERSISTENT_ATTRIBUTION_METHOD
    adaptive_background_scaling = PERSISTENT_ADAPTIVE_BACKGROUND_SCALING
    gpu = PERSISTENT_GPU  # Use global GPU detection

    # =============================================================================
    # Set up save paths for essential data only
    # =============================================================================
    py_dir = os.path.dirname(os.path.abspath(__file__))
    save_path_essential = os.path.join(py_dir, f'outputs_deepstarr_local_{attribution_method}')
    if not os.path.exists(save_path_essential):
        os.makedirs(save_path_essential)

    # =============================================================================
    # Use persistent model and data
    # =============================================================================
    # Use the persistent model instead of loading it again
    model = PERSISTENT_MODEL
    X_dataset = PERSISTENT_X_DATASET
    alphabet = PERSISTENT_ALPHABET

    x_ref = X_dataset[seq_index]
    x_ref = np.expand_dims(x_ref,0)

    # Define mutagenesis window for sequence
    seq_length = x_ref.shape[1]
    mut_window = [0, seq_length]  # [start_position, stop_position]

    # Forward pass to get output for the specific head
    output = model(x_ref)
    pred = model.predict(x_ref)[task_index]

    # =============================================================================
    # SQUID API
    # Create in silico mutagenesis library
    # =============================================================================
    
    # Set up predictor class for in silico MAVE
    pred_generator = squid.predictor.ScalarPredictor(
        pred_fun=model.predict_on_batch,
        task_idx=task_index,
        batch_size=512
    )

    # Set up mutagenizer class for in silico MAVE
    mut_generator = squid.mutagenizer.RandomMutagenesis(
        mut_rate=mut_rate,
        seed=42
    )

    # Generate in silico MAVE
    mave = squid.mave.InSilicoMAVE(
        mut_generator,
        pred_generator,
        seq_length,
        mut_window=mut_window
    )
    
    # Get pre-allocated arrays from memory pool
    x_mut, y_mut, attributions_pool, slots = PERSISTENT_MEMORY_POOL.get_arrays(num_seqs, seq_length)
    
    # Generate mutagenesis data into pre-allocated arrays
    x_mut_temp, y_mut_temp = mave.generate(x_ref[0], num_sim=num_seqs)
    
    # Copy data into pre-allocated arrays with optimized data types
    x_mut[:] = x_mut_temp.astype(np.int8)  # SQUID creates int8/uint8, convert to int8 for consistency
    y_mut[:] = y_mut_temp.astype(np.float16)  # Convert to float16 for memory efficiency

    # =============================================================================
    # SEAM API
    # Compile sequence analysis data into a standardized format
    # =============================================================================
    # Initialize compiler
    compiler = Compiler(
        x=x_mut,
        y=y_mut,
        x_ref=x_ref,
        y_bg=None,
        alphabet=alphabet,
        gpu=gpu
    )

    mave_df = compiler.compile()
    ref_index = 0 # index of reference sequence (zero by default)

    # =============================================================================
    # SEAM API
    # Compute attribution maps for each sequence in library
    # =============================================================================
    attributer = Attributer(
        model,
        method=attribution_method,
        task_index=task_index
    )

    # Compute attribution maps into pre-allocated arrays
    attributions = attributer.compute(
        x=x_mut,
        x_ref=x_ref,
        save_window=None,
        batch_size=32,
        gpu=gpu,
        num_steps=10
    )
    
    # Copy attributions into pre-allocated array with float16 optimization
    attributions_pool[:] = attributions.astype(np.float16)

    # =============================================================================
    # SEAM API
    # Cluster attribution maps using PCA + K-Means Clustering
    # =============================================================================
    
    if 1:  # PCA + K-Means clustering (current default)
        # Initialize clusterer with PCA method
        clusterer = Clusterer(
            attributions,
            method='pca',
            gpu=gpu
        )

        # Compute PCA embedding with specified number of components
        n_components = PERSISTENT_N_PCA_COMPONENTS
        
        # Get pre-allocated PCA embedding array from memory pool
        pca_embedding_pool = PERSISTENT_MEMORY_POOL.get_pca_embedding(num_seqs, n_components)
        
        # Compute PCA embedding into pre-allocated array
        pca_embedding_temp = clusterer.embed(
            n_components=n_components,
            plot_eigenvalues=False,  # No plotting in essential mode
            save_path=None
        )
        
        # Copy PCA embedding into pre-allocated array with float16 optimization
        pca_embedding_pool[:] = pca_embedding_temp.astype(np.float16)
        pca_embedding = pca_embedding_pool
        
        # Perform k-means clustering on PCA space using Clusterer's method
        # This will automatically use GPU k-means if gpu=True and cuML is available
        cluster_labels = clusterer.cluster(
            embedding=pca_embedding,
            method='kmeans',
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
                    
        # Calculate cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        # Ensure cluster labels are zero-indexed and consecutive (0, 1, 2, ...)
        # This matches how hierarchical clustering handles indexing
        unique_labels_sorted = np.sort(unique_labels)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels_sorted)}
        cluster_labels = np.array([label_map[label] for label in cluster_labels])
                
        # Store k-means results in clusterer for compatibility with SEAM API
        clusterer.cluster_labels = cluster_labels
        
        # Create a mock kmeans object for compatibility with SEAM API
        # Since we don't have centroids from the Clusterer's method, we'll create empty centroids
        class MockKMeans:
            def __init__(self, centroids, labels):
                self.cluster_centers_ = centroids
                self.labels_ = labels
        
        # Create dummy centroids (not used by SEAM but needed for compatibility)
        dummy_centroids = np.zeros((n_clusters, attributions.shape[1] * attributions.shape[2]))
        kmeans = MockKMeans(dummy_centroids, cluster_labels)
        clusterer.kmeans_model = kmeans
        
        # Create a mock linkage matrix for compatibility (not used in k-means)
        # This is just a placeholder to maintain SEAM API compatibility
        mock_linkage = np.zeros((n_clusters-1, 4))
        clusterer.linkage = mock_linkage
        
        # Set up membership dataframe that MetaExplainer expects
        clusterer.membership_df = pd.DataFrame({
            'Cluster': cluster_labels,
            'Cluster_Sorted': cluster_labels
        })

        # Get cluster labels (for k-means, these are already the final labels)
        labels_n = clusterer.cluster_labels
        
    else:  # Hierarchical clustering with Ward linkage
        # Initialize clusterer for hierarchical clustering
        clusterer = Clusterer(
            attributions,
            gpu=gpu
        )

        # Perform hierarchical clustering directly on attribution maps
        link_method = 'ward'
        linkage = clusterer.cluster(
            method='hierarchical',
            link_method=link_method
        )
        
        # Get cluster labels from hierarchical clustering
        labels_n, cut_level = clusterer.get_cluster_labels(
            linkage,
            criterion='maxclust',
            n_clusters=n_clusters
        )
        
        # Use the hierarchical clustering labels
        cluster_labels = labels_n
        
        # Store hierarchical clustering results in clusterer for compatibility with SEAM API
        clusterer.cluster_labels = cluster_labels
        clusterer.linkage = linkage
        
        # Set up membership dataframe that MetaExplainer expects
        clusterer.membership_df = pd.DataFrame({
            'Cluster': cluster_labels,
            'Cluster_Sorted': cluster_labels
        })

    # =============================================================================
    # SEAM API
    # Generate meta-explanations and related statistics
    # =============================================================================
    sort_method = 'median' # sort clusters by median DNN prediction (default)

    # Initialize MetaExplainer
    meta = MetaExplainer(
        clusterer=clusterer,
        mave_df=mave_df,
        attributions=attributions,
        sort_method=sort_method,
        ref_idx=0,
        mut_rate=mut_rate
    )

    # Generate Mechanism Summary Matrix (MSM) - essential for TFBS identification
    msm = meta.generate_msm(
        gpu=gpu
    )
    
    # Manually create Cluster_Sorted column if sort_method is specified
    if sort_method is not None and meta.cluster_order is not None:
        mapping_dict = {old_k: new_k for new_k, old_k in enumerate(meta.cluster_order)}
        meta.membership_df['Cluster_Sorted'] = meta.membership_df['Cluster'].map(mapping_dict)

    # =============================================================================
    # SEAM API
    # Background separation
    # =============================================================================
    # Separate individual clusters from average background over clusters
    background_multiplier = 0.5  # default threshold factor for background separation

    # Compute background for essential data (without creating logos)
    meta.compute_background(mut_rate, background_multiplier, adaptive_background_scaling, process_logos=False)

    # Save essential data for downstream genome-wide SEAM-based annotation
    # Determine reference cluster and calculate its cluster-averaged attribution matrix
    # Always look up the cluster containing the reference sequence
    if sort_method is not None:
        ref_cluster = meta.membership_df.loc[ref_index, 'Cluster_Sorted']
    else:
        ref_cluster = meta.membership_df.loc[ref_index, 'Cluster']
    
    # Calculate reference cluster-averaged attribution matrix
    ref_cluster_avg = np.mean(meta.get_cluster_maps(ref_cluster), axis=0).astype(np.float16)
    
    reference_attribution = attributions[ref_index].astype(np.float16) # only needed if requested in arrow file below
    
    # Find the cluster-averaged attribution matrix most similar to background
    # Get all cluster-averaged attribution matrices
    num_clusters = len(np.unique(cluster_labels))
    all_cluster_averages = []
    for cluster_idx in range(num_clusters):
        cluster_avg = np.mean(meta.get_cluster_maps(cluster_idx), axis=0)
        all_cluster_averages.append(cluster_avg)
    
    # Apply background scaling factors if available
    if adaptive_background_scaling and meta.background_scaling is not None:
        # Scale the background by each cluster's scaling factor for fair comparison
        scaled_backgrounds = []
        for i in range(len(all_cluster_averages)):
            if i < len(meta.background_scaling):
                scaling_factor = meta.background_scaling[i]
                scaled_background = meta.background * scaling_factor
                scaled_backgrounds.append(scaled_background)
            else:
                scaled_backgrounds.append(meta.background)
    else:
        scaled_backgrounds = [meta.background] * len(all_cluster_averages)
    
    # Calculate Euclidean distances to background
    distances = []
    for i, cluster_avg in enumerate(all_cluster_averages):
        scaled_background = scaled_backgrounds[i]
        distance = np.linalg.norm(cluster_avg - scaled_background)
        distances.append(distance)
    
    # Find the cluster with minimum distance
    closest_cluster_idx = np.argmin(distances)
    min_distance = distances[closest_cluster_idx]
    
    # Find the sequence in the closest cluster with median prediction value
    # Get all sequences in the closest cluster using SEAM's data structure
    cluster_seqs_df = meta.show_sequences(closest_cluster_idx)
    cluster_predictions = cluster_seqs_df['DNN'].values
    
    # Check if cluster has sequences
    if len(cluster_seqs_df) == 0:
        print(f"ERROR: Cluster {closest_cluster_idx} has no sequences!")
        return
    
    # Find the sequence with median prediction value
    median_prediction = np.median(cluster_predictions)
    # Sort predictions and find the actual median index
    sorted_indices = np.argsort(cluster_predictions)
    median_idx = sorted_indices[len(sorted_indices) // 2]  # True median index
    
    # Get the sequence and prediction at the median index
    background_sequence_row = cluster_seqs_df.iloc[median_idx]
    background_sequence_str = background_sequence_row['Sequence']
    background_prediction = float(background_sequence_row['DNN'])
    
    background_sequence = np.zeros((len(background_sequence_str), len(meta.alphabet)), dtype=np.int8)
    for pos, char in enumerate(background_sequence_str):
        char_idx = meta.alphabet.index(char)
        background_sequence[pos, char_idx] = 1
    
    background_float16 = meta.background.astype(np.float16)
    
    # Create Arrow table with all essential data at once
    # Convert arrays to bytes for PyArrow compatibility
    if 1:  # Without reference attribution (in case it's already saved in an external file)
        table = pa.table({
            'reference_cluster_average': [ref_cluster_avg.tobytes()],
            'average_background': [background_float16.tobytes()], 
            'background_sequence_onehot': [background_sequence.tobytes()],
            'msm_data': [meta.msm.to_dict('records')],
            'array_shapes': [str(ref_cluster_avg.shape) + '|' + str(background_float16.shape) + '|' + str(background_sequence.shape)],
            'array_dtypes': [str(ref_cluster_avg.dtype) + '|' + str(background_float16.dtype) + '|' + str(background_sequence.dtype)],
            'cluster_order': [meta.cluster_order.tolist() if meta.cluster_order is not None else None],
            'sort_method': [sort_method],
            'reference_cluster_index': [ref_cluster]
        })
        
        # Create new schema with metadata
        metadata = {
            b'seq_index': str(seq_index).encode(),
            b'task_index': str(task_index).encode(),
            b'description': b'SEAM essential data: ref_cluster_avg_bytes, avg_bg_bytes, bg_seq_onehot_bytes, msm, array_shapes, ref_cluster_idx'
        }
    else:  # Version with reference attribution included
        table = pa.table({
            'reference_attribution': [reference_attribution.tobytes()],
            'reference_cluster_average': [ref_cluster_avg.tobytes()],
            'average_background': [background_float16.tobytes()], 
            'background_sequence_onehot': [background_sequence.tobytes()],
            'msm_data': [meta.msm.to_dict('records')],
            'array_shapes': [str(reference_attribution.shape) + '|' + str(ref_cluster_avg.shape) + '|' + str(background_float16.shape) + '|' + str(background_sequence.shape)],
            'array_dtypes': [str(reference_attribution.dtype) + '|' + str(ref_cluster_avg.dtype) + '|' + str(background_float16.dtype) + '|' + str(background_sequence.dtype)],
            'cluster_order': [meta.cluster_order.tolist() if meta.cluster_order is not None else None],
            'sort_method': [sort_method],
            'reference_cluster_index': [ref_cluster]
        })
        
        # Create new schema with metadata
        metadata = {
            b'seq_index': str(seq_index).encode(),
            b'task_index': str(task_index).encode(),
            b'description': b'SEAM essential data: ref_attribution_bytes, ref_cluster_avg_bytes, avg_bg_bytes, bg_seq_onehot_bytes, msm, array_shapes, ref_cluster_idx'
        }
    table = table.replace_schema_metadata(metadata)
    
    # Save Arrow file with compression
    filename = f'seq{seq_index}_task{task_index}.arrow'
    filepath = os.path.join(save_path_essential, filename)
    feather.write_feather(table, filepath, compression='lz4')
    print(f"Saved essential data to: {filepath}")

    # =============================================================================
    # Total execution time
    # =============================================================================
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"TOTAL EXECUTION TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*60}")
    
    # Always release memory pool arrays, even if processing fails
    PERSISTENT_MEMORY_POOL.release_arrays(slots)
        
    # Optional: Print memory pool status for debugging
    # pool_status = PERSISTENT_MEMORY_POOL.get_pool_status()
    # print(f"Memory pool status: {pool_status}")

    return

if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) > 3:
        # Range mode: start_index, stop_index, task_index
        start_index = int(sys.argv[1])
        stop_index = int(sys.argv[2])
        task_index = int(sys.argv[3])
        range_mode = True
    elif len(sys.argv) > 2:
        # Single sequence mode: seq_index, task_index
        start_index = stop_index = int(sys.argv[1])
        task_index = int(sys.argv[2])
        range_mode = False
    else:
        print('')
        print('Script must be run with either:')
        print('  Single sequence: python example_deepstarr_local_batch_persistent.py <seq_index> <task_index>')
        print('  Range mode: python example_deepstarr_local_batch_persistent.py <start_index> <stop_index> <task_index>')
        print('where task_index: 0=Dev, 1=Hk')
        print('')
        sys.exit(0)
    
    # Initialize persistent resources once for the entire batch
    print(f"Initializing persistent resources for batch processing...")
    initialize_persistent_resources()
    
    # Process range of sequences
    total_sequences = stop_index - start_index + 1
    successful_sequences = 0
    failed_sequences = 0
    
    print(f"Processing sequences {start_index} to {stop_index} (task {task_index})")
    print(f"Total sequences to process: {total_sequences}")
    print(f"{'='*60}")
    
    batch_start_time = time.time()
    
    for seq_index in range(start_index, stop_index + 1):
        sequence_start_time = time.time()
        print(f"\nProcessing sequence {seq_index} ({seq_index - start_index + 1}/{total_sequences})")
        
        try:
            process_sequence(seq_index, task_index)
            successful_sequences += 1
            sequence_time = time.time() - sequence_start_time
            print(f"✓ Sequence {seq_index} completed successfully in {sequence_time:.2f}s")
            
        except Exception as e:
            failed_sequences += 1
            sequence_time = time.time() - sequence_start_time
            print(f"✗ Error processing sequence {seq_index} after {sequence_time:.2f}s: {e}")
            print(f"Continuing with next sequence...")
            continue
    
    # Batch completion summary
    batch_time = time.time() - batch_start_time
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {batch_time:.2f}s ({batch_time/60:.2f} minutes)")
    print(f"Successful sequences: {successful_sequences}/{total_sequences}")
    print(f"Failed sequences: {failed_sequences}/{total_sequences}")
    print(f"Average time per sequence: {batch_time/total_sequences:.2f}s")
    if successful_sequences > 0:
        print(f"Average time per successful sequence: {batch_time/successful_sequences:.2f}s")
    print(f"{'='*60}")


# =============================================================================
# ARROW FILE LOADING UTILITIES
# Minimalistic code to load saved Arrow files and extract essential data
# =============================================================================

def load_arrow_data(filepath):
    """
    Load essential data from a saved Arrow file.
    
    Args:
        filepath: Path to the Arrow file
        
    Returns:
        dict: Dictionary containing all essential data arrays and metadata
    """
    print(f"Loading data from: {filepath}")
    
    # Read Arrow file using PyArrow's native reader to preserve metadata
    try:
        with pa.ipc.open_file(filepath) as reader:
            table = reader.read_all()
    except:
        # Fallback to feather reader
        table = feather.read_feather(filepath)
        # If it's a DataFrame, convert to Table
        if hasattr(table, 'to_arrow'):
            table = table.to_arrow()
    
    # Extract metadata
    metadata = table.schema.metadata
    seq_index = metadata[b'seq_index'].decode()
    task_index = metadata[b'task_index'].decode()
    description = metadata[b'description'].decode()
    
    print(f"Sequence index: {seq_index}")
    print(f"Task index: {task_index}")
    print(f"Description: {description}")
    
    # Convert to pandas for easier data access
    df = table.to_pandas()
    
    # Parse array shapes and dtypes
    shapes_str = df['array_shapes'][0]
    shapes = shapes_str.split('|')
    ref_cluster_shape = eval(shapes[0])
    background_shape = eval(shapes[1])
    bg_seq_shape = eval(shapes[2])
    
    # Parse array dtypes (with fallback for old files)
    try:
        dtypes_str = df['array_dtypes'][0]
        dtypes = dtypes_str.split('|')
        ref_cluster_dtype = np.dtype(dtypes[0])
        background_dtype = np.dtype(dtypes[1])
        bg_seq_dtype = np.dtype(dtypes[2])
        print("Using stored dtypes from Arrow file")
    except (KeyError, IndexError):
        # Fallback for old Arrow files without dtype information
        print("No dtype information found in Arrow file, using default dtypes...")
        # Use the correct dtypes based on what the example script saves
        ref_cluster_dtype = np.float16
        background_dtype = np.float16
        bg_seq_dtype = np.int8
    
    print(f"Array shapes: ref_cluster={ref_cluster_shape}, background={background_shape}, bg_seq={bg_seq_shape}")
    print(f"Array dtypes: ref_cluster={ref_cluster_dtype}, background={background_dtype}, bg_seq={bg_seq_dtype}")
    
    # Extract bytes
    ref_cluster_bytes = df['reference_cluster_average'][0]
    background_bytes = df['average_background'][0]
    bg_seq_bytes = df['background_sequence_onehot'][0]
    
    # Debug: print byte sizes to understand the data type
    print(f"Byte sizes: ref_cluster={len(ref_cluster_bytes)}, background={len(background_bytes)}, bg_seq={len(bg_seq_bytes)}")
    
    # Reconstruct arrays using the stored dtypes
    ref_cluster_avg = np.frombuffer(ref_cluster_bytes, dtype=ref_cluster_dtype).reshape(ref_cluster_shape)
    background = np.frombuffer(background_bytes, dtype=background_dtype).reshape(background_shape)
    background_sequence = np.frombuffer(bg_seq_bytes, dtype=bg_seq_dtype).reshape(bg_seq_shape)
    
    print(f"ref_cluster_avg: min={np.min(ref_cluster_avg):.6f}, max={np.max(ref_cluster_avg):.6f}, mean={np.mean(ref_cluster_avg):.6f}")
    print(f"background: min={np.min(background):.6f}, max={np.max(background):.6f}, mean={np.mean(background):.6f}")
    print(f"background_sequence: min={np.min(background_sequence):.6f}, max={np.max(background_sequence):.6f}, mean={np.mean(background_sequence):.6f}")
    
    # Load MSM data
    msm_data = df['msm_data'][0]
    # Unwrap the list of dictionaries into a proper DataFrame
    # Use json_normalize to properly expand the dictionaries
    msm_df = pd.json_normalize(msm_data)
    print(f"MSM data loaded: {len(msm_data)} records")
    print(f"MSM DataFrame columns after normalization: {list(msm_df.columns)}")
    
    # Load sorting information
    cluster_order = df['cluster_order'][0] if 'cluster_order' in df.columns else None
    sort_method = df['sort_method'][0] if 'sort_method' in df.columns else None
    reference_cluster_index = df['reference_cluster_index'][0] if 'reference_cluster_index' in df.columns else None
    print(f"Sort method: {sort_method}")
    print(f"Cluster order: {cluster_order}")
    print(f"Reference cluster index: {reference_cluster_index}")
    
    return {
        'seq_index': seq_index,
        'task_index': task_index,
        'ref_cluster_avg': ref_cluster_avg,
        'background': background,
        'background_sequence': background_sequence,
        'msm_df': msm_df,
        'cluster_order': cluster_order,
        'sort_method': sort_method,
        'reference_cluster_index': reference_cluster_index,
        'shapes': {
            'ref_cluster': ref_cluster_shape,
            'background': background_shape,
            'bg_seq': bg_seq_shape
        }
    }


def load_multiple_arrow_files(output_dir, pattern="seq*_task*.arrow"):
    """
    Load multiple Arrow files from a directory.
    
    Args:
        output_dir: Directory containing Arrow files
        pattern: Glob pattern to match Arrow files
        
    Returns:
        list: List of dictionaries containing loaded data
    """
    import glob
    
    arrow_files = glob.glob(os.path.join(output_dir, pattern))
    loaded_data = []
    
    print(f"Found {len(arrow_files)} Arrow files matching pattern '{pattern}'")
    
    for filepath in arrow_files:
        try:
            data = load_arrow_data(filepath)
            loaded_data.append(data)
            print(f"✓ Loaded: {os.path.basename(filepath)} (seq {data['seq_index']}, task {data['task_index']})")
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Successfully loaded {len(loaded_data)}/{len(arrow_files)} Arrow files")
    return loaded_data


def extract_essential_summary(loaded_data):
    """
    Extract a summary of essential data across multiple sequences.
    
    Args:
        loaded_data: List of loaded data dictionaries
        
    Returns:
        dict: Summary statistics and aggregated data
    """
    if not loaded_data:
        return {}
    
    # Aggregate reference cluster averages
    ref_cluster_avgs = [data['ref_cluster_avg'] for data in loaded_data]
    
    # Aggregate backgrounds
    backgrounds = [data['background'] for data in loaded_data]
    
    # Extract sequence indices and task indices
    seq_indices = [data['seq_index'] for data in loaded_data]
    task_indices = [data['task_index'] for data in loaded_data]
    
    # Get unique tasks
    unique_tasks = list(set(task_indices))
    
    # Aggregate MSM data
    msm_dfs = [data['msm_df'] for data in loaded_data]
    
    # Get shape information
    shapes_info = [data['shapes'] for data in loaded_data]
    
    # Extract reference cluster indices
    reference_cluster_indices = [data['reference_cluster_index'] for data in loaded_data]
    
    print(f"Summary: {len(loaded_data)} sequences across tasks {unique_tasks}")
    print(f"Sequence indices: {seq_indices}")
    print(f"MSM data: {len(msm_dfs)} DataFrames")
    print(f"Reference cluster indices: {reference_cluster_indices}")
    
    return {
        'num_sequences': len(loaded_data),
        'sequence_indices': seq_indices,
        'unique_tasks': unique_tasks,
        'reference_cluster_averages': ref_cluster_avgs,
        'backgrounds': backgrounds,
        'msm_dataframes': msm_dfs,
        'shapes_info': shapes_info,
        'reference_cluster_indices': reference_cluster_indices,
        'loaded_data': loaded_data
    }


# Example usage (uncomment to test):
# if __name__ == '__main__':
#     # Load a single Arrow file
#     # data = load_arrow_data('outputs_deepstarr_local_intgrad/seq463513_task1.arrow')
#     # print(f"Loaded data for sequence {data['seq_index']}, task {data['task_index']}")
#     # print(f"Reference cluster average shape: {data['ref_cluster_avg'].shape}")
#     # print(f"MSM DataFrame shape: {data['msm_df'].shape}")
#     # print(f"Reference cluster index: {data['reference_cluster_index']}")
#     
#     # Load multiple Arrow files
#     # all_data = load_multiple_arrow_files('outputs_deepstarr_local_intgrad')
#     # summary = extract_essential_summary(all_data)
#     # print(f"Loaded {summary['num_sequences']} sequences across tasks {summary['unique_tasks']}")
#     # print(f"MSM DataFrames: {len(summary['msm_dataframes'])}")
#     # print(f"Reference cluster indices: {summary['reference_cluster_indices']}")