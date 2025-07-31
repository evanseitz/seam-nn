# SEAM Examples

This folder contains example scripts demonstrating how to use SEAM with different models and use cases.

## Script Categories

### 1. Individual Locus Analysis Scripts (Non-Persistent)

These scripts are designed for detailed analysis of individual loci with comprehensive outputs and visualizations:

- **`example_deepstarr_local_20647.py`** - DeepSTARR enhancer analysis with full SEAM pipeline
- **`example_chrombpnet_local_ppif.py`** - ChromBPNet analysis for PPIF foreground/background signals
- **`example_pbm_combinatorial_zfp187.py`** - PBM combinatorial sequence library analysis

#### Features:
- **Comprehensive Outputs**: Generate and save extensive SEAM outputs including:
  - Multiple sequence logo visualizations (PNG format)
  - Hierarchical clustering dendrograms
  - Attribution maps and cluster analyses
  - Quantitative figures and plots
  - Detailed CSV data files with mutation analysis
  - NumPy arrays for intermediate computations

- **Multiple Checkpoints**: Support for loading/saving intermediate results:
  - Mutated sequence libraries (`x_mut.npy`, `y_mut.npy`)
  - Attribution maps (`attributions_*.npy`)
  - Hierarchical clustering linkage matrices
  - Previous analysis results

- **Rich Visualization Options**:
  - Sequence logos for reference, clusters, and backgrounds
  - Variability logos showing sequence diversity
  - Additive and epistatic parameter visualizations
  - Customizable DPI and output formats

- **Flexible Configuration**: Extensive user settings for:
  - Model parameters and attribution methods
  - Output control (figures, logos, data)
  - GPU/CPU processing options
  - Memory management and cleanup

### 2. Genome-Wide Batch Processing Script (Persistent)

**`example_deepstarr_local_batch_persistent.py`** - Optimized for efficient processing of many loci across the genome.

#### Features:
- **GPU Memory Optimization**: 
  - Persistent TensorFlow worker that loads model once and keeps it in GPU memory
  - Memory pooling with pre-allocated arrays to reduce fragmentation
  - Optimized data types (float16, int8) for maximum efficiency

- **Essential Data Only**: 
  - Saves only core SEAM outputs needed for downstream genome-wide annotation
  - Compressed Arrow format files with essential data:
    - Reference cluster averages
    - Background sequences and predictions
    - MSM (Mutation Sensitivity Matrix) data
    - Cluster ordering information
  - No figure generation or logo rendering

- **Batch Processing Capabilities**:
  - Process ranges of sequences efficiently (start_index to stop_index)
  - Fault-tolerant: continues processing if individual sequences fail
  - Manual multi-GPU support: run separate processes for different sequence ranges
  - Command-line interface for batch execution

- **Performance Optimizations**:
  - Eliminates repeated model and data loading overhead
  - Maintains warm GPU state between sequences
  - Memory-efficient array lifecycle management
  - GPU-accelerated algorithms when available

## Usage Scenarios

### Choose Individual Locus Scripts When:
- Analyzing specific genomic regions in detail
- Generating publication-quality figures and visualizations
- Exploring SEAM methodology and outputs
- Debugging or developing new SEAM features
- Need comprehensive intermediate data for further analysis

### Choose Persistent Script When:
- Processing numerous loci across the genome
- Running SEAM in production pipelines
- Need maximum computational efficiency
- Only require essential SEAM outputs for downstream analysis
- Working with limited storage space

## Command Line Usage

### Individual Locus Scripts:
```bash
python example_deepstarr_local_20647.py
python example_chrombpnet_local_ppif.py
python example_pbm_combinatorial_zfp187.py
```

### Persistent Batch Script:
```bash
# Single sequence mode
python example_deepstarr_local_batch_persistent.py <seq_index> <task_index>

# Range mode for batch processing
python example_deepstarr_local_batch_persistent.py <start_index> <stop_index> <task_index>

# Multi-GPU example
CUDA_VISIBLE_DEVICES=0 python example_deepstarr_local_batch_persistent.py 0 99999 0
CUDA_VISIBLE_DEVICES=1 python example_deepstarr_local_batch_persistent.py 100000 199999 0
```

## Output Formats

### Individual Locus Scripts:
- **Visualizations**: PNG files for sequence logos, dendrograms, and plots
- **Data Files**: NumPy arrays (.npy), CSV files, HDF5 files
- **Comprehensive**: All intermediate and final SEAM outputs

### Persistent Script:
- **Essential Data**: Compressed Arrow files (.arrow) with core SEAM data
- **Minimal**: Only data needed for downstream genome-wide annotation
- **Efficient**: Optimized for storage and I/O performance

## Requirements

### Individual Locus Scripts:
- Standard SEAM dependencies
- Matplotlib for visualizations
- Optional: GPU acceleration

### Persistent Script:
- **GPU Required**: Will fail immediately if no GPU is detected
- TensorFlow with GPU support
- Additional: `squid-nn`, `seam-nn`, `pyarrow`
- Optional: `cuml`, `kmeanstf`, `cupy` for enhanced GPU acceleration 