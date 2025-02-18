Optimized DeepSHAP Implementation
================================

# DeepSTARR SHAP Implementation

This repository contains an optimized TensorFlow implementation of DeepSHAP, adapted from the Kundaje lab's implementation ([kundajelab_shap_29d2ffa](https://github.com/kundajelab/shap/tree/29d2ffa)).

## Implementation Overview

This repository modernizes and optimizes the original TensorFlow implementation (`deep_tf_sess.py` from Kundaje lab's repository) for current TensorFlow versions, while maintaining the PyTorch implementation (`deep_pytorch.py`).

### Key Versions

1. `deep_tf_sess.py` (Previously `deep_tf.py` in Kundaje lab repo)
   - Original TF1 (and early TF2) implementation
   - Session-based, graph-mode execution
   - Limited compatibility with modern TF2 versions

3. `deep_tf_batch.py` (Current optimized version)
   - Modern TF2 implementation
   - Efficient batch processing
   - Memory optimizations

### Major Implementation Changes

#### From TF1 Session-based to TF2
- Removed session handling and graph-based operations
- Switched to eager execution model
- Replaced tf.gradients with GradientTape
- Removed learning phase flags handling
- Simplified operator handling
- Removed symbolic computation graphs
- Added batch processing capabilities

#### From Non-batched to Batched Version
- Added efficient batch processing of samples
- Implemented background tiling using tf.repeat
- Added persistent gradient tape for batch computation
- Optimized tensor operations with reshape and einsum
- Added memory management with explicit cleanup
- Improved tensor shape handling for batches

### Current Batch Processing Features
- Processes N samples against M backgrounds efficiently
- Uses tiling/repeating for parallel computation
- Shares background data across all samples in batch
- Manages memory with explicit tensor cleanup
- Uses persistent gradient tape for gradient computation
- Employs einsum for efficient batch multiplication

### Additional Information

For a comprehensive list of potential optimizations and detailed implementation notes, see the documentation section at the bottom of `deep_tf_batch.py`.

### Credits
- Adapted code: Kundaje lab
- Current optimization: Evan Seitz, 2025
- Additional optimizations: Implemented with Claude AI assistance