import numpy as np

# Make cupy import optional
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

from functools import wraps
from logomaker_batch.matrix import ALPHABET_DICT
from logomaker_batch.colors import get_rgb

"""
GPUTransformer: A utility class designed to accelerate path transformations using GPU.

This class is currently a placeholder for future GPU acceleration of batch logo operations.
While the current BatchLogo implementation achieves significant speedup through CPU
optimizations (path caching, efficient transformations, minimal object creation),
further performance gains could be achieved through GPU acceleration of matrix operations.

Future Implementation Plans:
- Use TensorFlow to batch-process path vertex transformations
- Accelerate matrix operations for character scaling and positioning
- Handle multiple logos simultaneously on GPU
- Process large batches of transformations in parallel

Current Status:
- Class checks for GPU availability and TensorFlow installation
- Provides framework for future GPU-accelerated batch operations
- Falls back to CPU processing if GPU/TensorFlow unavailable
- Not currently integrated with BatchLogo class

Example Future Usage:
    transformer = GPUTransformer()
    if transformer.use_gpu:
        # Batch process multiple transformations at once
        transformed_vertices = transformer.batch_transform_vertices(
            vertices_list,    # List of character vertices
            transforms_list,  # List of transformation parameters
            batch_size=1000   # Process in batches of 1000
        )
"""

class GPUTransformer:
    def __init__(self):
        """Initialize GPU transformer and check hardware availability"""
        self.tf = None
        self.use_gpu = False
        try:
            import tensorflow as tf
            self.tf = tf
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                self.use_gpu = True
                # Allow memory growth for the GPU
                try:
                    for device in physical_devices:
                        tf.config.experimental.set_memory_growth(device, True)
                    print("GPU acceleration enabled")
                except RuntimeError as e:
                    print(f"Warning: {e}")
                    print("Continuing with default GPU memory settings")
                    self.use_gpu = True
            else:
                print("Warning: No GPU devices found, falling back to CPU")
        except ImportError:
            print("Warning: TensorFlow not installed, falling back to CPU")

        if not HAS_CUDA:
            print("CUDA not available. Using CPU fallback.")
            self.device = 'cpu'
        else:
            self.device = 'gpu'

    def batch_transform_vertices(self, vertices_list, transforms_list, batch_size=1000):
        """
        Batch process multiple glyph transformations at once
        
        Parameters:
        -----------
        vertices_list: list of np.ndarray
            List of vertex arrays for each glyph
        transforms_list: list of dict
            List of transformation parameters for each glyph
            Each dict contains: width, height, x_center, y_center
            
        Returns:
        --------
        list of np.ndarray
            Transformed vertices for each glyph
        """
        if not self.use_gpu:
            return None

        if len(vertices_list) > batch_size:
            # Process in batches
            results = []
            for i in range(0, len(vertices_list), batch_size):
                batch_vertices = vertices_list[i:i + batch_size]
                batch_transforms = transforms_list[i:i + batch_size]
                batch_results = self._process_batch(batch_vertices, batch_transforms)
                results.extend(batch_results)
            return results
        else:
            return self._process_batch(vertices_list, transforms_list)

    def _process_batch(self, vertices_list, transforms_list):
        """Process a batch of vertices with their transformations
        
        Parameters:
        -----------
        vertices_list: list of np.ndarray
            List of vertex arrays for each glyph
        transforms_list: list of dict
            List of transformation parameters for each glyph
            
        Returns:
        --------
        list of np.ndarray
            Transformed vertices for each glyph
        """
        # Pad all vertex arrays to same size
        max_vertices = max(v.shape[0] for v in vertices_list)
        padded_vertices = []
        mask = []
        
        for vertices in vertices_list:
            padding = max_vertices - vertices.shape[0]
            padded = np.pad(vertices, ((0, padding), (0, 0)), mode='constant')
            padded_vertices.append(padded)
            mask.append(np.array([1]*vertices.shape[0] + [0]*padding))
            
        # Convert to tensors
        vertices_tensor = self.tf.convert_to_tensor(padded_vertices, dtype=self.tf.float32)
        mask_tensor = self.tf.convert_to_tensor(mask, dtype=self.tf.float32)
        
        # Create transformation matrices
        transforms = []
        for t in transforms_list:
            matrix = [
                [t['width'], 0, t['x_center']],
                [0, t['height'], t['y_center']],
                [0, 0, 1]
            ]
            transforms.append(matrix)

            transforms_tensor = self.tf.convert_to_tensor(transforms, dtype=self.tf.float32)
        
        # Add homogeneous coordinate
        ones = self.tf.ones([vertices_tensor.shape[0], vertices_tensor.shape[1], 1])
        vertices_homog = self.tf.concat([vertices_tensor, ones], axis=2)
        
        # Apply transformations
        transformed = self.tf.matmul(vertices_homog, transforms_tensor, transpose_b=True)
        transformed = transformed[:, :, :2] * mask_tensor[..., None]
        result = transformed.numpy()
        
        # Unpad results
        return [res[:int(m.sum())] for res, m in zip(result, mask)]

    def transform(self, data):
        if self.device == 'gpu':
            return cp.array(data)
        return np.array(data)

def handle_gpu_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except tf.errors.ResourceExhaustedError:
            print("GPU memory exhausted, falling back to CPU")
            return None
        except Exception as e:
            print(f"GPU error: {str(e)}, falling back to CPU")
            return None
    return wrapper 