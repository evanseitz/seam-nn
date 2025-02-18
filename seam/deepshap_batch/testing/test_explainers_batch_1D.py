import sys
import os

sys.dont_write_bytecode = True
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import torch
import tensorflow as tf
from deep.deep_tf_batch import TF2DeepExplainer
from deep.deep_torch import PyTorchDeepExplainer
import matplotlib.pyplot as plt


def test_deep_explainer_parity():
    """Test parity between PyTorch and TF2 DeepExplainer implementations."""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    tf.random.set_seed(42)
    
    # Generate random test data
    np.random.seed(0)
    X = np.random.randn(5, 5)
    background = np.random.randn(3, 5)
    
    # Define equivalent model architectures
    class PyTorchModel(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.dense1 = torch.nn.Linear(input_dim, 20)
            self.relu = torch.nn.ReLU()
            self.dense2 = torch.nn.Linear(20, 2)
            
        def forward(self, x):
            x = self.dense1(x)
            x = self.relu(x)
            return self.dense2(x)
    
    def create_tf_model(output_dim):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(20, input_shape=(5,)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_dim)
        ])
    
    # Create models with matching input/output dimensions
    pytorch_model = PyTorchModel(5)
    tf_model = create_tf_model(2)
    
    # Copy weights from PyTorch to TF to ensure identical starting points
    with torch.no_grad():
        # Copy first dense layer weights
        tf_model.layers[0].set_weights([
            pytorch_model.dense1.weight.numpy().T,
            pytorch_model.dense1.bias.numpy()
        ])
        # Copy second dense layer weights
        tf_model.layers[2].set_weights([
            pytorch_model.dense2.weight.numpy().T,
            pytorch_model.dense2.bias.numpy()
        ])
    
    # Convert data to framework-specific formats with explicit types
    torch_X = torch.tensor(X, dtype=torch.float32)
    torch_background = torch.tensor(background, dtype=torch.float32)
    tf_X = tf.convert_to_tensor(X, dtype=tf.float32)
    tf_background = tf.convert_to_tensor(background, dtype=tf.float32)
    
    # Create explainers
    output_idx = 0  # choose output head to explain

    class TorchWrapper(torch.nn.Module):
        def __init__(self, model, output_idx):
            super().__init__()
            self.model = model
            self.output_idx = output_idx
            
        def forward(self, x):
            return self.model(x)[:, self.output_idx:self.output_idx+1]
        
    pytorch_model_output_idx = TorchWrapper(pytorch_model, output_idx=0)
    pytorch_explainer = PyTorchDeepExplainer(pytorch_model_output_idx, torch_background)

    tf_model_output_idx = tf.keras.Model(inputs=tf_model.input, outputs=tf_model.output[:,output_idx])
    tf_explainer = TF2DeepExplainer(tf_model_output_idx, tf_background)
    
    # Before computing SHAP values
    print("\n=== Model Output Comparison ===")
    with torch.no_grad():
        torch_out = pytorch_model(torch_X)
        print(f"PyTorch output shape: {torch_out.shape}")
        print(f"PyTorch output mean: {torch_out.mean().item():.6f}")
    
    tf_out = tf_model(tf_X)
    print(f"TF output shape: {tf_out.shape}")
    print(f"TF output mean: {tf.reduce_mean(tf_out).numpy():.6f}")
    
    # Compute SHAP values
    print("\n=== Computing PyTorch SHAP values ===")
    pytorch_shap = pytorch_explainer.shap_values(torch_X)
    print(pytorch_shap)

    print("\n=== Computing TF SHAP values ===")
    tf_shap = tf_explainer.shap_values(tf_X)
    print(tf_shap)

    # Convert to numpy for comparison
    if isinstance(pytorch_shap, list):
        pytorch_shap = [v.numpy() if torch.is_tensor(v) else v for v in pytorch_shap]
    if isinstance(tf_shap, list):
        tf_shap = [v.numpy() if tf.is_tensor(v) else v for v in tf_shap]
    
    # Compare results with visualization
    print("\n=== Testing SHAP values parity ===")
    for i, (pytorch_vals, tf_vals) in enumerate(zip(pytorch_shap, tf_shap)):
        diff = np.abs(pytorch_vals - tf_vals)
        mean_diff = diff.mean()
        max_diff = diff.max()
        print(f"Output {i}:")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")


if __name__ == "__main__":
    test_deep_explainer_parity()

'''
# Create new environment
conda create -n explainer python=3.8 -y
conda activate explainer

# Install packages one by one to ensure compatibility
pip install typing-extensions>=4.8.0
pip install numpy==1.24.3
pip install torch==2.1.0  # Older version that might be more compatible
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal
pip install shap
pip install matplotlib

# Verify installations
python -c "import torch; import tensorflow as tf; import shap; print(f'PyTorch: {torch.__version__}\nTensorFlow: {tf.__version__}\nSHAP: {shap.__version__}')"
'''