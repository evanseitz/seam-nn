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
    
    # Generate random test data with sequence dimensions
    np.random.seed(0)
    X = np.random.randn(5, 5, 4)  # (batch, seq_length, nucleotides)
    background = np.random.randn(3, 5, 4)
    
    # Define equivalent model architectures
    class PyTorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = torch.nn.Linear(4, 20)  # Input nucleotides dimension
            self.relu = torch.nn.ReLU()
            self.dense2 = torch.nn.Linear(20, 2)
            self.final = torch.nn.Linear(5, 1)  # Reduce sequence dimension to scalar
            
        def forward(self, x):
            x = self.dense1(x)
            x = self.relu(x)
            x = self.dense2(x)  # Shape: (batch, seq_length, 2)
            x = x.transpose(1, 2)  # Shape: (batch, 2, seq_length)
            return self.final(x).squeeze(-1)  # Shape: (batch, 2)
    
    def create_tf_model():
        return tf.keras.Sequential([
            tf.keras.layers.Dense(20, input_shape=(5, 4)),  # Input (seq_length, nucleotides)
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(2),
            tf.keras.layers.Permute((2, 1)),  # Match PyTorch transpose
            tf.keras.layers.Dense(1),  # Match PyTorch final layer
            tf.keras.layers.Flatten()  # Get final shape (batch, 2)
        ])
    
    # Create models with matching input/output dimensions
    pytorch_model = PyTorchModel()
    tf_model = create_tf_model()
    
    # Copy weights from PyTorch to TF to ensure identical starting points
    with torch.no_grad():
        # Copy first dense layer weights
        #print("\n=== Weight Comparison ===")
        #print("PyTorch dense1 weight shape:", pytorch_model.dense1.weight.shape)
        #print("PyTorch dense1 bias shape:", pytorch_model.dense1.bias.shape)
        #print("TF dense1 weight shape:", tf_model.layers[0].weights[0].shape)
        #print("TF dense1 bias shape:", tf_model.layers[0].weights[1].shape)
        
        tf_model.layers[0].set_weights([
            pytorch_model.dense1.weight.numpy().T,
            pytorch_model.dense1.bias.numpy()
        ])
        
        # Copy second dense layer weights
        #print("\nPyTorch dense2 weight shape:", pytorch_model.dense2.weight.shape)
        #print("PyTorch dense2 bias shape:", pytorch_model.dense2.bias.shape)
        #print("TF dense2 weight shape:", tf_model.layers[2].weights[0].shape)
        #print("TF dense2 bias shape:", tf_model.layers[2].weights[1].shape)
        
        tf_model.layers[2].set_weights([
            pytorch_model.dense2.weight.numpy().T,
            pytorch_model.dense2.bias.numpy()
        ])
        
        # Copy final layer weights
        #print("\nPyTorch final weight shape:", pytorch_model.final.weight.shape)
        #print("PyTorch final bias shape:", pytorch_model.final.bias.shape)
        #print("TF final weight shape:", tf_model.layers[4].weights[0].shape)
        #print("TF final bias shape:", tf_model.layers[4].weights[1].shape)
        
        tf_model.layers[4].set_weights([
            pytorch_model.final.weight.numpy().T,
            pytorch_model.final.bias.numpy()
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
    print("PyTorch SHAP type:", type(pytorch_shap))
    print("PyTorch SHAP shape:", pytorch_shap.shape if not isinstance(pytorch_shap, list) else [v.shape for v in pytorch_shap])
    print(pytorch_shap)

    print("\n=== Computing TF SHAP values ===")
    tf_shap = tf_explainer.shap_values(tf_X)
    print("TF SHAP type:", type(tf_shap))
    print("TF SHAP shape:", tf_shap.shape if not isinstance(tf_shap, list) else [v.shape for v in tf_shap])
    print(tf_shap)

    # Convert to numpy for comparison
    if isinstance(pytorch_shap, list):
        pytorch_shap = [v.numpy() if torch.is_tensor(v) else v for v in pytorch_shap]
    print("PyTorch SHAP after numpy conversion:", type(pytorch_shap))
    print("PyTorch SHAP shape after numpy:", [v.shape for v in pytorch_shap] if isinstance(pytorch_shap, list) else pytorch_shap.shape)

    if isinstance(tf_shap, list):
        tf_shap = [v.numpy() if tf.is_tensor(v) else v for v in tf_shap]
    print("TF SHAP after numpy conversion:", type(tf_shap))
    print("TF SHAP shape after numpy:", [v.shape for v in tf_shap] if isinstance(tf_shap, list) else tf_shap.shape)
    
    # Compare results
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