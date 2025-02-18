"""
SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model.
"""

from deep.deep_tf_batch import TF2DeepExplainer
from deep.deep_torch import PyTorchDeepExplainer
from deep import DeepExplainer

__all__ = [
    'DeepExplainer',
    'TF2DeepExplainer',
    'PyTorchDeepExplainer'
]