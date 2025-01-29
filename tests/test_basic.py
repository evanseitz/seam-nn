import pytest
import numpy as np
from seam import MetaExplainer, Compiler, Attributer

def test_compiler_initialization():
    x = np.random.random((10, 100, 4))  # Example sequence data
    y = np.random.random((10, 1))       # Example predictions
    compiler = Compiler(x=x, y=y)
    assert compiler is not None

def test_attributer_initialization():
    attributer = Attributer(None, method='saliency')  # Mock model
    assert attributer.method == 'saliency'
    assert attributer.gpu == True  # Default value

def test_version():
    import seam
    assert hasattr(seam, '__version__') 