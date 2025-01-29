from pathlib import Path
import sys

# Add package root to path
package_root = str(Path(__file__).parent)
if package_root not in sys.path:
    sys.path.append(package_root)

# Import key classes and functions
from .meta_explainer import MetaExplainer
from .compiler import Compiler
from .attributer import Attributer
from .clusterer import Clusterer
from .utils import suppress_warnings, get_device

# Version info
__version__ = "0.1.4"

# Expose key classes
__all__ = [
    'MetaExplainer',
    'Compiler',
    'Attributer', 
    'Clusterer',
    'suppress_warnings',
    'get_device'
]
