from .Logo import Logo
from .batch_logo import BatchLogo
from .gpu_utils import GPUTransformer
from .colors import get_rgb, get_color_dict
from .Glyph import Glyph
from .error_handling import check, handle_errors
from .matrix import ALPHABET_DICT
from .validate import validate_matrix

def __init__(self, values, alphabet=None, figsize=[10,2.5], batch_size=10, gpu=False, **kwargs):
    # Initialize class-level caches if they don't exist
    if not hasattr(BatchLogo, '_path_cache'):
        BatchLogo._path_cache = {}
    if not hasattr(BatchLogo, '_m_path_cache'):
        BatchLogo._m_path_cache = {}
    if not hasattr(BatchLogo, '_transform_cache'):
        BatchLogo._transform_cache = {}