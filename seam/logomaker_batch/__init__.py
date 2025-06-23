from .batch_logo import BatchLogo

def __init__(self, values, alphabet=None, figsize=[10,2.5], batch_size=10, gpu=False, **kwargs):
    # Initialize class-level caches if they don't exist
    if not hasattr(BatchLogo, '_path_cache'):
        BatchLogo._path_cache = {}
    if not hasattr(BatchLogo, '_m_path_cache'):
        BatchLogo._m_path_cache = {}
    if not hasattr(BatchLogo, '_transform_cache'):
        BatchLogo._transform_cache = {}