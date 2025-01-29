Utilities
=========

.. automodule:: seam.utils
   :members:
   :undoc-members:
   :show-inheritance:

The utils module provides utility functions for SEAM.

Example Usage
------------

.. code-block:: python

    from seam import utils
    
    # Suppress warnings
    utils.suppress_warnings()
    
    # Get compute device
    device = utils.get_device(gpu=True)
    
    # Convert array to pandas DataFrame
    df = utils.arr2pd(array, alphabet=['A', 'C', 'G', 'T']) 