Compiler
========

.. automodule:: seam.compiler
   :members:
   :undoc-members:
   :show-inheritance:

The Compiler class handles sequence data compilation and preprocessing.

Example Usage
------------

.. code-block:: python

    import seam
    
    # Initialize compiler with sequence data and predictions
    compiler = seam.Compiler(x=sequences, y=predictions)
    
    # Compile data into MAVE format
    mave = compiler.compile()
    
    # Calculate sequence distances
    distances = compiler.calculate_distances() 