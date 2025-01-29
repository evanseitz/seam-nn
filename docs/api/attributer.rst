Attributer
==========

.. automodule:: seam.attributer
   :members:
   :undoc-members:
   :show-inheritance:

The Attributer class generates attribution maps from deep learning models.

Example Usage
------------

.. code-block:: python

    import seam
    
    # Initialize attributer with model
    attributer = seam.Attributer(model, method='saliency')
    
    # Generate attribution maps
    maps = attributer.generate(sequences)
    
    # Optional: use other attribution methods
    maps_ism = attributer.generate(sequences, method='ism')
    maps_shap = attributer.generate(sequences, method='shap') 