MetaExplainer
============

.. automodule:: seam.meta_explainer
   :members:
   :undoc-members:
   :show-inheritance:

The MetaExplainer class is the main interface for generating and analyzing meta-explanations.

Example Usage
------------

.. code-block:: python

    import seam
    
    # Initialize MetaExplainer with attribution maps
    meta = seam.MetaExplainer(maps, alphabet=['A', 'C', 'G', 'T'])
    
    # Generate and plot MSM
    msm = meta.generate_msm()
    meta.plot_msm(column='Entropy')
    
    # Generate sequence logos
    meta.generate_logos() 