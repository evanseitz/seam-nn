API Documentation
===============

.. toctree::
   :maxdepth: 2

   meta_explainer
   compiler
   attributer
   clusterer
   utils

The SEAM package provides several key classes for analyzing and interpreting sequence-based deep learning models:

* :class:`MetaExplainer`: Main interface for meta-explanations
* :class:`Compiler`: Handles sequence data compilation and preprocessing
* :class:`Attributer`: Generates attribution maps from models
* :class:`Clusterer`: Performs embedding and clustering of attribution maps

Below is a flowchart showing the relationships between these components:

.. image:: /_static/api_flowchart.png 