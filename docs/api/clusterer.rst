Clusterer
=========

.. automodule:: seam.clusterer
   :members:
   :undoc-members:
   :show-inheritance:

The Clusterer class performs embedding and clustering of attribution maps.

Example Usage
------------

.. code-block:: python

    import seam
    
    # Initialize clusterer with attribution maps
    clusterer = seam.Clusterer(maps)
    
    # Generate embedding
    embedding = clusterer.embed(method='umap')
    
    # Perform clustering
    labels = clusterer.cluster(method='kmeans', n_clusters=10)
    
    # Visualize results
    clusterer.plot_embedding() 