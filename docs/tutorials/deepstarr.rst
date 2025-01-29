DeepSTARR Tutorial
================

This tutorial demonstrates how to use SEAM with the DeepSTARR model.

Setup
-----

First, load the required packages and data:

.. code-block:: python

    import seam
    import squid
    import numpy as np
    import tensorflow as tf
    
    # Load DeepSTARR model
    model = tf.keras.models.load_model('deepstarr.h5')
    
    # Load sequences
    sequences = np.load('sequences.npy')

Analysis Pipeline
--------------

1. Generate Attribution Maps
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize attributer
    attributer = seam.Attributer(model, method='saliency')
    
    # Generate maps
    maps = attributer.generate(sequences)

2. Cluster Mechanisms
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize clusterer
    clusterer = seam.Clusterer(maps)
    
    # Generate embedding
    embedding = clusterer.embed(method='umap')
    
    # Perform clustering
    labels = clusterer.cluster(method='kmeans', n_clusters=10)

3. Analyze Results
~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize meta-explainer
    meta = seam.MetaExplainer(maps)
    
    # Generate MSM
    msm = meta.generate_msm()
    
    # Plot results
    meta.plot_msm()
    meta.generate_logos() 