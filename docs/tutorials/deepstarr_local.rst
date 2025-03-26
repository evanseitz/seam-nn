DeepSTARR Local Analysis Tutorial
===============================

This tutorial demonstrates how to use SEAM to analyze local regulatory mechanisms in the DeepSTARR model, reproducing Figure 2 from our paper.

.. note::
   Expected runtime: ~3.2 minutes using Google Colab A100 GPU

Setup
-----

First, let's import the required packages:

.. code-block:: python

    import time
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import squid
    import seam

    from seam import MetaExplainer, Compiler, Attributer, Clusterer
    from seam import suppress_warnings, get_device

    # Optional: suppress warnings
    suppress_warnings()

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

Loading Data and Model
--------------------

We'll use the DeepSTARR model and a local sequence library. The model predicts developmental and housekeeping enhancer activity in Drosophila S2 cells.

.. code-block:: python

    # Load model
    model = tf.keras.models.load_model('deepstarr.h5')

    # Load sequence data
    sequences = np.load('sequences.npy')
    predictions = model.predict(sequences)

    print(f"Sequences shape: {sequences.shape}")
    print(f"Predictions shape: {predictions.shape}")

Data Preprocessing
---------------

We'll use the Compiler class to organize our data:

.. code-block:: python

    # Initialize compiler
    compiler = Compiler(x=sequences, y=predictions)
    
    # Compile data into MAVE format
    mave = compiler.compile()
    
    print("MAVE dataframe head:")
    print(mave.head())

Attribution Map Generation
-----------------------

Next, we'll generate attribution maps using the saliency method:

.. code-block:: python

    # Initialize attributer
    attributer = Attributer(
        model, 
        method='saliency',
        gpu=True,  # Use GPU if available
        batch_size=32
    )
    
    # Generate maps
    t1 = time.time()
    maps = attributer.generate(sequences)
    print(f"Attribution time: {time.time() - t1:.2f} seconds")
    print(f"Maps shape: {maps.shape}")

Clustering Analysis
----------------

We'll use hierarchical clustering to group similar regulatory mechanisms:

.. code-block:: python

    # Initialize clusterer
    clusterer = Clusterer(maps)
    
    # Generate UMAP embedding
    embedding = clusterer.embed(
        method='umap',
        n_components=2,
        n_neighbors=15,
        min_dist=0.1
    )
    
    # Perform hierarchical clustering
    labels = clusterer.cluster(
        method='hierarchical',
        n_clusters=10,
        metric='euclidean',
        linkage='ward'
    )
    
    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot dendrogram
    clusterer.plot_dendrogram(ax=ax[0])
    
    # Plot embedding
    clusterer.plot_embedding(
        embedding=embedding,
        labels=labels,
        ax=ax[1]
    )
    
    plt.tight_layout()
    plt.show()

Mechanism Analysis
---------------

Now we'll use MetaExplainer to analyze the identified mechanisms:

.. code-block:: python

    # Initialize meta-explainer
    meta = MetaExplainer(
        maps,
        alphabet=['A', 'C', 'G', 'T'],
        window_size=20
    )
    
    # Generate Mechanism Summary Matrix (MSM)
    msm = meta.generate_msm(
        gpu=True  # Use GPU if available
    )
    
    # Plot MSM
    meta.plot_msm(
        column='Entropy',
        square_cells=True,
        view_window=[50,170],
        cmap='rocket_r'
    )
    
    # Generate sequence logos for each cluster
    logos = meta.generate_logos(
        center_values=True,
        figsize=(20, 2.5)
    )

Interpreting Results
-----------------

The results show:

1. **Clustering**: The dendrogram reveals distinct groups of regulatory mechanisms
2. **UMAP**: The embedding shows how mechanisms are related in 2D space
3. **MSM**: The entropy matrix highlights regions of mechanistic importance
4. **Logos**: Sequence logos reveal the specific patterns in each cluster

Advanced Visualization
-------------------

For more detailed analysis, we can customize the visualizations:

.. code-block:: python

    # Plot MSM with different options
    meta.plot_msm(
        column='Frequency',  # Use frequency instead of entropy
        square_cells=True,
        view_window=[50,170],
        cmap='viridis'
    )
    
    # Generate logos with different settings
    meta.generate_logos(
        indices=[0,1,2],  # Only show first 3 clusters
        center_values=True,
        figsize=(15, 2)
    )

Saving Results
------------

Finally, we can save our results:

.. code-block:: python

    # Save MSM data
    np.save('msm_data.npy', msm)
    
    # Save cluster labels
    np.save('cluster_labels.npy', labels)
    
    # Save embedding
    np.save('umap_embedding.npy', embedding)

.. note::
   For more examples and advanced usage, please refer to our GitHub repository and the API documentation. 