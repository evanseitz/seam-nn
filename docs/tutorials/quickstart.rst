Quickstart Guide
==============

This tutorial will help you get started with SEAM quickly.

Basic Setup
----------

.. code-block:: python

    import seam
    import numpy as np
    import tensorflow as tf
    
    # Suppress warnings (optional)
    seam.suppress_warnings()
    
    # Check for GPU
    device = seam.get_device(gpu=True)

Working with Sequences
-------------------

.. code-block:: python

    # Example sequence data
    sequences = np.random.random((100, 200, 4))  # 100 sequences, length 200, one-hot encoded
    predictions = model.predict(sequences)        # Get model predictions
    
    # Compile data
    compiler = seam.Compiler(x=sequences, y=predictions)
    mave = compiler.compile()

Generating Attributions
--------------------

.. code-block:: python

    # Initialize attributer
    attributer = seam.Attributer(model, method='saliency')
    
    # Generate attribution maps
    maps = attributer.generate(sequences)

Analyzing Mechanisms
-----------------

.. code-block:: python

    # Initialize meta-explainer
    meta = seam.MetaExplainer(maps, alphabet=['A', 'C', 'G', 'T'])
    
    # Generate and plot MSM
    msm = meta.generate_msm()
    meta.plot_msm(column='Entropy')
    
    # Generate sequence logos
    meta.generate_logos() 