Installation
============

SEAM requires Python 3.7.2 or later. We recommend using Anaconda to manage dependencies.

Quick Install
------------

.. code-block:: bash

   conda create --name seam
   conda activate seam
   pip install seam-nn

Dependencies
-----------

Core Dependencies:
~~~~~~~~~~~~~~~~

* numpy
* matplotlib >= 3.2.0
* pandas
* tqdm
* psutil
* biopython
* tensorflow >= 2.0.0
* scipy >= 1.7.0
* squid-nn
* seaborn

Optional Dependencies:
~~~~~~~~~~~~~~~~~~~~

For embedding methods:

* umap-learn (for UMAP)
* phate (for PHATE)
* openTSNE (for t-SNE)
* scikit-learn (for PCA, K-means, DBSCAN)

For attribution methods:

* shap (for DeepSHAP)

Development
----------

For development installation:

.. code-block:: bash

   git clone https://github.com/evanseitz/seam-nn.git
   cd seam-nn
   pip install -e .[dev]

Notes
-----

* SEAM has been tested on Mac and Linux operating systems
* Installation typically takes less than 1 minute
* For issues with SQUID installation, see: https://squid-nn.readthedocs.io/
* Older DNNs may require separate environments for TF1.x and TF2.x 