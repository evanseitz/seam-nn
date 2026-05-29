Installation
============

SEAM requires Python 3.7.2 or later. We recommend using Anaconda to manage dependencies.

Standard Install (CPU)
----------------------

With `Anaconda <https://docs.anaconda.com/free/anaconda/install/index.html>`_ sourced, create a new environment with **Python 3.8 or later**:

.. code-block:: bash

   conda create --name seam python=3.9
   conda activate seam
   pip install seam-nn

When you are done using the environment, always exit via ``conda deactivate``.

.. note::

   Some specialized workflows (e.g., the SEAM GUI, certain example scripts, or model-specific pipelines) may require older Python or package versions. Review the installation notes at the top of any relevant script before creating your environment.

.. note::

   GPU access is **not required** to use SEAM. ``pip install seam-nn`` installs TensorFlow as a dependency, but GPU acceleration is only used by optional code paths—primarily **Attributer** (attribution maps) and **Clusterer** (hierarchical clustering distance matrices). **Compiler**, **MetaExplainer**, and **Identifier** do not require a GPU. All GPU-enabled paths fall back to CPU when no GPU is detected; attribution and hierarchical clustering on large sequence libraries are the steps most noticeably slower without one.

GPU Support (Optional)
----------------------

SEAM uses TensorFlow for GPU acceleration in **Attributer** and in **Clusterer** hierarchical clustering (distance-matrix computation). To utilize GPU acceleration, your environment must have a strictly matched combination of Python, TensorFlow, CUDA, and cuDNN.

Installing ``seam-nn`` via pip pulls in TensorFlow, but does not guarantee the correct CUDA/cuDNN runtime libraries are installed. If you see the warning ``Could not find cuda drivers on your machine, GPU will not be used``, your GPU is present but the CUDA runtime is missing or mismatched.

Recommended: Python 3.9+ with TensorFlow 2.16+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TensorFlow 2.16+ bundles the required NVIDIA libraries via pip. You do not need to install CUDA via Conda. **Use Python 3.9 or later** and install GPU-enabled TensorFlow before SEAM:

.. code-block:: bash

   conda create --name seam-gpu python=3.9
   conda activate seam-gpu
   pip install "tensorflow[and-cuda]"
   pip install seam-nn

On Python 3.9+, ``pip install seam-nn`` alone may also enable GPU support (as it pulls a recent TensorFlow), but installing ``tensorflow[and-cuda]`` first is the most reliable approach.

Legacy GPU Environments (Python 3.8 or TensorFlow < 2.16)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you use **Python 3.8**, pip installs TensorFlow 2.13, which does **not** support ``tensorflow[and-cuda]``. You must manually install the exact CUDA Toolkit and cuDNN versions that match your TensorFlow version. For example, TensorFlow 2.12–2.14 requires CUDA 11.8 and cuDNN 8.6:

.. code-block:: bash

   conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6
   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
   pip install seam-nn

If ``cudnn=8.6`` is unavailable on your platform, try ``conda install -c conda-forge cudnn`` without a version pin. You may need to add the ``LD_LIBRARY_PATH`` export to your shell profile or job script so it persists across sessions.

Verify GPU detection with:

.. code-block:: python

   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))

.. warning::

   If you are managing dependencies manually, consult the official `TensorFlow Tested Build Configurations Table <https://www.tensorflow.org/install/source#gpu>`_ to find the exact CUDA and cuDNN versions required for your specific version of TensorFlow and Python.

Getting Help
------------

If you have any issues installing SEAM, please see:

* https://seam-nn.readthedocs.io/en/latest/installation.html
* https://github.com/evanseitz/seam-nn/issues

For issues installing SQUID, the package used for sequence generation and inference, please see:

* https://squid-nn.readthedocs.io/en/latest/installation.html
* https://github.com/evanseitz/squid-nn/issues

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

* numpy
* matplotlib >= 3.6.0
* pandas
* tqdm
* psutil
* biopython
* tensorflow >= 2.0.0
* scipy >= 1.7.0
* squid-nn
* seaborn

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* umap-learn (for UMAP)
* phate (for PHATE)
* openTSNE (for t-SNE)
* scikit-learn (for PCA, K-means, DBSCAN)
* cuml (for GPU-accelerated UMAP, t-SNE, and PCA)
* kmeanstf (for GPU-accelerated K-means)
* shap (for DeepSHAP)

Development
-----------

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
* DNNs trained with TF1.x may require separate environments for TF1.x and TF2.x
