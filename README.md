SEAM: systematic explanation of attribution-based mechanisms for regulatory genomics
========================================================================
[![PyPI version](https://badge.fury.io/py/seam-nn.svg)](https://badge.fury.io/py/seam-nn)
[![Downloads](https://static.pepy.tech/badge/seam-nn)](https://pepy.tech/project/seam-nn) 
[![Documentation Status](https://readthedocs.org/projects/seam-nn/badge/?version=latest)](https://seam-nn.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17314556.svg)](https://doi.org/10.5281/zenodo.17314556)

<p align="center">
	<img src="https://raw.githubusercontent.com/evanseitz/seam-nn/main/docs/_static/seam_logo_light.png#gh-light-mode-only" width="250" height="250">
</p>
<p align="center">
	<img src="https://raw.githubusercontent.com/evanseitz/seam-nn/main/docs/_static/seam_logo_dark.png#gh-dark-mode-only" width="250" height="250">
</p>

This repository contains the Python implementation of **SEAM** (**S**ystematic **E**xplanation of **A**ttribution-based **M**echanisms), an AI interpretation framework that systematically investigates how mutations reshape regulatory mechanisms. For an extended discussion of this approach and its applications, please refer to our manuscript, which is currently in review:

* Seitz, E.E., McCandlish, D.M., Kinney, J.B., and Koo P.K. Uncovering the Mechanistic Landscape of Regulatory DNA with Deep Learning. *bioRxiv*, October 8, 2025. https://www.biorxiv.org/content/10.1101/2025.10.07.681052v1

---

## Installation:

### Standard Install (CPU)

With [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) sourced, create a new environment with **Python 3.8 or later**:

```bash
conda create --name seam python=3.9
```

Next, activate this environment and install SEAM:

```bash
conda activate seam
pip install seam-nn
```

Finally, when you are done using the environment, always exit via `conda deactivate`.

> **Note:** Some specialized workflows (e.g., the SEAM GUI, certain example scripts, or model-specific pipelines) may require older Python or package versions. Review the installation notes at the top of any relevant script before creating your environment.

> **Note:** GPU access is **not required** to use SEAM. `pip install seam-nn` installs TensorFlow as a dependency, but GPU acceleration is only used by optional code paths—primarily **`Attributer`** (attribution maps) and **`Clusterer`** (hierarchical clustering distance matrices). **`Compiler`**, **`MetaExplainer`**, and **`Identifier`** do not require a GPU. All GPU-enabled paths fall back to CPU when no GPU is detected; attribution and hierarchical clustering on large sequence libraries are the steps most noticeably slower without one.

### GPU Support (Optional)

SEAM uses TensorFlow for GPU acceleration in **`Attributer`** and in **`Clusterer`** hierarchical clustering (distance-matrix computation). To utilize GPU acceleration, your Python, TensorFlow, CUDA, cuDNN, and NVIDIA driver versions must be compatible with one another.

Installing `seam-nn` via pip pulls in TensorFlow, but does not guarantee that the CUDA, cuDNN, and NVIDIA driver stack required for GPU acceleration is correctly configured. If you see the warning `Could not find cuda drivers on your machine, GPU will not be used`, TensorFlow was unable to access a compatible CUDA installation and will fall back to CPU execution.

#### Recommended: Python 3.9+ with TensorFlow 2.16+

TensorFlow 2.16+ supports GPU installation through tensorflow[and-cuda], which installs the required NVIDIA CUDA/cuDNN runtime packages via pip. In most cases, you do not need to install CUDA or cuDNN separately through Conda. **Use Python 3.9 or later** and install GPU-enabled TensorFlow before SEAM:

```bash
conda create --name seam-gpu python=3.9
conda activate seam-gpu
pip install "tensorflow[and-cuda]"
pip install seam-nn
```

On Python 3.9+, `pip install seam-nn` may be sufficient for GPU support on some systems because it installs a recent TensorFlow release. However, explicitly installing `tensorflow[and-cuda]` first is the recommended and more reproducible approach.

#### Legacy GPU Environments (Python 3.8 or TensorFlow < 2.16)

If you use Python 3.8, you will generally receive an older TensorFlow release (currently 2.13.x) that does not support `tensorflow[and-cuda]`. You must manually install the exact CUDA Toolkit and cuDNN versions that match your TensorFlow version. For example, TensorFlow 2.12–2.14 requires CUDA 11.8 and cuDNN 8.6:

```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
pip install seam-nn
```

If `cudnn=8.6` is unavailable on your platform, try `conda install -c conda-forge cudnn` without a version pin. You may need to add the `LD_LIBRARY_PATH` export to your shell profile or job script so it persists across sessions.

> **⚠️ Troubleshooting:** If you are managing dependencies manually, you must consult the official [TensorFlow Tested Build Configurations Table](https://www.tensorflow.org/install/source#gpu) to find the exact CUDA and cuDNN versions required for your specific version of TensorFlow and Python.

You can verify GPU detection with:

```bash
nvidia-smi
```

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

If no GPU is detected, first verify that the NVIDIA driver is visible to the operating system using `nvidia-smi`.

> If you have any issues installing SEAM, please see:
> - https://seam-nn.readthedocs.io/en/latest/installation.html
> - https://github.com/evanseitz/seam-nn/issues
>
> For issues installing SQUID, the package used for sequence generation and inference, please see:
> - https://squid-nn.readthedocs.io/en/latest/installation.html
> - https://github.com/evanseitz/squid-nn/issues

## Usage and Requirements:
SEAM provides a unified interface for mechanistic interpretation of sequence-based deep learning models. 

<img src="https://raw.githubusercontent.com/evanseitz/seam-nn/main/docs/_static/framework.png" alt="fig" width="800"/>


The framework takes as input a sequence-based oracle (e.g., a genomic DNN) and requires four key components to perform analysis:

1. **Sequence Library** (`numpy.ndarray`): One-hot encoded sequences of shape (N, L, A), where:
   - N: Number of sequences
   - L: Sequence length
   - A: Number of features (e.g., 4 for DNA nucleotides)

2. **Predictions/Measurements** (`numpy.ndarray`): Experimental or model-derived values of shape (N,1), corresponding to each sequence's functional output.

3. **Attribution Maps** (`numpy.ndarray`): Mechanistic importance scores of shape (N, L, A), quantifying the contribution of each position-feature pair to the sequence's function. These can be generated using various attribution methods:

4. **Clustering/Embedding** (either):
   - Hierarchical clustering linkage matrix (e.g., from `scipy.cluster.hierarchy.linkage`)
   - Dimensionality reduction embedding of shape (N,Z), where Z is the number of dimensions in the embedded space

These required files can be generated either externally or using SEAM's specialized modules (described below). Once provided, SEAM applies a meta-explanation approach to interpret the sequence-function-mechanism dataset, deciphering the determinants of mechanistic variation in regulatory sequences.

For detailed examples of how to generate these requirements using SEAM's modules and apply the analysis pipeline to reproduce key findings from our main manuscript, see the Examples section at the end of this document.

## SEAM Modules:
SEAM’s analysis pipeline is organized into modular components, with outputs from each stage feeding into the next. The `Mutagenizer`, `Compiler`, `Attributer`, and `Clusterer` modules generate core data products, which are integrated by the `MetaExplainer` to characterize each SEAM-derived mechanism. The `Identifier` module then builds on these outputs to annotate regulatory elements and quantify their combinatorial relationships.

- **Mutagenizer** (from [SQUID](https://github.com/evanseitz/squid-nn)): Generates *in silico* sequence libraries through various mutagenesis strategies, including local, global, optimized, and complete libraries (supporting all combinatorial mutations up to a specified order). Features GPU-acceleration and batch processing for efficient sequence generation.

- **Compiler**: Standardizes sequence analysis by converting one-hot encoded sequences to string format and computing associated metrics. Compiles sequences and functional properties into a DataFrame, with support for metrics such as Hamming distances and global importance analysis scores. Implements GPU-accelerated sequence conversion and vectorized operations.

- **Attributer**: Computes attribution maps that quantify the base-wise contribution to regulatory activity. SEAM provides TensorFlow 2 GPU-accelerated implementations of Saliency Maps, IntGrad, SmoothGrad, and ISM. (For TF2, DeepSHAP is not yet optimized for GPU accelerated batch processing across the sequence library and requires external libraries `pip install kundajelab-shap==1` and `pip install deeplift`).

- **Clusterer**: Computes mechanistic clusters and embeddings from attribution maps to identify distinct regulatory mechanisms. Supports hierarchical clustering (GPU-optimized), K-means, and DBSCAN algorithms, with optional dimensionality reduction (UMAP, t-SNE, PCA) for complementary interpretability.

- **MetaExplainer**: The core SEAM module that integrates results to identify and interpret mechanistic patterns. Generates cluster-averaged attribution maps (shape: (L, A) for each cluster) and the Mechanism Summary Matrix (MSM), a DataFrame containing position-wise statistics (entropy, consensus matches, reference mismatches) for each cluster. Also implements background separation and provides visualization tools for sequence logos, attribution logos, and cluster statistics, with support for both PWM-based and enrichment-based analysis.

- **Identifier**: Analyzes cluster-averaged attribution maps in conjunction with the MSM to identify such properties as the precise locations of motifs and their epistatic interactions.

> **Note on terminology**: Our API and examples currently use the legacy terminology "MSM" (Mechanism Summary Matrix), which corresponds to the renamed "CSM" (Cluster Summary Matrix) in our bioRxiv preprint. A future update will address this terminology inconsistency.

## Examples

**Google Colab examples** for applying SEAM on previously-published deep learning models (DeepSTARR) and experimental datasets (PBMs) are available at the links below.

> **Note:** Due to memory requirements for calculating distance matrices, Colab Pro may be required for examples using hierarchical clustering with their current settings.

- [Local library to annotate all TFBSs and binding configurations](https://colab.research.google.com/drive/1HOM_ysa4HIh_ZoYzLwa4jZu4evyRntF7?usp=sharing)
	- DeepSTARR: Enhancer 20647 (Fig.2a)
	- Local library with 30k sequences and 10% mutation rate | Integrated gradients; hierarchical clustering
    - Expected run time: **~3 minutes** on Colab A100 GPU
- [Local library to reveal low-affinity motifs using background separation](https://colab.research.google.com/drive/1lkcLYMyVMYPh3ARzYjI-gJjh69PK9COt?usp=sharing)
	- DeepSTARR: Enhancer 5353 (Fig.TBD)
	- Local library with 60k sequences and 10% mutation rate | Integrated gradients; hierarchical clustering
    - Expected run time: **~8.5 minutes** on Colab A100 GPU
- [Local library to explore mechanism space of an enhancer TFBS](https://colab.research.google.com/drive/1JSDAJNdSNhLOVd2L8hcZXLocWz2iwycq?usp=sharing)
	- DeepSTARR: Enhancer 13748 (SFig.TBD)
	- Local library with 100k sequence and 10% mutation rate | Saliency maps; UMAP with K-Means clustering
	- Expected run time: **~3.9 minutes** on Colab A100 GPU
- [Combinatorial-complete library with empirical mutagenesis maps](https://colab.research.google.com/drive/1IWzjJtKzZCCvN1vndHalThLhPL6AsPEa?usp=sharing)
	- PBM: Zfp187 (Fig.TBD)
	- Combinatorial-complete library with 65,536 sequences | ISM; Hierarchical clustering
	- Expected run time: **~12 minutes** on Colab A100 GPU
- [Combinatorial-complete library with interactive mechanism space viewer](https://colab.research.google.com/drive/1E8_30yW_2i-1y6OFwGOg4jDrhwZLAhMj?usp=sharing)
	- PBM: Hnf4a (Fig.TBD)
	- Combinatorial-complete library with 65,536 sequences | ISM; UMAP with K-Means clustering
	- Expected run time: **~4.9 minutes** on Colab A100 GPU
- [Global library to compare mechanistic heterogeneity of an enhancer TFBS](https://colab.research.google.com/drive/17EvfEa8LGtSjb6JkvVSPs6X0m7Rwb8_l?usp=sharing)
	- DeepSTARR: CREB/ATF (Fig.TBD)
	- Global library with 100k sequences | Saliency maps: UMAP with K-Means clustering
	- Expected run time: **~3.2 minutes** on Colab A100 GPU
- [Global library to compare mechanisms across different developmental programs](https://colab.research.google.com/drive/1uCZ_HpuTiLyL8nmsbZ8lExFrbLWpZ5nG?usp=sharing)
	- DeepSTARR: DRE (Fig.TBD)
	- Global library with 100k sequences | Saliency maps; UMAP with K-Means clustering
	- Expected run time: **~2.7 minutes** on Colab A100 GPU
- [Global library to compare mechanisms associated with genomic and synthetic TFBSs](https://colab.research.google.com/drive/1stdhABAF5Eehg7-n-XfLxoqBYahJ5LX_?usp=sharing)
	- DeepSTARR: AP-1 (Fig.TBD)
	- Global library with 100k sequences | Integrated gradients; UMAP with K-Means clustering
	- Expected run time: **~3.9 minutes** on Colab A100 GPU

**Python script examples** are provided in the `examples` folder for locally running SEAM and exporting outputs to file. These examples additionally include ChromBPNet models and attribution methods that are not compatible with the latest libraries supported by Google Colab:

- [Local library to analyze foreground and background signals at human promotors and enhancers](https://github.com/evanseitz/seam-nn/blob/main/examples/example_chrombpnet_local_ppif.py)
	- ChromBPNet: PPIF promoter/enhancer (Fig.3)
	- Local library with 100k sequences and 10% mutation rate | {Saliency, IntGrad, SmoothGrad, ISM}; Hierarchical clustering

Additional dependencies for these Python examples may be required and outlined at the top of each script.

**Complete pipelines, configurations, and processed data** for reproducing all analyses and figures from the SEAM manuscript, including additional models beyond DeepSTARR, PBMs, and ChromBPNet, are available on [Zenodo](https://zenodo.org/records/17314556).


## SEAM Interactive Interpretability Tool:
A graphic user interface (GUI) is available for dynamically interpretting SEAM results, allowing users to explore and analyze pre-computed inputs from the e. The GUI can be run using the command line interface from the `seam` folder via `python seam_gui.py` with the `seam-gui` environment activated (see below). The SEAM GUI requires pre-computed inputs that can be saved using the example scripts above. Instructions for downloading demo files for running the SEAM GUI are available in the `seam/seam_gui_demo` folder. A full walkthrough of the SEAM GUI using this demo dataset is available on [YouTube](https://youtu.be/0UTnwJj68r0).

<img src="https://raw.githubusercontent.com/evanseitz/seam-nn/main/docs/_static/seam_gui.png" alt="fig" width="800"/>

**SEAM GUI environment** requires alternative imports to the default `seam` environment (above). The `seam-gui` environment can be installed following these steps:

```bash
conda create --name seam-gui python==3.8*
```

Next, activate this environment via `conda activate seam-gui`, and install the following packages:

```bash
	pip install --upgrade pip
	pip install PyQt5
	pip3 install --user psutil
	pip install biopython
	pip install scipy
	pip install seaborn
	pip install -U scikit-learn
	pip install pysam
	pip install seam-nn
	pip install matplotlib==3.6
```
> To avoid conflicts, matplotlib==3.6 must be the last package installed

Finally, when you are done using the environment, always exit via `conda deactivate`.


## Citation:
If this code is useful in your work, please cite our paper:

```bibtex
@article{Seitz2025.10.07.681052,
	author = {Seitz, Evan and McCandlish, David Martin and Kinney, Justin Block and Koo, Peter},
	title = {Uncovering the Mechanistic Landscape of Regulatory DNA with Deep Learning},
	elocation-id = {2025.10.07.681052},
	year = {2025},
	doi = {10.1101/2025.10.07.681052},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/10/08/2025.10.07.681052},
	eprint = {https://www.biorxiv.org/content/early/2025/10/08/2025.10.07.681052.full.pdf},
	journal = {bioRxiv}
}
```

## License:
Copyright (C) 2023–2025 Evan Seitz, David McCandlish, Justin Kinney, Peter Koo

The software, code samples and their documentation made available on this website could include technical or other mistakes, inaccuracies or typographical errors. We may make changes to the software or documentation made available on its web site at any time without prior notice. We assume no responsibility for errors or omissions in the software or documentation available from its web site. For further details, please see the LICENSE file.
