SEAM: systematic explanation of attribution-based mechanisms for regulatory genomics
========================================================================
[![PyPI version](https://badge.fury.io/py/seam-nn.svg)](https://badge.fury.io/py/seam-nn)
[![Downloads](https://static.pepy.tech/badge/seam-nn)](https://pepy.tech/project/seam-nn) 
[![Documentation Status](https://readthedocs.org/projects/seam-nn/badge/?version=latest)](https://seam-nn.readthedocs.io/en/latest/?badge=latest)
<!-- [![DOI](https://zenodo.org/badge/711703377.svg)](https://zenodo.org/doi/10.5281/zenodo.11060671) -->

<p align="center">
	<img src="https://raw.githubusercontent.com/evanseitz/seam-nn/main/docs/_static/seam_logo_light.png#gh-light-mode-only" width="250" height="250">
</p>
<p align="center">
	<img src="https://raw.githubusercontent.com/evanseitz/seam-nn/main/docs/_static/seam_logo_dark.png#gh-dark-mode-only" width="250" height="250">
</p>

**SEAM** (**S**ystematic **E**xplanation of **A**ttribution-based for **M**echanisms) is a Python suite to use meta-explanations to interpret sequence-based deep learning models for regulatory genomics data. For installation instructions, tutorials, and documentation, please refer to the SEAM website, https://seam-nn.readthedocs.io/. For an extended discussion of this approach and its applications, please refer to our paper:

* Seitz, E.E., McCandlish, D.M., Kinney, J.B., and Koo P.K. Deciphering the determinants of mechanistic variation in regulatory sequences. <em>bioRxiv</em> (2025). (unpublished)
---

## Installation:

With [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) sourced, create a new environment via the command line:

```bash
conda create --name seam
```

Next, activate this environment via `conda activate seam`, and install the following packages:

```bash
pip install seam-nn
```

Finally, when you are done using the environment, always exit via `conda deactivate`.


### Notes

SEAM has been tested on Mac and Linux operating systems. Typical installation time on a normal computer is less than 1 minute.

If you have any issues installing SEAM, please see:
- https://seam-nn.readthedocs.io/en/latest/installation.html
- https://github.com/evanseitz/seam-nn/issues

For issues installing SQUID, the package used for sequence generation and inference, please see:
- https://squid-nn.readthedocs.io/en/latest/installation.html
- https://github.com/evanseitz/squid-nn/issues

Older DNNs that require inference via Tensorflow 1.x or related packages may be in conflict with SEAM defaults. Users will need to run SEAM piecewise within separate environments:
1. Tensorflow 1.x environment for generating *in silico* sequence-function-mechanism dataset
2. Tensorflow 2.x environment for applying SEAM to explain *in silico* sequence-function-mechanism dataset

## Usage:
SEAM provides a simple interface that takes as input a sequence-based oracle (e.g., a genomic DNN), which is used to generate an *in silico* sequence-function-mechanism dataset representing a localized region of sequence space. SEAM uses a meta-explanation framework to interpret the *in silico* sequence-function-mechanism dataset, deciphering the determinants of mechanistic variation in regulatory sequences.

<img src="https://raw.githubusercontent.com/evanseitz/seam-nn/main/docs/_static/framework.png" alt="fig" width="800"/>

### Examples

**Google Colab examples** for applying SEAM on previously-published deep learning models are available at the following links:

- [Local library to annotate all TFBSs and biophysical states](https://colab.research.google.com/drive/1HOM_ysa4HIh_ZoYzLwa4jZu4evyRntF7?usp=sharing)
	- DeepSTARR: Enhancer 20647 (Fig.2a)
	- Local library with 30k sequences and 10% mutation rate | Integrated gradients; hierarchical clustering
    - Expected run time: **~3.2 minutes** on Colab A100 GPU
- [Local library augmented with SNV and DNV libraries to map evolvability of mechanisms](TODO)
	- DeepSTARR: Enhancer 22612 (Fig.2b)
	- Local library with 30,000 sequences and 10% mutation rate
	- SNV and DNV library with 77,029 sequences
	- Integrated gradients; PCA with K-Means clustering
	- Expected run time: **~TODO** minutes
- [Local library to reveal low-affinity motifs using background separation](https://colab.research.google.com/drive/1lkcLYMyVMYPh3ARzYjI-gJjh69PK9COt?usp=sharing)
	- DeepSTARR: Enhancer 4071 (Fig.TBD)
	- Local library with 60k sequences and 10% mutation rate | Integrated gradients; hierarchical clustering
    - Expected run time: **~9.3 minutes** on Colab A100 GPU
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


**Python script examples** are provided in the `examples/` folder for locally running SEAM and exporting outputs to file. Additional dependencies for these examples may be required and outlined at the top of each script.

## Citation:
If this code is useful in your work, please cite our paper.

bibtex TODO

## License:
Copyright (C) 2023–2025 Evan Seitz, David McCandlish, Justin Kinney, Peter Koo

The software, code sample and their documentation made available on this website could include technical or other mistakes, inaccuracies or typographical errors. We may make changes to the software or documentation made available on its web site at any time without prior notice. We assume no responsibility for errors or omissions in the software or documentation available from its web site. For further details, please see the LICENSE file.
