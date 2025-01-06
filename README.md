SEAM: systematic explanation of attribution-based mechanisms for regulatory genomics
========================================================================
[![PyPI version](https://badge.fury.io/py/seam-nn.svg)](https://badge.fury.io/py/seam-nn)
<!-- [![Downloads](https://static.pepy.tech/badge/seam-nn)](https://pepy.tech/project/seam-nn) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/squid-nn/badge/?version=latest)](https://squid-nn.readthedocs.io/en/latest/?badge=latest) -->
<!-- [![DOI](https://zenodo.org/badge/711703377.svg)](https://zenodo.org/doi/10.5281/zenodo.11060671) -->

<p align="center">
	<img src="./docs/_static/seam_logo_light.png#gh-light-mode-only" width="250" height="250">
</p>
<p align="center">
	<img src="./docs/_static/seam_logo_dark.png#gh-dark-mode-only" width="250" height="250">
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
pip install squid-nn
```

Finally, when you are done using the environment, always exit via `conda deactivate`.


### Notes

SEAM has been tested on Mac and Linux operating systems. Typical installation time on a normal computer is less than 2 minutes.

If you have any issues installing SEAM, please see:
- https://seam-nn.readthedocs.io/en/latest/installation.html
- https://github.com/evanseitz/seam-nn/issues

For issues installing SQUID, the package used for sequence generation and inference, please see:
- https://squid-nn.readthedocs.io/en/latest/installation.html
- https://github.com/evanseitz/squid-nn/issues

Older DNNs may require inference via Tensorflow 1.x or related packages in conflict with SEAM defaults. Users will need to run SEAM piecewise within separate environments:
1. Tensorflow 1.x environment for generating *in silico* sequence-function-mechanism dataset
2. Tensorflow 2.x environment for applying SEAM to explain *in silico* sequence-function-mechanism dataset

## Usage:
SEAM provides a simple interface that takes as input a sequence-based deep-learning model (e.g., a DNN), which is used as an oracle to generate an *in silico* sequence-function-mechanism dataset representing a localized region of sequence space. SEAM uses a meta-explanation framework to interpret the *in silico* sequence-function-mechanism dataset, deciphering the determinants of mechanistic variation in regulatory sequences.

<!-- <img src="./docs/_static/framework.png" alt="fig" width="1000"/> -->

### Examples

**Google Colab examples** for applying SEAM on previously-published deep learning models are available at the following links:

- To be done.

**Python script examples** are provided in the `examples/` folder for locally running SEAM and exporting outputs to file. Additional dependencies for these examples may be required and outlined at the top of each script. Examples include:

- To be done.

<!-- As well, the [seam-manuscript](https://github.com/evanseitz/seam-manuscript) repository contains examples to reproduce results in the manuscript, including the application of SQUID on other DNNs such as ChromBPNet and directly to experimental datasets -->

<!-- Expected run time for the "Variant effect (local) prediction with DeepSTARR–Kipoi" demo (above) is 4 minutes using Google Colab V100 GPU. -->

## Citation:
If this code is useful in your work, please cite our paper.

bibtex TBD

## License:
Copyright (C) 2023–2025 Evan Seitz, David McCandlish, Justin Kinney, Peter Koo

The software, code sample and their documentation made available on this website could include technical or other mistakes, inaccuracies or typographical errors. We may make changes to the software or documentation made available on its web site at any time without prior notice. We assume no responsibility for errors or omissions in the software or documentation available from its web site. For further details, please see the LICENSE file.
