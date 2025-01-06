SEAM: systematic explanation of attribution-based mechanisms for regulatory genomics
========================================================================
[![PyPI version](https://badge.fury.io/py/seam-nn.svg)](https://badge.fury.io/py/seam-nn)
<!-- [![Downloads](https://static.pepy.tech/badge/seam-nn)](https://pepy.tech/project/seam-nn) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/squid-nn/badge/?version=latest)](https://squid-nn.readthedocs.io/en/latest/?badge=latest) -->
<!-- [![DOI](https://zenodo.org/badge/711703377.svg)](https://zenodo.org/doi/10.5281/zenodo.11060671) -->

<br/>
<p align="center">
	<img src="./docs/_static/seam_logo_light.png#gh-light-mode-only" width="200" height="200">
</p>
<p align="center">
	<img src="./docs/_static/seam_logo_dark.png#gh-dark-mode-only" width="200" height="200">
</p>
<br/>

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

SEAM has been tested on Mac and Linux operating systems. Typical installation time on a normal computer is less than 2 minutes.

If you have any issues installing SEAM, please see:
- https://seam-nn.readthedocs.io/en/latest/installation.html
- https://github.com/evanseitz/seam-nn/issues

For issues installing SQUID, the package used for sequence generation and inference, please see:
- https://squid-nn.readthedocs.io/en/latest/installation.html
- https://github.com/evanseitz/squid-nn/issues