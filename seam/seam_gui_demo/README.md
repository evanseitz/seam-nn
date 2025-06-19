# SEAM GUI Demo

This demo requires four files derived using the SQUID/SEAM API based on the DeepSTARR DNN:

## Required Files

- **MAVE dataset** - [Download MAVE dataset](https://drive.google.com/file/d/1YcItpu1zSkO2m7LVwkuZdcrmlmJLR3gJ/view?usp=sharing)
- **Attribution maps** - [Download attribution maps](https://drive.google.com/file/d/1jWShhFzBhxJ22DUNxjDIVFOrzuf-7i0P/view?usp=sharing)
- **Embedding** - [Download UMAP embedding](https://drive.google.com/file/d/1pk1UQ5-HE2thrYTqGyJ2nFGVpIC-lBHU/view?usp=sharing)
- **Linkage** - [Download linkage data](https://drive.google.com/file/d/1Qqc6FOBq4C31TiMekTXxvati3XNSWTaN/view?usp=sharing)

Download these files using the links above and place them in the current directory before running the GUI demo.

## SEAM Hyperparameters

- **MAVE dataset:** Local library with N=100,000 sequences (L=250 nt) generated with 10% mutation rate applied to a DNA (A=4) reference sequence in the DeepSTARR test set (an enhancer at index 22612)

- **Attribution maps:** DeepSHAP attribution maps of shape=(N,L,A), generated with respect to the Developmental head (task 0)

- **Clusters:** Clusters can be produced from either a UMAP 2D embedding or hierarchical clustering with Ward's linkage