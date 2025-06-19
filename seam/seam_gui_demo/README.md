# SEAM GUI Demo

This demo requires four files derived using the SQUID/SEAM API based on the DeepSTARR DNN:

## Required Files

- **MAVE dataset** - [Download MAVE dataset](https://drive.google.com/file/d/1YcItpu1zSkO2m7LVwkuZdcrmlmJLR3gJ/view?usp=sharing)
    - DataFrame (.csv) containing N=100,000 sequence-function relationships in the 'Sequence' and 'DNN' columns. The 'Sequence' column contains a local library with N=100,000 sequences (L=250 nt) generated with 10% mutation rate applied to reference enhancer (DNA, A=4) in the DeepSTARR test set (index 22612). The 'DNN' column contains DNN predictions for these sequences with respect to the Development head (task 0)
- **Attribution maps** - [Download attribution maps](https://drive.google.com/file/d/1jWShhFzBhxJ22DUNxjDIVFOrzuf-7i0P/view?usp=sharing)
    - Array (.npy) of shape=(N, L, A) containing DeepSHAP attribution maps generated with respect to task 0
- **Embedding** - [Download UMAP embedding](https://drive.google.com/file/d/1pk1UQ5-HE2thrYTqGyJ2nFGVpIC-lBHU/view?usp=sharing)
    - Array (.npy) of shape=(N, 2) containing UMAP 2D embedding based on attribution maps
- **Linkage** - [Download linkage data](https://drive.google.com/file/d/1Qqc6FOBq4C31TiMekTXxvati3XNSWTaN/view?usp=sharing)
    - Array (.npy) of shape=(N-1, 4) containing hierarchical clustering linkage matrix with Ward's linkage

Download these files using the links above and place them in the current directory before running the GUI demo.