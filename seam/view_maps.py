import os, sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import squid
py_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(py_dir)


if 1:
    dir_name = 'outputs_local_chrombpnet_PPIF_promoter/mut_pt10/seed2_N100k_allFolds'
    attr_map = 'deepshap'
    alphabet = ['A','C','G','T']


inputs_dir = os.path.join(parent_dir, 'examples/%s' % dir_name)
mave_fname = os.path.join(inputs_dir, 'mave.csv')
mave = pd.read_csv(mave_fname)
nS = len(mave)
seq_length = len(mave['Sequence'][0])
dim = (seq_length)*len(''.join(alphabet))
maps_fname = os.path.join(inputs_dir, 'maps_%s.npy' % attr_map)
maps = np.load(maps_fname, mmap_mode='r')


if 0:
    for i in range(0,50):
        fig, ax = plt.subplots(figsize=(10,1))
        logo = squid.utils.arr2pd(maps[i,278:385,:], alphabet)
        logomaker.Logo(df=logo,
                        ax=ax,
                        fade_below=.5,
                        shade_below=.5,
                        width=.9,
                        center_values=True,
                        font_name='Arial Rounded MT Bold',
                        )
        ax.axis('off')
        plt.rcParams['savefig.dpi'] = 200

        #plt.show()
        plt.savefig(os.path.join(py_dir, 'map_%s.png' % i), facecolor='w', dpi=200)
        plt.close()

if 1:
    for i in [0, 7, 17, 1, 42]:
        print(mave['Sequence'][i][278:385])