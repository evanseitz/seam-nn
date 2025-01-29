import os, sys, time
import numpy as np
from matplotlib import pyplot as plt
import squid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Logo import Logo as LogoClass
from batch_logo import BatchLogo


attributions = np.load('maps_intgrad_N10k.npy')
alphabet = ['A', 'C', 'G', 'T']


if 1: # batch version (new)
    # Create and process all logos
    t1 = time.time()
    batch_logos = BatchLogo(attributions[0:1000,:],
                            #attributions[0:1,80:100,:],
                            alphabet=alphabet,
                            fig_size=[20,2.5],
                            batch_size=50,
                            )

    batch_logos.process_all()
    fig, ax = batch_logos.draw_single(0)  # Draw just logo 0
    t2 = time.time() - t1
    print('Logo time:', t2)
    plt.show()
    fig, ax = batch_logos.draw_single(1)  # Draw just logo 1
    plt.show()
    fig, ax = batch_logos.draw_single(2)  # Draw just logo 2
    plt.show()

if 0: # CPU only version (new) --> DON'T USE THIS OVER THE ABOVE VERSION
    t1 = time.time()
    logo = LogoClass(squid.utils.arr2pd(attributions[0]),
                  fig_size=[20,2.5],
                  )
    t2 = time.time() - t1
    print('Logo time:', t2)
    plt.show()

if 0: # original logomaker version (for benchmarking)
    try:
        import logomaker
        t1 = time.time()
        logo = logomaker.Logo(squid.utils.arr2pd(attributions[0]))
        t2 = time.time() - t1
        print('Logo time:', t2)
        plt.show()
    except ImportError:
        print("Original logomaker not installed. To benchmark against it:")
        print("pip install logomaker")



