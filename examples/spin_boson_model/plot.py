import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

if __name__ == "__main__":
    h5 = h5py.File('sbm.h5', 'r')
    t = np.array(h5.get('t'))
    n1 = np.array(h5.get('Sz'))
    h5.close()
    plt.plot(t, n1, label = r"$\alpha="+str(alpha)+"$", linewidth=2)
    plt.set_xlim([0, 30])
    plt.set_ylim([-0.4, 0.5])
    plt.show()
