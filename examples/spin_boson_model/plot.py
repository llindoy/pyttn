import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

if __name__ == "__main__":
    h5 = h5py.File('sbm.h5', 'r')
    t = np.array(h5.get('t'))
    n1 = np.array(h5.get('Sz'))
    n2 = np.array(h5.get('maxchi'))
    h5.close()
    plt.figure(1)
    plt.xlim([0, 30])
    plt.plot(t, n1)
    plt.figure(2)
    plt.plot(t, n2)
    plt.xlim([0, 30])
    #plt.ylim([-0.4, 0.5])
    plt.show()
