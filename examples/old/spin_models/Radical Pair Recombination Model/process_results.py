import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the output of the csm_sampling code')
    parser.add_argument('fname', type = str)
    args = parser.parse_args()

    h5 = h5py.File(args.fname, 'r')
    t = np.array(h5.get('t'))
    n1 = np.array(h5.get('Sz'))
    h5.close()

    Ns = (n1[:, 0] == 0.0).argmax()
    if(Ns == 0):
        Ns = n1.shape[0]

    print(Ns)
    x = np.mean(n1[:Ns, :], axis =0)
    dx = 2.0*np.std(n1[:Ns, :], axis=0, ddof=1)/np.sqrt(Ns)

    plt.fill_between(t, x-dx, x+dx, color='gray', alpha=0.2)
    plt.plot(t, x, color='k')
    plt.show()
