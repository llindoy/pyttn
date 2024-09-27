import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def plot(fnames):
    labels = [r'$\epsilon=0.05$', r'$\epsilon=0.9$', r'$\nu=1$', r'$\nu=10$']
    for fname, label in zip(fnames, labels):
        t = None
        try:
            h5 = None
            h5 = h5py.File(fname, 'r')
            t = np.array(h5.get('t'))
            nu = np.array(h5.get('n_u'))
            nd = np.array(h5.get('n_d'))
            h5.close()
            plt.plot(t, np.real(nu), '-', label=label)
        except:
            print("Failed to read input file")
            continue
    plt.ylim([0.02, 0.07])
    plt.legend(frameon=False)
    plt.xlim([18, 26])
    plt.savefig("time_dependent_nd.pdf", bbox_inches='tight')
    plt.xlabel("Wt")
    plt.ylabel(r"$S_z$")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', nargs='+')

    args = parser.parse_args()
    plot(args.fname)


