import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def plot(fnames):
    for fname in fnames:
        t = None
        try:
            h5 = None
            h5 = h5py.File(fname, 'r')
            t = np.array(h5.get('t'))
            nu = np.array(h5.get('n_u'))
            nd = np.array(h5.get('n_d'))
            h5.close()
            plt.plot(t, np.real(nd-nu), '-', label=r'$S_z$'+fname)
        except:
            print("Failed to read input file")
            continue
    #plt.ylim([0, 1])
    plt.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', nargs='+')

    args = parser.parse_args()
    plot(args.fname)


