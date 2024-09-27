import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def plot(fnames):
    for fname in fnames:
        t = None
        res = None
        try:
            h5 = None
            h5 = h5py.File(fname, 'r')
            t = np.array(h5.get('t'))
            res = np.array(h5.get('Sz'))
            h5.close()
        except:
            print("Failed to read input file")
            continue

        try:
            #res = res[np.nonzero(res)]
            rc = np.real(res[:, 0])
            print(rc)
            print(rc>0)
            ind = (rc>0).argmin()
            print(ind)
            val = np.mean(np.real(res[:ind, :]), axis=0)*4
            #for i in range(ind):
            #    plt.plot(t, np.real(res[i, :])*4)
            plt.plot(t, val, linewidth=5)
            err = 2.0*np.std(4*np.real(res[:ind, :]), axis=0, ddof=1)/np.sqrt(ind+1.0)
            plt.fill_between(t, val-2*err, val+2*err, alpha=0.3)
        except:
            print("Failed to plot")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', nargs='+')

    args = parser.parse_args()
    plot(args.fname)


