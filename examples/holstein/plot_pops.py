import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def plot(fnames):
    for fname in fnames:
        pars = []
        t = None
        try:
            h5 = None
            h5 = h5py.File(fname, 'r')
            t = np.array(h5.get('t'))
            N = len(h5.keys())-2
            res = np.zeros((N, len(t)))
            for i in range(N):
                label = '|'+str(i)+'><'+str(i)+'|'
                res[i, :] = np.array(h5.get(label))

            h5.close()
        except:
            print("Failed to read input file")
            continue

        for i in range(len(t)):
            res[:, i] = res[:, i]/np.max(res[:, i])

        plt.figure(fname)
        plt.imshow(res)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', nargs='+')

    args = parser.parse_args()
    plot(args.fname)


