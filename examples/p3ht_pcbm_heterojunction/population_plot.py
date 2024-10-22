import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse


def plot_imag(fnames):
    for fname in fnames:
        t = None
        res = None
        try:
            h5 = None
            h5 = h5py.File(fname, 'r')
            t = np.array(h5.get('t'))

            res = np.zeros((len(t), 26))
            for i in range(13):
                print(i)
                d1 = h5.get('|CS%d><CS%d|'%(i, i))
                d2 = h5.get('|LE%d><LE%d|'%(i, i))
                res[:, 12-i] = np.real(d1)
                res[:, 13+i] = np.real(d2)

            h5.close()
        except:
            print("Failed to read input file")
            continue

        fig, ax = plt.subplots(num=fname)
        im = ax.imshow(res, aspect=0.125, interpolation='Nearest', vmin=0, vmax=0.25, origin='lower', extent=[0, 25, 0, 200])

        labels = ['' for i in range(26)]
        labels[0] = 'CS$_{13}$'
        labels[12] = 'CS$_{1}$'
        labels[13] = 'LE$_{1}$'
        labels[25] = 'LE$_{13}$'
        ax.set_xticks([i for i in range(26)], labels=labels)
    plt.show()


def plot_diff(fnames):
    resvals = []
    for fname in fnames:
        t = None
        res = None
        try:
            h5 = None
            h5 = h5py.File(fname, 'r')
            t = np.array(h5.get('t'))

            res = np.zeros((len(t), 26))
            for i in range(13):
                print(i)
                d1 = h5.get('|CS%d><CS%d|'%(i, i))
                d2 = h5.get('|LE%d><LE%d|'%(i, i))
                res[:, 12-i] = np.real(d1)
                res[:, 13+i] = np.real(d2)
            resvals.append(res)
            h5.close()
        except:
            print("Failed to read input file")
            continue

    for i in range(len(resvals)):
        fig, ax = plt.subplots(num=fname)
        im = ax.imshow(resvals[i]-resvals[0], aspect=0.125, interpolation=None, vmin=-0.05, vmax=0.05, origin='lower', extent=[0, 25, 0, 200])

        labels = ['' for i in range(26)]
        labels[0] = 'CS$_{13}$'
        labels[12] = 'CS$_{1}$'
        labels[13] = 'LE$_{1}$'
        labels[25] = 'LE$_{13}$'
        ax.set_xticks([i for i in range(26)], labels=labels)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', nargs='+')

    args = parser.parse_args()
    plot_imag(args.fname)


