import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def plot(fnames, Ns):
    colors = ['k', 'r','b']
    styles = ['-', '--','-.']
    c = 0
    for fname in fnames:
        pars = []
        t = None
        h5 = None
        h5 = h5py.File(fname, 'r')
        t = np.array(h5.get('t'))
        for N  in range(Ns):
            if 'rSz'+str(N) in h5:
                pars.append(np.array(h5.get('rSz'+str(N))))
            else:
                pars.append(np.array(h5.get('Sz'+str(N))))

        h5.close()

        for N in range(Ns):
            plt.plot(t, 0.5-0.5*np.real(pars[N]/np.amax(np.abs(pars[1]))), styles[c], label='Sz'+str(N)+" "+fname)
        c=c+1
        c=c%3
    plt.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', nargs='+')
    parser.add_argument('--Ns', type =int, default=21)

    args = parser.parse_args()
    plot(args.fname, args.Ns)


