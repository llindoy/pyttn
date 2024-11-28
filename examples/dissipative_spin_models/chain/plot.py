import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def plot(fnames, params):
    #colors = ['k', 'r','b']
    c = 0
    for fname in fnames:
        pars = []
        t = None
        try:
            h5 = None
            h5 = h5py.File(fname, 'r')
            t = np.array(h5.get('t'))
            for par  in params:
                pars.append(np.array(h5.get(par)))
            print(fname, np.array(h5.get('time')))
            h5.close()
        except:
            print("Failed to read input file")
            continue

        for par, label in zip(pars, params):
            try:
                plt.plot(t, 0.5-np.real(par), '-', label=label+'_'+fname)
                #plt.plot(t, np.real(par)/np.amax(np.real(par)), '-', label=label+'_'+fname)
            except:
                print("Failed to plot: "+label)
        c+=1
        c = c%3
    #plt.ylim([0, 1])
    plt.xlim([0, 40])
    plt.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', nargs='+')
    parser.add_argument('--labels', nargs='+', default = ['Sz'])

    args = parser.parse_args()
    plot(args.fname, args.labels)


