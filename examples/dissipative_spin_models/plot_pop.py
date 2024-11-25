import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

def plot(fnames, Ns):
    colors = ['k', 'r','b']
    styles = ['-', '--','-.']
    c = 0
    for fname in fnames:
        pars = None
        t = None
        try:
            h5 = None
            h5 = h5py.File(fname, 'r')
            t = np.array(h5.get('t'))
            for N  in range(Ns):
                if isinstance(pars, np.ndarray):
                    if 'rSz'+str(N) in h5:
                        pars += (np.array(h5.get('rSz'+str(N))))
                    else:
                        pars += (np.array(h5.get('Sz'+str(N))))
                else:
                    if 'rSz'+str(N) in h5:
                        pars = (np.array(h5.get('rSz'+str(N))))
                    else:
                        pars = (np.array(h5.get('Sz'+str(N))))

            h5.close()
        except:
            print("Failed to read input file")
            continue

        plt.plot(t, (0.5-np.real(pars)), styles[c], label='Sztot_'+fname)
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


