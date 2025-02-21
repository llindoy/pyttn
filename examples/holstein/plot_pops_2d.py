import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

import matplotlib.animation as animation

def plot(fname):
    pars = []
    t = None
    N=0
    try:
        h5 = None
        h5 = h5py.File(fname, 'r')
        t = np.array(h5.get('t'))
        N = int(np.sqrt(len(h5.keys())-2))
        res = np.zeros((N*N, len(t)))
        for i in range(N):
            for j in range(N):
                label = '|'+str(i)+','+str(j)+'><'+str(i)+','+str(j)+'|'
                res[i*N+j, :] = np.array(h5.get(label))

        h5.close()
    except:
        print("Failed to read input file")
        exit()

    fig = plt.figure()
    logdata = np.log(res+1e-12)/np.log(10)
    im = plt.imshow(logdata[:, 0].reshape((N, N)).T, cmap='binary', vmin=-1.8, vmax=-0.25)
    tot = np.sum(res, axis=0)
    ind = np.argmax(tot == 0)
    if ind == 0:
        ind = res.shape[1]

    def animate(i):
        im.set_array(logdata[:, i].reshape((N, N)).T)
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames = ind, interval = 100, blit=True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', type=str)

    args = parser.parse_args()
    plot(args.fname)


