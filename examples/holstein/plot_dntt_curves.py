import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse

import matplotlib.animation as animation

def gen_lattice(a, b, N):
    res = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            if j % 2 == 0:
                res[i, j, :] = np.array( [ i*a, (j//2)*b, i*N+j])
            else:
                res[i, j, :] = np.array( [ i*a+a/2, (j//2)*b+b/2, i*N+j])
    return res


def plot_mobility(fname):
    a = 7.662
    b = 6.187
    x = gen_lattice(a, b, N)

    x0 = x[N//2, N//2, 0]
    y0 = x[N//2, N//2, 1]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.sqrt((x[i, j, 0]-x0)**2 + (x[i, j, 1]-y0)**2)

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

    tot = np.sum(res, axis=0)
    ind = np.argmax(tot == 0)
    if ind == 0:
        ind = res.shape[1]

    d = 

def plot(fnames):
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

    logdata = res#np.log(res+1e-12)/np.log(10)

    scatter = plt.scatter(x[:, :, 0], x[:, :, 1], c=logdata[:, 0].reshape((N,N)).T.flatten(), cmap='binary', vmin=0, vmax=0.5, s=300)

    tot = np.sum(res, axis=0)
    ind = np.argmax(tot == 0)
    if ind == 0:
        ind = res.shape[1]

    def animate(i):
        print(i, logdata[:, i])
        scatter.set_array(logdata[:, i].reshape((N,N)).T.flatten())
        return scatter, 

    anim = animation.FuncAnimation(fig, animate, frames = ind, interval = 100, blit=True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density matrix elements output by a heom calculation.')
    parser.add_argument('fname', type=str)

    args = parser.parse_args()
    plot(args.fname)


