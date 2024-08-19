from sbm import *

def run(spawning_threshold = 1e-5):
    alpha = 1
    wc  = 5
    s = 1

    eps = 0.0
    delta = 1.0

    beta = None

    Ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    chi = 20
    degree = 2
    nbose = 20

    dt = 0.005
    tmax = 30

    subspace = True
    nunoccupied = 0
    unoccupied_threshold = 1e-4

    nstep = int(tmax/dt)+1


    ts = []
    for N in Ns:
        ts.append(sbm_dynamics(N, alpha, wc, s, eps, delta, chi, nbose, dt, beta = beta, nstep = nstep, nunoccupied=nunoccupied, spawning_threshold=spawning_threshold, unoccupied_threshold = unoccupied_threshold, adaptive = subspace, degree = degree))
        h5 = h5py.File("Nb_scaling.h5", 'w')
        h5.create_dataset('Ns', data=np.array(Ns))
        h5.create_dataset('ts', data=np.array(ts))
        h5.close()


run(1e-5)
