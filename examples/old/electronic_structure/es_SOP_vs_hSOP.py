import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
sys.path.append("../../")

from pyttn import *


def molecular_hamiltonian_test(t, U, chi, dt, nstep = 1, degree = 2, compress = True):
    sys = electronic_structure(t, U)
    H = sys.hamiltonian()
    sysinf = sys.system_info()
    
    N= t.shape[0]
    mode_dims = [2 for i in range(N)]
    #and add the node that forms the root of the bath
    topo = ntreeBuilder.mlmctdh_tree(mode_dims, degree, chi)
    ntreeBuilder.sanitise(topo)

    A = ttn(topo, dtype=np.complex128)
    A.random()

    h = sop_operator(H, A, sysinf, compress=compress)
    mel = matrix_element(A)

    sweep = tdvp(A, h, krylov_dim = 12)
    sweep.dt = dt
    sweep.coefficient = 1.0j
    sweep.prepare_environment(A, h)

    timings = np.zeros(nstep)
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h, dt)
        t2 = time.time()
        timings[i] = t2-t1

    stdev = 0
    if(nstep > 1):
        stdev = np.std(timings)
    return np.mean(timings), stdev

def system_size_scaling_SOP_vs_hSOP():
    chi = 20
    Nsys = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    timings_hSOP = []
    stdevs_hSOP = []

    timings_SOP = []
    stdevs_SOP = []
    for nb in Nsys:
        print(nb)
        t = np.random.rand(nb, nb)
        U = np.random.rand(nb, nb, nb, nb)
        Ut = copy.deepcopy(U)
        Ut = (Ut.reshape((nb*nb, nb*nb)).T).reshape((nb, nb, nb, nb))
        t = (t + t.T)/2.0
        U = (U + Ut)/2.0
        Ub
        m, std = molecular_hamiltonian_test(t, U, chi, 0.001, nstep=1, compress = True)
        timings_hSOP.append(m)
        stdevs_hSOP.append(std)
        print(nb, timings_hSOP[-1], stdevs_hSOP[-1])

    plt.semilogy(Nsys, timings_hSOP)
    for nb in Nsys:
        t = np.random.rand(nb, nb)
        U = np.random.rand(nb, nb, nb, nb)
        Ut = copy.deepcopy(U)
        Ut = (Ut.reshape((nb*nb, nb*nb)).T).reshape((nb, nb, nb, nb))
        t = (t + t.T)/2.0
        U = (U + Ut)/2.0
        m, std = molecular_hamiltonian_test(t, U, chi, 0.001, nstep=1, compress = False)
        timings_SOP.append(m)
        stdevs_SOP.append(std)
        print(nb, timings_hSOP[-1], stdevs_hSOP[-1], timings_SOP[-1], stdevs_SOP[-1])

    plt.figure(1)
    plt.semilogy(Nsys, timings_SOP)
    plt.show()

system_size_scaling_SOP_vs_hSOP()
