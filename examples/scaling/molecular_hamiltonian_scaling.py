import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy
sys.path.append("../../")

from pyttn import *

import memory_profiler as mp

#@profile
def do_step(A, h, sweep, nstep):
    timings = np.zeros(nstep)
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        timings[i] = t2-t1

    stdev = 0
    if(nstep > 1):
        stdev = np.std(timings)
    return np.mean(timings), stdev

#@profile
def molecular_hamiltonian_test(t, U, chi, dt, nstep = 1, degree = 2, compress = True):
    sys = models.electronic_structure(t, U)
    H = sys.hamiltonian()
    sysinf = sys.system_info()

    N= t.shape[0]
    mode_dims = [2 for i in range(N)]
    #and add the node that forms the root of the bath
    topo = ntreeBuilder.mlmctdh_tree(mode_dims, degree, chi)
    ntreeBuilder.sanitise(topo)

    A = ttn(topo, dtype=np.complex128)
    A.random()

    start_mem = mp.memory_usage(max_usage=True, interval=.00001)
    h = sop_operator(H, A, sysinf, compress=compress)
    mel = matrix_element(A)

    sweep = tdvp(A, h, krylov_dim = 12)

    sweep.dt = dt
    sweep.coefficient = 1.0j
    sweep.prepare_environment(A, h) 

    stop_mem = mp.memory_usage(max_usage=True, interval=.00001)
    print(start_mem, stop_mem)

    return *do_step(A, h, sweep, nstep), stop_mem-start_mem


def sop_timing(nb, compress, nstep=10):
    chi=20
    t = np.random.rand(nb, nb)
    U = np.random.rand(nb, nb, nb, nb)
    Ut = copy.deepcopy(U)
    Ut = (Ut.reshape((nb*nb, nb*nb)).T).reshape((nb, nb, nb, nb))
    t = (t + t.T)/2.0
    U = (U + Ut)/2.0
    m, std, mem = molecular_hamiltonian_test(t, U, chi, 0.001, nstep=nstep, compress = compress)
    return m, std, mem

def system_size_scaling_SOP_vs_hSOP():
    chi = 20
    Nsys = [12]#, 7, 8, 9, 10]

    timings_hSOP = []
    stdevs_hSOP = []
    mem_hSOP = []

    timings_SOP = []
    stdevs_SOP = []
    mem_SOP = []

    nstep=1
    for nb in Nsys:
        m, std, mem = sop_timing(nb, False, nstep=nstep)
        timings_SOP.append(m)
        stdevs_SOP.append(std)
        mem_SOP.append(mem)
        print(nb,  timings_SOP[-1], stdevs_SOP[-1], mem_SOP[-1])

    for nb in Nsys:
        m, std, mem = sop_timing(nb, FTrue, nstep=nstep)
        timings_hSOP.append(m)
        stdevs_hSOP.append(std)
        mem_hSOP.append(mem)
        print(nb, timings_hSOP[-1], stdevs_hSOP[-1], mem_hSOP[-1])


    plt.figure(1)
    plt.semilogy(Nsys, timings_hSOP)
    plt.semilogy(Nsys, timings_SOP)

    plt.figure(2)
    plt.semilogy(Nsys, mem_hSOP)
    plt.semilogy(Nsys, mem_SOP)
    plt.show()

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Molecular Hamiltonian Scaling Test.")
    parser.add_argument('N', type=int)
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--nstep', type=int,  default=1)

    args = parser.parse_args()

    label = "hSOP"
    if not args.compress:
        label="SOP"

    m, std, mem = sop_timing(args.N, args.compress, nstep=args.nstep)
    print(args.N, label, m, std, mem)
