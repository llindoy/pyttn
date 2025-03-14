import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import copy
import argparse
import memory_profiler as mp

import pyttn


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

def electronic_structure_hamiltonian_test(t, U, chi, dt, nstep = 1, degree = 2, compress = True):
    r""" Function for setting up and running tdvp for an electronic structure Hamiltonian computing the time required per step and memory required

    :param t: A matrix containing the 1-electron integrals of the system
    :type t: np.ndarray
    :param U: A tensor containing the 2-electron integrals of the system
    :type param: np.ndarray
    :param chi: The fixed bond dimension used throughout the tensor network
    :type chi: int
    :param dt: The timestep used for integration
    :type dt: float
    :param nstep: The number of steps to run when evaluating mean and stdev of timings (default: 1)
    :type nstep: int, optional
    :param degree: The degree of the tree tensor network used to represent the bath (default: 2)
    :type degree: int, optional
    :param compress: Whether or not to compress the SOP Hamiltonian (default: True)
    :type param: bool, optional
    """
    sys = pyttn.models.electronic_structure(t, U)
    H = sys.hamiltonian()
    sysinf = sys.system_info()

    N= t.shape[0]
    mode_dims = [2 for i in range(N)]
    #and add the node that forms the root of the bath
    topo = pyttn.ntreeBuilder.mlmctdh_tree(mode_dims, degree, chi)
    pyttn.ntreeBuilder.sanitise(topo)

    A = pyttn.ttn(topo, dtype=np.complex128)
    A.random()

    start_mem = mp.memory_usage(max_usage=True, interval=.00001)
    h = pyttn.sop_operator(H, A, sysinf, compress=compress)

    sweep = pyttn.tdvp(A, h, krylov_dim = 12)

    sweep.dt = dt
    sweep.coefficient = -1.0j
    sweep.prepare_environment(A, h) 

    stop_mem = mp.memory_usage(max_usage=True, interval=.00001)
    print(start_mem, stop_mem)

    return *do_step(A, h, sweep, nstep), stop_mem-start_mem


def sop_timing(N, compress, nstep=10):
    r""" Function for setting up and running a random molecular Hamiltonian problem and computing 

    :param N: The number of spin-orbitals to include in the problem
    :type N: int
    :param compress: Whether or not to compress the SOP Hamiltonian
    :type param: bool
    :param nstep: The number of steps to run when evaluating mean and stdev of timings (default: 10)
    :type nstep: int, optional
    """
    chi=20
    t = np.random.rand(N, N)
    U = np.random.rand(N, N, N, N)
    Ut = copy.deepcopy(U)
    Ut = (Ut.reshape((N*N, N*N)).T).reshape((N, N, N, N))
    t = (t + t.T)/2.0
    U = (U + Ut)/2.0
    m, std, mem = electronic_structure_hamiltonian_test(t, U, chi, 0.001, nstep=nstep, compress = compress)
    return m, std, mem

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
