import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import copy
import h5py

sys.path.append("../../")
from pyttn import *
from pyrazine_tree import *
from pyrazine_hamiltonian import *

fs = 41.341374575751

def pyrazine_dynamics(N1, N2, N3, N4, N5, nstep, dt, adaptive = True, spawning_threshold=1e-6, unoccupied_threshold=1e-4, nunoccupied=0, ofname='pyrazine.h5'):

    N = 25
    #set up the vibrational basis set sizes
    m = [40, 32, 20, 12, 8, 4, 8, 24, 24, 8, 8, 24, 20, 4, 72, 80, 6, 20, 6, 6, 6, 32, 6, 4]
    composite_modes = [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
    Nc = len(composite_modes)

    mc = []
    #set up the system information object
    sysinf = system_modes(Nc+1)
    sysinf[0] = generic_mode(2)
    for ind, comb in enumerate(composite_modes):
        sysinf[ind+1] = [boson_mode(m[x]) for x in comb]
        mc.append(sysinf[ind+1].lhd())

    N1_0 = N1
    N2_0 = N2
    N3_0 = N3
    N4_0 = N4
    N5_0 = N5
    if(adaptive):
        N1_0 = min(8, N1)
        N2_0 = min(8, N2)
        N3_0 = min(6, N3)
        N4_0 = min(4, N4)
        N5_0 = min(6, N5)

    #build topology and capacity trees
    topo = build_topology_mode_combination(N1_0,N2_0,N3_0,N4_0,N5_0,mc)
    capacity= build_topology_mode_combination(N1, N2, N3, N4, N5, mc)

    #set up the sum of product operator Hamiltonian
    H, opdict = hamiltonian(m)  

    #setup the wavefunction
    A = ttn(topo, capacity, dtype=np.complex128)

    state = np.zeros(Nc+1, dtype=int)
    state[0]=1
    A.set_state(state)

    B = ttn(topo, dtype=np.complex128)
    state = np.zeros(Nc+1, dtype=int)
    state[0]=1
    B.set_state(state)

    #setup the hierarchical SOP hamiltonian
    h = sop_operator(H, A, sysinf, opdict)

    mel = matrix_element(A)

    #setup the evolution object
    sweepA = None
    if(adaptive):
        sweepA = tdvp(A, h, krylov_dim = 16, expansion='subspace')
        sweepA.spawning_threshold = spawning_threshold
        sweepA.unoccupied_threshold=unoccupied_threshold
        sweepA.minimum_unoccupied=nunoccupied
    else:
        sweep = tdvp(A, h, krylov_dim = 12)
        
    sweepA.expmv_tol=1e-12
    sweepA.dt = dt
    sweepA.coefficient = -1.0j

    res = np.zeros(nstep+1, dtype=np.complex128)
    maxchi = np.zeros(nstep+1)
    res[0] = mel(B, A)
    maxchi[0] = A.maximum_bond_dimension()

    tp = 0
    ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), 5)
    for i in range(5):
        dti = ts[i]-tp
        sweepA.dt = dti
        sweepA.step(A, h)
        tp = ts[i]

    #B = copy.deepcopy(A)
    #B.conj()
    res[1] = mel(B, A)
    maxchi[1] = A.maximum_bond_dimension()

    sweepA.dt = dt

    for i in range(1, nstep):
        t1 = time.time()
        sweepA.step(A, h)
        t2 = time.time()
        #B = copy.deepcopy(A)
        #B.conj()
        res[i+1] = mel(B, A)
        maxchi[i+1] = A.maximum_bond_dimension()
        print(i, res[i+1])

        if(i % 1 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt/fs))
            h5.create_dataset('a(t)', data=res)
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt/fs))
    h5.create_dataset('a(t)', data=res)
    h5.create_dataset('maxchi', data=maxchi)
    h5.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #exponential bath cutoff parameters
    parser.add_argument('--N1', type = int, default=32)
    parser.add_argument('--N2', type = int, default=32)
    parser.add_argument('--N3', type = int, default=24)
    parser.add_argument('--N4', type = int, default=12)
    parser.add_argument('--N5', type = int, default=16)

    #output file name
    parser.add_argument('--fname', type=str, default='pyrazine.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-6)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)


    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--tmax', type=float, default=150)

    args = parser.parse_args()

    nsteps = int(args.tmax/args.dt)+1
    pyrazine_dynamics(args.N1, args.N2, args.N3, args.N4, args.N5, nsteps, args.dt*fs, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace)
