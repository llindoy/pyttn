import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import copy
import h5py

from pyttn import system_modes, generic_mode, boson_mode

from pyttn import ttn, sop_operator, matrix_element, tdvp
from pyrazine_tree import build_topology_mode_combination
from pyrazine_hamiltonian import hamiltonian

fs = 41.341374575751

def run_initial_step(A, h, sweep, dt, nstep=10):
    tp = 0
    ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), nstep)
    for i in range(nstep):
        dti = ts[i]-tp
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
    return A, h, sweep

def initial_bond_dimensions(N1, N2, N3, N4, N5, adaptive=True):
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
    return N1_0, N2_0, N3_0, N4_0, N5_0


def output_results(ofname, timepoints, res, maxchi, runtime):
    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=timepoints)
    h5.create_dataset('a(t)', data=res)
    h5.create_dataset('maxchi', data=maxchi)
    h5.create_dataset('runtime', data=runtime*np.ones(1))
    h5.close()

def pyrazine_dynamics(N1, N2, N3, N4, N5, tmax, dt, adaptive = True, spawning_threshold=1e-6, unoccupied_threshold=1e-4, nunoccupied=0, ofname='pyrazine.h5', output_skip=1):
    #Here we half the total integration time as we are computing a(t) = <\psi(t/2)^*|\psi(t/2)>
    nsteps = int(tmax/(2*dt))+1

    #The total number of modes in the system
    N = 25          

    #The dimension of each of the bosonic modes
    m = [40, 32, 20, 12, 8, 4, 8, 24, 24, 8, 8, 24, 20, 4, 72, 80, 6, 20, 6, 6, 6, 32, 6, 4]

    #The composite mode definition
    composite_modes = [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
    Nc = len(composite_modes)

    composite_mode_dimensions = []
    #set up the system information object
    sysinf = system_modes(Nc+1)
    sysinf[0] = generic_mode(2)
    for ind, comb in enumerate(composite_modes):
        sysinf[ind+1] = [boson_mode(m[x]) for x in comb]
        composite_mode_dimensions.append(sysinf[ind+1].lhd())

    #set up the initial bond dimensions for the tree structure
    N1_0, N2_0, N3_0, N4_0, N5_0 = initial_bond_dimensions(N1, N2, N3, N4, N5, adaptive=adaptive)

    #build topology and capacity trees
    topo = build_topology_mode_combination(N1_0,N2_0,N3_0,N4_0,N5_0,composite_mode_dimensions)
    capacity= build_topology_mode_combination(N1, N2, N3, N4, N5, composite_mode_dimensions)

    #set up the sum of product operator Hamiltonian and operator dictionary
    H, opdict = hamiltonian()  

    #setup the wavefunction
    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_seed(0)
    state = np.zeros(Nc+1, dtype=int)
    state[0]=1
    A.set_state(state)

    #setup a reference wavefunction 
    B = ttn(topo, dtype=np.complex128)
    state = np.zeros(Nc+1, dtype=int)
    state[0]=1
    B.set_state(state)

    #setup the hierarchical SOP hamiltonian
    h = sop_operator(H, A, sysinf, opdict)
    mel = matrix_element(A)

    #setup the evolution object
    sweep = None
    if(adaptive):
        sweep = tdvp(A, h, krylov_dim = 12, expansion='subspace', subspace_krylov_dim=12, subspace_neigs=6)
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold=unoccupied_threshold
        sweep.minimum_unoccupied=nunoccupied
    else:
        sweep = tdvp(A, h, krylov_dim = 12)
        
    sweep.dt = dt
    sweep.coefficient = -1.0j

    res = np.zeros(nsteps+1, dtype=np.complex128)
    maxchi = np.zeros(nsteps+1)
    res[0] = mel(B, A)
    maxchi[0] = A.maximum_bond_dimension()

    t1 = time.time()

    #perform a set of steps with a logarithmic timestep discretisation
    A, h, sweep = run_initial_step(A, h, sweep, dt)

    B=copy.deepcopy(A)      #copy A into B 
    B.conj()                #and conjugate the result
    res[1] = mel(B, A)
    maxchi[1] = A.maximum_bond_dimension()

    sweep.dt = dt

    #multiply the time-points object by 2 as we are only evolving the wavefunction to tmax/2
    #but still extracting the correlation function to tmax
    timepoints = (np.arange(nsteps+1)*dt*2/fs)

    for i in range(1, nsteps):
        print(i, nsteps)
        sys.stdout.flush()
        sweep.step(A, h)
        
        B=copy.deepcopy(A)      #copy A into B 
        B.conj()                #and conjugate the result
        t2 = time.time()

        res[i+1] = mel(B, A)    #evaluate <psi(t/2)^*|psi(t/2)>
        maxchi[i+1] = A.maximum_bond_dimension()

        if(i % output_skip == 0):
            output_results(ofname, timepoints, res, maxchi, (t2-t1))

    output_results(ofname, timepoints, res, maxchi, (t2-t1))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #The bond dimension parameters
    parser.add_argument('--N1', type = int, default=16)
    parser.add_argument('--N2', type = int, default=16)
    parser.add_argument('--N3', type = int, default=10)
    parser.add_argument('--N4', type = int, default=8)
    parser.add_argument('--N5', type = int, default=12)

    #output file name
    parser.add_argument('--fname', type=str, default='pyrazine.h5')

    #subspace expansion parameters
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-12)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.25)
    parser.add_argument('--tmax', type=float, default=150)

    parser.add_argument('--output_skip', type=int, default=1)

    args = parser.parse_args()

    pyrazine_dynamics(args.N1, args.N2, args.N3, args.N4, args.N5, args.tmax*fs, args.dt*fs, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, output_skip=args.output_skip)
