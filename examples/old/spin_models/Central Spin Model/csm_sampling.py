import numpy as np
import time
import sys
import copy
import h5py

sys.path.append("../../../")
from pyttn import *

def hyperfines_exp(N, N0):
    return np.sqrt((1-np.exp(-2.0/(N0-1)))/(1-np.exp(-2*N/(N0-1))))*np.exp(-np.arange(N)/(N0-1))

def hyperfines_uniform(N):
    return np.sqrt(6.0*N/(2.0*N*N+3*N+1)) * (N-np.arange(N))/N

def hamiltonian(B, A):
    Nb = len(A)
    N = Nb+1
    #set up the Hamiltonian
    H = SOP(N)

    H += B*sOP("sz", 0)

    for i in range(Nb):
        H += A[i] * (sOP("sx", 0) * sOP("sx", i+1) )
        H += A[i] * (sOP("sy", 0) * sOP("sy", i+1) )
        H += A[i] * (sOP("sz", 0) * sOP("sz", i+1) )

    return H

#need to make sure this is correctly sampling the spin coherent states
def sample_spin_half_coherent_state():
    x = np.random.normal(size=3)
    r = np.sqrt(x[0]*x[0] + x[1]*x[1]+x[2]*x[2])
    v = x/r
    theta = np.arccos(v[2])
    phi = np.arctan2(v[0], v[1])
    mu = np.exp(1.0j*phi)*np.tan(theta/2.0)

    return np.array([np.sin(theta/2), np.cos(theta/2)*np.exp(-1.0j*phi)])
    

def csm_dynamics(A, B, chi, dt, nstep = 1, yrange=[0, 0.25], nsamples=100, sampling_scheme = 'cs', ofname = 'csm.h5', degree = 2, adaptive=True, spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0, seed = 0):
    Nb = len(A)
    N = Nb+1
    H = hamiltonian(B, A)
    print(H)

    bath_dims = [2 for i in range(Nb)]

    #set up the system information
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = spin_mode(2)

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 4

    #and add the node that forms the root of the bath.  
    #TODO: Add some better functions for handling the construction of tree structures
    topo = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], bath_dims, degree, chi0)
    else:
        ntreeBuilder.mps_subtree(topo()[1], bath_dims, chi0, min(chi0, 2))
    ntreeBuilder.sanitise(topo)

    capacity = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(capacity()[1], bath_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(capacity()[1], bath_dims, chi, min(chi, 2))
    ntreeBuilder.sanitise(capacity)

    np.random.seed(seed)
    #op = site_operator([1/np.sqrt(2), 1.0j/np.sqrt(2), 1/np.sqrt(2), 1.0j/np.sqrt(2)], optype="diagonal_matrix", mode=0)
    sz_op = site_operator([0.5, -0.5], optype="diagonal_matrix", mode=0)

    res = np.zeros((nsamples, nstep+1))
    for sample in range(nsamples):
        A = ttn(topo, capacity, dtype=np.complex128)
        mel = matrix_element(A)
        h = sop_operator(H, A, sysinf)
        if(sampling_scheme == 'cs'):
            state = []
            state.append(np.array([1, 0], dtype=np.complex128))
            for i in range(Nb):
                state.append(sample_spin_half_coherent_state())
            A.set_product(state)
        else:
            dist = [0.5*np.ones(2) for i in range(Nb+1)]
            dist[0][0] = 1
            dist[0][1] = 0
            A.sample_product_state(dist)

        #set up the tdvp engine
        sweep = None
        if not adaptive:
            sweep = tdvp(A, h, krylov_dim = 12)
        else:
            sweep = tdvp(A, h, krylov_dim = 12, expansion='subspace')
            sweep.spawning_threshold = spawning_threshold
            sweep.unoccupied_threshold=unoccupied_threshold
            sweep.minimum_unoccupied=nunoccupied
        sweep.dt = dt
        sweep.coefficient = -1.0j

        res[sample, 0] = np.real(mel(sz_op, A, A))
        #do the time evolution
        for i in range(nstep):
            t1 = time.time()
            sweep.step(A, h)
            t2 = time.time()
            res[sample, i+1] = np.real(mel(sz_op, A, A))
            print((i+1)*dt, nstep*dt, res[sample, i+1], mel(A, A), A.maximum_bond_dimension())

        
        h5 = h5py.File(ofname, 'w')
        h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
        h5.create_dataset('Sz', data=res)
        h5.close()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sampling based dynamics of the central spin model')
    parser.add_argument('N', type = int)
    parser.add_argument('--chi', type=int, default=64)
    parser.add_argument('--nsamples', type=int, default=512)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--type', type = str, default='uniform')
    parser.add_argument('--fname', type=str, default='csm')
    parser.add_argument('--seed', type = int, default=0)

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-5)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)
    args = parser.parse_args()

    N = args.N
    chi = args.chi
    nsamples = args.nsamples

    A = None
    if(args.type == 'uniform'):
        A = hyperfines_uniform(N)
    else:
        A = hyperfines_exp(48, N)

    csm_dynamics(A, 0.0, args.chi, 0.025, nstep = 10000, nsamples=nsamples, ofname = args.fname+"_"+str(args.type)+"_"+str(N)+"_"+str(args.seed)+"_"+str(args.chi)+"_"+str(args.spawning_threshold)+".h5", degree = args.degree, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, seed=args.seed)

