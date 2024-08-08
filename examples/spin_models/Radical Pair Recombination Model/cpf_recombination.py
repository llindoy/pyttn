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

def hamiltonian(w, A, ks, kt):
    Nb = len(A)
    N = Nb+2
    #set up the Hamiltonian
    H = SOP(N)

    H += -1*w*sOP("sz", 0)
    H += -1*w*sOP("sz", 1)

    #add on the recombination super operator
    H += (-0.25j*ks-0.75j*kt)
    H += -1.0j*(-ks+kt)*(sOP("sx", 0) * sOP("sx", 1) )
    H += -1.0j*(-ks+kt)*(sOP("sy", 0) * sOP("sy", 1) )
    H += -1.0j*(-ks+kt)*(sOP("sz", 0) * sOP("sz", 1) )

    for i in range(Nb):
        H += A[i] * (sOP("sx", 0) * sOP("sx", i+2) )
        H += A[i] * (sOP("sy", 0) * sOP("sy", i+2) )
        H += A[i] * (sOP("sz", 0) * sOP("sz", i+2) )

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
    

def csm_dynamics(w, A, chi, dt, nstep = 1, yrange=[0, 0.25], nsamples=100, sampling_scheme = 'cs', ofname = 'csm.h5'):
    Nb = len(A)
    N = Nb+2
    H = hamiltonian(w, A)
    PS = SOP(N)
    PS += 1.0
    PS += (-1.0)*sOP("sx", 0)*sOP("sx", 1)
    PS += (-1.0)*sOP("sy", 0)*sOP("sy", 1)
    PS += (-1.0)*sOP("sz", 0)*sOP("sz", 1)

    bath_dims = [2 for i in range(Nb)]

    #set up the system information
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = spin_mode(2)

    #construct the tree topology used for these calculations.  Here we are using a binary tree to partition the up and down spin sectors
    #with the impurity orbitals taking the first site on each subtree
    topo = ntree("(1(4(2(2))(2(2)))(4))")

    degree = 2 
    #then we are adding on trees for the bath degrees of freedom
    ntreeBuilder.mlmctdh_subtree(topo()[1], bath_dims, degree, chi)
    ntreeBuilder.sanitise(topo)

    A = ttn(topo, dtype=np.complex128)

    h = sop_operator(H, A, sysinf)
    Ps = sop_operator(PS, A, sysinf)

    res = np.zeros((nsamples, nstep+1))
    mel = matrix_element(Ps, A)

    for sample in range(nsamples):
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
        sweep = tdvp(A, h, krylov_dim = 8)
        sweep.dt = dt
        sweep.coefficient = -1.0j

        res[sample, 0] = np.real(mel(Ps, A))
        #do the time evolution
        for i in range(nstep):
            t1 = time.time()
            sweep.step(A, h)
            t2 = time.time()
            res[sample, i+1] = np.real(mel(Ps, A))
            print(i)

        
        h5 = h5py.File(ofname, 'w')
        h5.create_dataset('Ps', data=res)
        h5.close()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sampling based dynamics of the CPF radical pair triad')
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--nsamples', type=int, default=512)
    parser.add_argument('--fname', type = str, default='cpf.h5')
    args = parser.parse_args()

    gammae = 176 #mus^-1 mT^-1

    B = 
    ks = 1.8e1      #mus^-1
    kt = 7.1e-2.0   #mus^-1

    N = args.N
    chi = args.chi
    nsamples = args.nsamples

    A = None
    if(args.type == 'uniform'):
        A = hyperfines_uniform(N)
    else:
        A = hyperfines_exp(48, N)

    csm_dynamics(A, 0.0, 16, 0.05, nstep = 20, nsamples=nsamples)

