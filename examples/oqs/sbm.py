import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs

from numba import jit


def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta = None, Ncut = 20, nstep = 1, Nw = 7.5, geom='star', ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0):

    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.bosonic_bath(J, sOP("sz", 0), beta=beta)

    #and discretise the bath getting the star Hamiltonian parameters using the orthpol discretisation strategy
    g,w = bath.discretise(Nb, Nw*wc, method='orthopol')

    #ct = bath.Ct(t, Nw*wc)

    #set up the total Hamiltonian
    N = Nb+1
    H = SOP(N)

    #and add on the system parts
    H += eps*sOP("sz", 0)
    H += delta*sOP("sx", 0)

    #now add on the bath bits getting additionally getting a frequency parameter that can be used in energy based
    #truncation schemes
    H, w = oqs.add_bath_hamiltonian(H, bath.Sp, 2*g, w, geom=geom)

    #mode_dims = [nbose for i in range(Nb)]
    mode_dims = [min(max(4, int(wc*Ncut/l[i])), nbose) for i in range(Nb)]

    #setup the system information object
    sysinf = system_modes(N)
    sysinf[0] = spin_mode(2)
    for i in range(Nb):
        sysinf[i+1] = boson_mode(mode_dims[i])
    print(sysinf.mode_indices)

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 4

    #and add the node that forms the root of the bath.  
    #TODO: Add some better functions for handling the construction of tree structures
    topo = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi0)
    else:
        ntreeBuilder.mps_subtree(topo()[1], mode_dims, chi0, min(chi0, nbose))
    ntreeBuilder.sanitise(topo)

    capacity = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(capacity()[1], mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(capacity()[1], mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(capacity)


    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(Nb+1)])
    print(topo)

    h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)

    mel = matrix_element(A)

    op = site_operator_complex(
        sOP("sz", 0),
        sysinf
    )

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

    if(geom == 'ipchain'):
        sweep.use_time_dependent_hamiltonian = True

    res = np.zeros(nstep+1)
    maxchi = np.zeros(nstep+1)
    res[0] = np.real(mel(op, A, A))
    maxchi[0] = A.maximum_bond_dimension()

    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h, dt)
        t2 = time.time()
        res[i+1] = np.real(mel(op, A, A))
        maxchi[i+1] = A.maximum_bond_dimension()

        if(i % 100):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            h5.create_dataset('Sz', data=res)
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    h5.create_dataset('Sz', data=res)
    h5.create_dataset('maxchi', data=maxchi)
    h5.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #exponential bath cutoff parameters
    parser.add_argument('alpha', type = float)
    parser.add_argument('--wc', type = float, default=5)
    parser.add_argument('--s', type = float, default=1)

    #number of bath modes
    parser.add_argument('--N', type=int, default=300)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=1)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--degree', type=int, default=2)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=20)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--tmax', type=float, default=30)

    #output file name
    parser.add_argument('--fname', type=str, default='sbm.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-5)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.N, args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree)
