import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs
from pyttn.oqs.heom import softmspace

from numba import jit


def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta = None, Ncut = 20, nstep = 1, nbeta=100, Nw = 7.5, ofname='sbm_purification.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0):

    geom='chain'
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.bosonic_bath(J, sOP("sz", 0), beta=None)

    #and discretise the bath getting the star Hamiltonian parameters using the orthpol discretisation strategy
    g,w = bath.discretise(Nb, Nw*wc, method='orthopol')

    #ct = bath.Ct(t, Nw*wc)

    #set up the total Hamiltonian
    N = Nb+1
    H = SOP(N)

    #and add on the system parts
    H += eps*sOP("sz", 0)
    H += 2*delta*sOP("sx", 0)

    #now add on the bath bits getting additionally getting a frequency parameter that can be used in energy based
    #truncation schemes
    H, w = oqs.add_bath_hamiltonian(H, bath.Sp, 2*g, w, geom=geom)

    mode_dims = [nbose for i in range(Nb)]
    #mode_dims = [min(max(4, int(wc*Ncut/l[i])), nbose) for i in range(Nb)]

    #setup the system information object
    sysinf = system_modes(N)
    sysinf[0] = spin_mode(2)
    for i in range(Nb):
        sysinf[i+1] = boson_mode(mode_dims[i])

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 2

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


    A = ttn(topo, capacity, dtype=np.complex128, purification=True)
    A.set_identity_purification()

    B = ttn(topo, capacity, dtype=np.complex128, purification=True)
    B.set_identity_purification()

    C = ttn(topo, capacity, dtype=np.complex128, purification=False)
    C.set_state([0 for x in range(N)])


    mel = matrix_element(A)

    sz_op = site_operator([0.5, 0.5, -0.5, -0.5], optype="diagonal_matrix", mode=0)

    #estimate the ground state energy to allow for a less dramatic shift of the dynamics
    if(True):
        h = sop_operator(H, C, sysinf, identity_opt=True, compress=True)
        sweepC = dmrg(C, h, krylov_dim = 12)
        sweepC.step(C, h)
        print(sweepC.E())
        sweepC.step(C, h)
        print(sweepC.E())
        sweepC.step(C, h)
        print(sweepC.E())
        sweepC.step(C, h)
        E = sweepC.E()
        H -= E

    h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)
    sweepA = None
    if not adaptive:
        sweepA = tdvp(A, h, krylov_dim = 12)
    else:
        sweepA = tdvp(A, h, krylov_dim = 12, expansion='subspace')
        sweepA.spawning_threshold = spawning_threshold
        sweepA.unoccupied_threshold=unoccupied_threshold
        sweepA.minimum_unoccupied=nunoccupied
    sweepA.coefficient = -1.0


    sweepB = None
    if not adaptive:
        sweepB = tdvp(B, h, krylov_dim = 12)
    else:
        sweepB = tdvp(B, h, krylov_dim = 12, expansion='subspace')
        sweepB.spawning_threshold = spawning_threshold
        sweepB.unoccupied_threshold=unoccupied_threshold
        sweepB.minimum_unoccupied=nunoccupied
    sweepB.coefficient = -1.0

    beta_steps = softmspace(1e-9, beta/2.0, nbeta)
    beta_p = 0
    for i in range(beta_steps.shape[0]):
        sweepA.dt = beta_steps[i]-beta_p
        beta_p = beta_steps[i]
        sweepA.step(A, h)
        rhoA = A.normalise()
        print(sweepA.t, i, A.maximum_bond_dimension(), rhoA)
        sys.stdout.flush()
    #perform the imaginary time evolution steps
    B = copy.deepcopy(A)
    B.apply_one_body_operator(sz_op)


    res = np.zeros(nstep+1)
    maxchi = np.zeros(nstep+1)
    res[0] = np.real(mel(sz_op, B, A))
    maxchi[0] = A.maximum_bond_dimension()

    sweepA.t = 0
    sweepA.dt = dt
    sweepA.coefficient = -1.0j
    sweepA.use_time_dependent_hamiltonian = True

    sweepB.t = 0
    sweepB.dt = dt
    sweepB.coefficient = -1.0j
    sweepB.use_time_dependent_hamiltonian = True

    for i in range(nstep):
        t1 = time.time()
        sweepA.step(A, h)
        sweepB.step(B, h)
        t2 = time.time()
        res[i+1] = np.real(mel(sz_op, B, A))
        maxchi[i+1] = A.maximum_bond_dimension()

        print((i+1)*dt, res[i+1], maxchi[i+1])
        sys.stdout.flush()
            
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

    #system hamiltonian parameters
    parser.add_argument('--eps', type = float, default=0)
    parser.add_argument('--delta', type = float, default=1)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=5)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--degree', type=int, default=2)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=20)
    parser.add_argument('--nbeta', type=int, default=100)

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
    sbm_dynamics(args.N, args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, nbeta = args.nbeta, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree)
