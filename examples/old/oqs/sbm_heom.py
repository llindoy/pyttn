import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs
from pyttn._pyttn import operator_dictionary_complex
from pyttn._pyttn.linalg import csr_matrix_complex, diagonal_matrix_complex

from numba import jit


def sbm_dynamics(alpha, wc, s, eps, delta, chi, nbose, dt, nbose_min = None, beta = None, Ncut = 20, nstep = 1, Nw = 20, geom='star', ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, aaa_tol=1e-4):

    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.bosonic_bath(J, sOP("sz", 0), beta=beta)

    #and discretise the bath getting the star Hamiltonian parameters using the orthpol discretisation strategy
    dk,zk, Sw_aaa = bath.fitCt(wmax=Nw*wc, aaa_tol=aaa_tol)
    w = np.linspace(-Nw*wc, Nw*wc, 1000)

    Nb = 2*dk.shape[0]
    N = Nb+1

    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    mode_dims = oqs.heom.compute_dimensions(sz, dk, zk, nbose, Lmin=nbose_min)
    b_mode_dims = mode_dims[1:]
    print(N)
    print(mode_dims.shape)

    #set up the total Hamiltonian
    H = SOP(N)

    #and add on the system parts
    Hsys = eps*sz + delta *sx
    H += sOP("lsys", 0)

    for i in range(Nb):
        gk = 0
        hk = 0
        wk = 0

        if(i%2 == 0):
            wk = -1.0j*zk[i//2]
            gk = dk[i//2]/np.sqrt(np.abs(dk[i//2]))
            hk = np.sqrt(np.abs(dk[i//2]))
        else:
            wk = -1.0j*np.conj(zk[i//2])
            gk = -np.conj(dk[i//2])/np.sqrt(np.abs(dk[i//2]))
            hk = np.sqrt(np.abs(dk[i//2]))

        H += hk*sOP("sp", 0)*sOP("a", i+1)

        if i % 2 == 0:
            H += gk*sOP("sm_f", 0)*sOP("adag", i+1)
        else:
            H += gk*sOP("sm_b", 0)*sOP("adag", i+1)
        H += wk*sOP("n", i+1)

    #set up the operator dictionary information
    opdict = operator_dictionary_complex(N)

    #set up the different system operators
    opdict.insert(0, "lsys", site_operator(oqs.heom.commutator(Hsys), optype="matrix", mode=0))
    opdict.insert(0, "sp", site_operator(oqs.heom.Sp(sz), optype="matrix", mode=0))
    opdict.insert(0, "sm_f", site_operator(oqs.heom.Sm(sz, True), optype="matrix", mode=0))
    opdict.insert(0, "sm_b", site_operator(oqs.heom.Sm(sz, False), optype="matrix", mode=0))

    #setup the system information object
    sysinf = system_modes(N)
    sysinf[0] = generic_mode(mode_dims[0])
    for i in range(Nb):
        sysinf[i+1] = boson_mode(b_mode_dims[i])

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 2

    #and add the node that forms the root of the bath.  
    #TODO: Add some better functions for handling the construction of tree structures
    topo = ntree("(1(%d(%d))(%d))"%(mode_dims[0], mode_dims[0], mode_dims[0]))
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], b_mode_dims, degree, chi0)
    else:
        ntreeBuilder.mps_subtree(topo()[1], b_mode_dims, chi0, min(chi0, nbose))
    ntreeBuilder.sanitise(topo)

    capacity = ntree("(1(%d(%d))(%d))"%(mode_dims[0], mode_dims[0], mode_dims[0]))
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(capacity()[1], b_mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(capacity()[1], b_mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(capacity)


    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(Nb+1)])
    print(topo)

    h = sop_operator(H, A, sysinf, opdict)

    #set up ttns storing the observable to be measured.  Here these are just system observables
    #so we form a tree with the same topology as A but with all bath bond dimensions set to 1
    obstree = ntree("(1(%d(%d))(1))"%(mode_dims[0], mode_dims[0]))
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(obstree()[1], b_mode_dims, degree, 1)
    else:
        ntreeBuilder.mps_subtree(obstree()[1], b_mode_dims, 1, min(chi0, nbose))
    ntreeBuilder.sanitise(obstree)


    Sz_ttn = ttn(obstree, dtype=np.complex128)
    prod_state = [sz.flatten()]
    for i in range(Nb):
        state_vec = np.zeros(b_mode_dims[i], dtype=np.complex128)
        state_vec[0] = 1.0
        prod_state.append(state_vec)
    Sz_ttn.set_product(prod_state)

    mel = matrix_element(A)

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
    res[0] = np.real(mel(Sz_ttn, A))
    maxchi[0] = A.maximum_bond_dimension()

    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h, dt)
        t2 = time.time()
        res[i+1] = np.real(mel(Sz_ttn, A))
        maxchi[i+1] = A.maximum_bond_dimension()

        print((i+1)*dt, res[i+1], maxchi[i+1], np.real(mel(A, A)))
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
    parser.add_argument('--aaatol', type=float, default=1e-5)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=0.5)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=32)
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
    parser.add_argument('--spawning_threshold', type=float, default=1e-6)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, aaa_tol=args.aaatol, nbose_min=5)
