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


def build_topology(nsys, chi, nbose, b_mode_dims, degree):
    topo = ntree("(1(%d(%d))(%d))"%(nsys, nsys, nsys))
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], b_mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[1], b_mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(topo)
    return topo

def sbm_dynamics(alpha, wc, s, eps, delta, chi, nbose, dt, nbose_min = None, beta = None, nstep = 1, Nw = 10.0, ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, aaa_tol=1e-4, use_mode_combination=True):
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)
    dk, zk = bath.expfit(oqs.AAADecomposition(tol=1e-4, wmin = -2*Nw*wc, wmax=2*Nw*wc))

    Nb = 2*dk.shape[0]
    N = Nb+1

    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    mode_dims = oqs.heom.compute_dimensions(sz, dk, zk, nbose, Lmin=nbose_min)
    b_mode_dims = None

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
            gk = dk[i//2]/np.sqrt(dk[i//2])
            hk = np.sqrt(dk[i//2])
        else:
            wk = -1.0j*np.conj(zk[i//2])
            gk = -np.conj(dk[i//2])/np.sqrt(np.conj(dk[i//2]))
            hk = np.sqrt(np.conj(dk[i//2]))

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
    sysinf = None
    Nbc  = None
    if not use_mode_combination:
        b_mode_dims = mode_dims[1:]
        Nbc = Nb
        sysinf = system_modes(N)
        sysinf[0] = generic_mode(mode_dims[0])
        for i in range(Nb):
            sysinf[i+1] = generic_mode(b_mode_dims[i])
    else:
        Nbc = dk.shape[0]
        b_mode_dims = np.zeros(Nbc, dtype=int)
        for i in range(Nbc):
            b_mode_dims[i] = mode_dims[2*i+1]*mode_dims[2*i+2]

        sysinf = system_modes(Nbc+1)
        sysinf[0] = generic_mode(mode_dims[0])
        for i in range(Nbc):
            sysinf[i+1] = [generic_mode(mode_dims[2*i+1]), generic_mode(mode_dims[2*i+2])]

    def bops(Nb):
        d = np.sqrt(np.arange(Nb-1)+1.0)
        c = np.arange(Nb-1, dtype=int)
        r = c+1
        return scipy.sparse.coo_matrix((d, (c, r)), shape=(Nb, Nb)).tocsr(), scipy.sparse.coo_matrix((d, (r, c)), shape=(Nb, Nb)).tocsr()

    for i in range(Nb):
        a, adag = bops(mode_dims[i+1])
        print(np.arange(mode_dims[i+1]))
        opdict.insert(i+1, "n", site_operator(np.arange(mode_dims[i+1], dtype=np.complex128), optype="diagonal_matrix", mode=i+1))
        opdict.insert(i+1, "adag", site_operator(adag, optype="sparse_matrix", mode=i+1))
        opdict.insert(i+1, "a", site_operator(a, optype="sparse_matrix", mode=i+1))

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 4

    topo = build_topology(mode_dims[0], chi0, nbose, b_mode_dims, degree)
    capacity = build_topology(mode_dims[0], chi, nbose, b_mode_dims, degree)

    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(Nbc+1)])
    print(topo)

    h = sop_operator(H, A, sysinf, opdict)
    #set up ttns storing the observable to be measured.  Here these are just system observables
    #so we form a tree with the same topology as A but with all bath bond dimensions set to 1
    obstree = build_topology(mode_dims[0], 1, nbose, b_mode_dims, degree)

    Sz_ttn = ttn(obstree, dtype=np.complex128)
    #setup the Sz tree state
    prod_state = [sz.flatten()]
    for i in range(Nbc):
        state_vec = np.zeros(b_mode_dims[i], dtype=np.complex128)
        state_vec[0] = 1.0
        prod_state.append(state_vec)
    Sz_ttn.set_product(prod_state)

    #set up the identity tree state
    id_ttn = ttn(obstree, dtype=np.complex128)
    prod_state = [np.identity(2).flatten()]
    for i in range(Nbc):
        state_vec = np.zeros(b_mode_dims[i], dtype=np.complex128)
        state_vec[0] = 1.0
        prod_state.append(state_vec)
    id_ttn.set_product(prod_state)
    mel = matrix_element(A)

    #set up the tdvp object
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

    res = np.zeros(nstep+1)
    maxchi = np.zeros(nstep+1)
    res[0] = np.real(mel(Sz_ttn, A))
    maxchi[0] = A.maximum_bond_dimension()

    #perform the dynamics
    renorm = mel(id_ttn, A)
    i=0
    print((i)*dt, res[i], np.real(renorm), maxchi[i], np.real(mel(A, A)))
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        renorm = mel(id_ttn, A)
        #A*=(1/renorm)
        res[i+1] = np.real(mel(Sz_ttn, A))
        maxchi[i+1] = A.maximum_bond_dimension()

        print((i+1)*dt, res[i+1], np.real(renorm), maxchi[i+1], np.real(mel(A, A)))
        sys.stdout.flush()
        if(i % 10 == 0):
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
    parser.add_argument('--aaatol', type=float, default=1e-7)


    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=0.5)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=64)
    parser.add_argument('--degree', type=int, default=1)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=35)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--tmax', type=float, default=40)

    #output file name
    parser.add_argument('--fname', type=str, default='sbm.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-7)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, aaa_tol=args.aaatol, nbose_min=4, use_mode_combination=True)
