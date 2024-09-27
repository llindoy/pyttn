import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
import pyttn
from pyttn import *
from pyttn import oqs
from pyttn._pyttn import operator_dictionary_complex
from pyttn._pyttn.linalg import csr_matrix_complex, diagonal_matrix_complex

from numba import jit


def sbm_dynamics(alpha, wc, s, eps, delta, chi, nbose, dt, nbose_min = None, beta = None, Ncut = 20, nstep = 1, Nw = 7.5, geom='star', ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, aaa_tol=1e-4):

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

    Nb = dk.shape[0]

    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    mode_dims = oqs.heom.compute_dimensions(sz, dk, zk, nbose, Lmin=nbose_min)
    b_mode_dims = np.zeros(Nb, dtype=int)
    for i in range(Nb):
        b_mode_dims[i] = mode_dims[2*i+1]*mode_dims[2*i+2]

    #set up the total Hamiltonian
    H = SOP(2*Nb+1)

    gk = np.real(zk)
    Ek = np.imag(zk)

    Vk = np.real(np.sqrt(dk))
    Mk = -np.imag(np.sqrt(dk))

    #and add on the system parts
    Hsys = eps*sz + delta *sx
    H += sOP("lsys", 0)

    for i in range(Nb):
        i1 = 2*i+1
        i2 = 2*i+2

        H += Ek[i]*sOP("n", i1)-Ek[i]*sOP("n", i2)
        H += 2.0j*gk[i]*sOP("a", i1)*sOP("a", i2) - 1.0j*gk[i]*sOP("n", i1) - 1.0j*gk[i]* sOP("n", i2)
        H += Vk[i]*sOP("sl", 0) * sOP("adag", i1)+Vk[i]*sOP("sl", 0) *sOP("a", i1)  - Vk[i]*sOP("sr", 0) * sOP("adag", i2)- Vk[i]*sOP("sr", 0) *sOP("a", i2)
        H += 2.0j*Mk[i]*sOP("sr", 0)*sOP("a", i1) - 1.0j*Mk[i]* sOP("sl", 0)*sOP("a", i1) -1.0j*Mk[i]* sOP("sr", 0)*sOP("adag", i2)
        H += 2.0j*Mk[i]*sOP("sl", 0)*sOP("a", i2) - 1.0j*Mk[i] * sOP("sl", 0)*sOP("adag", i1) - 1.0j*Mk[i] * sOP("sr", 0)*sOP("a", i2)

    #set up the operator dictionary information
    opdict = operator_dictionary_complex(2*Nb+1)

    #set up the different system operators
    opdict.insert(0, "lsys", site_operator(oqs.heom.commutator(Hsys), optype="matrix", mode=0))
    opdict.insert(0, "sl", site_operator(oqs.heom.Sl(sz), optype="matrix", mode=0))
    opdict.insert(0, "sr", site_operator(oqs.heom.Sr(sz), optype="matrix", mode=0))

    #setup the system information object
    sysinf = system_modes(Nb+1)
    sysinf[0] = generic_mode(mode_dims[0])
    for i in range(Nb):
        sysinf[i+1] = [boson_mode(mode_dims[2*i+1]), boson_mode(mode_dims[2*i+2])]

    print(sysinf)

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 4

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

    for n in topo:
        print(n.value)

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
        state_vec = np.identity(mode_dims[2*i+1], dtype=np.complex128).flatten()
        prod_state.append(state_vec)
    Sz_ttn.set_product(prod_state)

    id_ttn = ttn(obstree, dtype=np.complex128)
    prod_state = [np.identity(2).flatten()]
    for i in range(Nb):
        state_vec = np.identity(mode_dims[2*i+1], dtype=np.complex128).flatten()
        state_vec[0] = 1.0
        prod_state.append(state_vec)
    id_ttn.set_product(prod_state)
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
        if(i % 100):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            h5.create_dataset('Sz', data=res)
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

    exit()
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
    parser.add_argument('--aaatol', type=float, default=1e-6)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=0.5)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=64)
    parser.add_argument('--degree', type=int, default=1)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=25)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.0025)
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
    sbm_dynamics(args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, aaa_tol=args.aaatol, nbose_min=8)
