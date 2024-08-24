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


def siam_tree(chi, chiU, degree, Nbo, Nbe):
    topo = ntree(str("(1(chiU(2(2))(chiU))(chiU(2(2))(chiU)))").replace('chiU', str(chiU)))
    print(topo)
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[0][1], [2 for i in range(Nbo)], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[0][1], [2 for i in range(Nbe)], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[1][1], [2 for i in range(Nbo)], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[1][1], [2 for i in range(Nbe)], degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[0][1], [2 for i in range(Nbo)], chi, min(chi, 2))
        ntreeBuilder.mps_subtree(topo()[0][1], [2 for i in range(Nbe)], chi, min(chi, 2))
        ntreeBuilder.mps_subtree(topo()[1][1], [2 for i in range(Nbo)], chi, min(chi, 2))
        ntreeBuilder.mps_subtree(topo()[1][1], [2 for i in range(Nbe)], chi, min(chi, 2))
    ntreeBuilder.sanitise(topo)
    return topo

def Ct(t, w, g):
    T, W = np.meshgrid(t, w)
    g2 = g**2
    fourier = np.exp(-1.0j*W*T)
    return g2@fourier

def siam_dynamics(Nb, Gamma, W, epsd, deps, U, chi, dt, chiU = None, beta = None, Ncut = 20, nstep = 1, Nw = 7.5, geom='star', ofname='sbm.h5', degree = 1, adaptive=True, spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0, init_state = 'up'):
    if chiU is None:
        chiU = chi

    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def V(w):
        return np.where(np.abs(w) <= W, Gamma, 0.0)

    #set up the open quantum system bath object
    bath = oqs.fermionic_bath(V, beta=beta)


    gf,wf, ge, we = bath.discretise(Nb, W, method='orthopol')
    g = np.sqrt(np.pi)*np.concatenate((gf, ge))
    w = np.concatenate((wf, we))

    Nbo = gf.shape[0]
    Nbe = ge.shape[0]
    Nb = gf.shape[0]+ge.shape[0]

    #set up the total Hamiltonian
    N = Nb+1
    H = SOP(2*N)

    modes_f_d = [N-1 - (x+1) for x in range(Nbo)]
    modes_f_u = [N+1 + x for x in range(Nbo)]

    modes_e_d = [N-1-Nbo - (x+1) for x in range(Nbe)]
    modes_e_u = [N+1+Nbo + x for x in range(Nbe)]

    #set up the topology tree - this structure would ensure that the mode ordering in the Hamiltonian would be {c_d f^o_1d, f^o_2d, \dots, f^o_{Nd},  f^e_1d, f^e_2d, \dots, f^e_{Nd}, c_u, f^o_1u, f^o_2u, \dots, f^o_{Nu},  f^e_1u, f^e_2u, \dots, f^e_{Nu}}
    topo = siam_tree(8, 8, degree, Nbo, Nbe)
    capacity = siam_tree(chi, chiU, degree, Nbo, Nbe)

    #In order to avoid long range JW strings we want the hamiltonian ordering to be  {f^e_Nd, \dots, f^e_{1d}, f^o_Nd, \dots, f^o_{1d}, c_d, c_u, f^o_1u, \dots, f^e_{Nu}, f^e_1u, \dots, f^e_{Nu}}.  
    #As such we will set up the optional mode ordering object allowed within the system_info class.  This allows us to specify a different ordering of modes
    #for the system information compared to tree structure - setup the mode ordering so that the down impurity ordering is flipped while the up ordering 
    #is the current order
    mode_ordering = [N - (x+1) for x in range(N)] + [N + x for x in range(N)]
    sysinf = system_modes([fermion_mode() for x in range(2*N)], mode_ordering)
    print(sysinf.mode_indices)

    #and discretise the bath getting the star Hamiltonian parameters using the orthpol discretisation strategy

    #add on the impurity Hamiltonian terms
    H += epsd*fOP("n", N-1)
    H += (epsd+deps)*fOP("n", N)    #the up impurity site has an energy of epsd + deps 
    H += U*fOP("n", N-1)*fOP("n", N)

    #add on spin down occupied chain
    H = oqs.add_fermionic_bath_hamiltonian(H, fOP("cdag", N-1), fOP("c", N-1), gf, wf, geom=geom, binds=modes_f_d)
    H = oqs.add_fermionic_bath_hamiltonian(H, fOP("cdag", N-1), fOP("c", N-1), ge, we, geom=geom, binds=modes_e_d)
    H = oqs.add_fermionic_bath_hamiltonian(H, fOP("cdag", N), fOP("c", N), gf, wf, geom=geom, binds=modes_f_u)
    H = oqs.add_fermionic_bath_hamiltonian(H, fOP("cdag", N), fOP("c", N), ge, we, geom=geom, binds=modes_e_u)

    print(H)
    H.jordan_wigner(sysinf)
    print(H)

    A = ttn(topo, capacity, dtype=np.complex128)
    state = [0 for i in range(2*N)]
    for i in range(Nbo):
        state[1+i] = 1
        state[N+1+i] = 1
    if init_state = 'up':
        state[N]=1
    else:
        state[N-1] = 1
    print(state)
    A.set_state(state)

    h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)

    mel = matrix_element(A)

    ops = []
    Nu = None
    if Nu is None:
        op = SOP(2*N)
        op += fOP("n", N-1)
        ops.append(sop_operator(op, A, sysinf))

        op = SOP(2*N)
        op += fOP("n", N)
        ops.append(sop_operator(op, A, sysinf))

        op = SOP(2*N)
        op += fOP("n", N-1)*fOP("n", N)
        ops.append(sop_operator(op, A, sysinf))
    labels = ["n_u", "n_d", "n_u n_d"]

    sweep = None
    if not adaptive:
        sweep = tdvp(A, h, krylov_dim = 12)
    else:
        sweep = tdvp(A, h, krylov_dim = 12, subspace_krylov_dim=10, subspace_neigs=2, expansion='subspace')
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold=unoccupied_threshold
        sweep.minimum_unoccupied=nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j

    if(geom == 'ipchain'):
        sweep.use_time_dependent_hamiltonian = True

    maxchi = np.zeros(nstep+1)
    results = []
    for i in range(len(ops)):
        results.append(np.zeros(nstep+1))


    for res, op in zip(results, ops):
        res[0] = np.real(mel(op, A, A))
    maxchi[0] = A.maximum_bond_dimension()
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        renorm = mel(A, A)
        for res, op in zip(results, ops):
            res[i+1] = np.real(mel(op, A, A)/renorm)
        maxchi[i+1] = A.maximum_bond_dimension()

        if(i % 100):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            for label, res in zip(labels, results):
                h5.create_dataset(label, data=res)
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()

    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    for label, res in zip(labels, results):
        h5.create_dataset(label, data=res)
    h5.create_dataset('maxchi', data=maxchi)
    h5.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the single impurity anderson model.')

    #exponential bath cutoff parameters
    parser.add_argument('--Gamma', type = float, default=1)
    parser.add_argument('--W', type = float, default=1)
    parser.add_argument('--epsd', type = float, default=-0.1875)
    parser.add_argument('--deps', type = float, default=0.0)
    parser.add_argument('--U', type = float, default=15)

    #number of bath modes
    parser.add_argument('--N', type=int, default=64)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=128)
    parser.add_argument('--chiU', type=int, default=36)
    parser.add_argument('--degree', type=int, default=1)


    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--tmax', type=float, default=100)

    #output file name
    parser.add_argument('--fname', type=str, default='siam.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-6)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    parser.add_argument('--initial_state', type=str, default='up')

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    siam_dynamics(args.N, args.Gamma, args.W, args.epsd, args.deps, args.U, args.chi, args.dt, chiU=args.chiU, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, init_state = args.initial_state)
