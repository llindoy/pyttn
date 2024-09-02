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


def siam_tree(chiU, chi, degree_impurity, degree, Nbo, Nbe):
    mdo = [2 for i in range(Nbo)]
    mde = [2 for i in range(Nbe)]
    topo = None
    if(degree_impurity > 1):
        topo = ntreeBuilder.mlmctdh_tree([chiU for i in range(4)], degree_impurity, chiU)
    else:
        topo = ntreeBuilder.mps_tree([chiU for i in range(4)], chiU, chiU)

    leaf_indices = topo.leaf_indices()
    for li in leaf_indices:
        topo.at(li).insert(2)
        topo.at(li)[0].insert(2)
        topo.at(li).insert(chiU)
        topo.at(li).insert(chiU)
        if(degree > 1):
            ntreeBuilder.mlmctdh_subtree(topo.at(li)[1], mdo, degree, chi)
            ntreeBuilder.mlmctdh_subtree(topo.at(li)[2], mde, degree, chi)
        else:
            ntreeBuilder.mps_subtree(topo.at(li)[1], md, chi, min(chi, 2))
            ntreeBuilder.mps_subtree(topo.at(li)[2], md, chi, min(chi, 2))

    ntreeBuilder.sanitise(topo)
    return topo

#arxiv 2305.17686
def dqd_dynamics(Nb, Gamma, W, epsd, U, chi, dt, chiU = None, beta = None, nstep = 1, ofname='siam.h5', degree = 1, adaptive=True, spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0):
    if chiU is None:
        chiU = chi

    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def V(w):
        return np.where(np.abs(w) <= W, Gamma * np.sqrt(1-(w/W)**2), 0.0)

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
    topo = siam_tree(4, 4, 2, degree, Nbo, Nbe)
    capacity = siam_tree(chi, chiU, 2, degree, Nbo, Nbe)

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
    H += epsd*fOP("n", N)
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
    parser.add_argument('--W', type = float, default=10)
    parser.add_argument('--epsd', type = float, default=-1.25)
    parser.add_argument('--U', type = float, default=2.5)

    #number of bath modes
    parser.add_argument('--N', type=int, default=64)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=156)
    parser.add_argument('--chiU', type=int, default=48)
    parser.add_argument('--degree', type=int, default=2)


    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--tmax', type=float, default=5)

    #output file name
    parser.add_argument('--fname', type=str, default='siam.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-5)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    siam_dynamics(args.N, args.Gamma, args.W, args.epsd*np.pi*args.Gamma, args.U*np.pi*args.Gamma, args.chi, args.dt, chiU=args.chiU, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree)
