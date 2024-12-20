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


def dqd_tree(chi, chiU, chiD, degree, Nbo, Nbe):
    topo = ntree(str("(1(chiD(chiU(2(2))(chiU))(chiU(2(2))(chiU)))(chiD(chiU(2(2))(chiU))(chiU(2(2))(chiU))))").replace('chiU', str(chiU)))
    print(topo)
    if(degree > 1):
        #left dot
        ntreeBuilder.mlmctdh_subtree(topo()[0][0][1], [2 for i in range(Nbo)], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[0][0][1], [2 for i in range(Nbe)], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[0][1][1], [2 for i in range(Nbo)], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[0][1][1], [2 for i in range(Nbe)], degree, chi)
        #right dot
        ntreeBuilder.mlmctdh_subtree(topo()[1][0][1], [2 for i in range(Nbo)], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[1][0][1], [2 for i in range(Nbe)], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[1][1][1], [2 for i in range(Nbo)], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[1][1][1], [2 for i in range(Nbe)], degree, chi)
    else:
        #left dot
        ntreeBuilder.mps_subtree(topo()[0][0][1], [2 for i in range(Nbo)], chi, min(chi, 2))
        ntreeBuilder.mps_subtree(topo()[0][0][1], [2 for i in range(Nbe)], chi, min(chi, 2))
        ntreeBuilder.mps_subtree(topo()[0][1][1], [2 for i in range(Nbo)], chi, min(chi, 2))
        ntreeBuilder.mps_subtree(topo()[0][1][1], [2 for i in range(Nbe)], chi, min(chi, 2))
        #right dot
        ntreeBuilder.mps_subtree(topo()[1][0][1], [2 for i in range(Nbo)], chi, min(chi, 2))
        ntreeBuilder.mps_subtree(topo()[1][0][1], [2 for i in range(Nbe)], chi, min(chi, 2))
        ntreeBuilder.mps_subtree(topo()[1][1][1], [2 for i in range(Nbo)], chi, min(chi, 2))
        ntreeBuilder.mps_subtree(topo()[1][1][1], [2 for i in range(Nbe)], chi, min(chi, 2))
    ntreeBuilder.sanitise(topo)
    return topo


def setup_hamiltonian(epsd, U, Uc, Tc, gf, wf, ge, we):
    Nbo = gf.shape[0]
    Nbe = ge.shape[0]
    Nb = gf.shape[0]+ge.shape[0]
    N = Nb+1

    modes_f_d = [N-1 - (x+1) for x in range(Nbo)]
    modes_f_u = [N+1 + x for x in range(Nbo)]

    modes_e_d = [N-1-Nbo - (x+1) for x in range(Nbe)]
    modes_e_u = [N+1+Nbo + x for x in range(Nbe)]

    H = SOP(4*N)
    #add on the impurity Hamiltonian terms for left dot
    H += epsd*fOP("n", N-1)
    H += epsd*fOP("n", N)    #the up impurity site has an energy of epsd + deps 
    H += U*fOP("n", N-1)*fOP("n", N)

    #add on the impurity Hamiltonian terms for right dot
    H += epsd*fOP("n", N-1+2*N)
    H += epsd*fOP("n", N+2*N)    #the up impurity site has an energy of epsd + deps 
    H += U*fOP("n", N-1+2*N)*fOP("n", N+2*N)

    #now add on the interaction terms between the two dots - density density interaction
    H += Uc*fOP("n", N-1)*fOP("n", 2*N+N-1)
    H += Uc*fOP("n", N)*fOP("n", 2*N+N-1)
    H += Uc*fOP("n", N-1)*fOP("n", 2*N+N)
    H += Uc*fOP("n", N)*fOP("n", 2*N+N)

    #hopping term
    H += Tc*fOP("adag", N-1)*fOP("a", 2*N+N-1)
    H += Tc*fOP("adag", 2*N+N-1)*fOP("a", N-1)
    H += Tc*fOP("adag", N)*fOP("a", 2*N+N)
    H += Tc*fOP("adag", 2*N+N)*fOP("a", N)

    #add on spin down occupied chain for left dot
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, fOP("cdag", N-1), fOP("c", N-1), gf, wf, geom=geom, binds=modes_f_d)
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, fOP("cdag", N-1), fOP("c", N-1), ge, we, geom=geom, binds=modes_e_d)
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, fOP("cdag", N), fOP("c", N), gf, wf, geom=geom, binds=modes_f_u)
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, fOP("cdag", N), fOP("c", N), ge, we, geom=geom, binds=modes_e_u)

    #add on spin down occupied chain for right dot
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, fOP("cdag",2*N+N-1), fOP("c", 2*N+N-1), gf, wf, geom=geom, binds=[2*N+i for i in modes_f_d])
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, fOP("cdag",2*N+N-1), fOP("c", 2*N+N-1), ge, we, geom=geom, binds=[2*N+i for i in modes_e_d])
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, fOP("cdag",2*N+N), fOP("c", N), 2*N+gf, wf, geom=geom, binds=[2*N+i for i in modes_f_u])
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, fOP("cdag",2*N+N), fOP("c", N), 2*N+ge, we, geom=geom, binds=[2*N+i for i in modes_e_u])

    return H

def dqd_dynamics(Nb, Gamma, W, epsd, U, Tc, Uc, chi, dt, chiU = None, chiD=None, beta = None, nstep = 1, geom='chain', ofname='dqd.h5', degree = 1, adaptive=True, spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0):
    if chiU is None:
        chiU = chi

    if chiD is None:
        chiD = chi

    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def V(w):
        return np.where(np.abs(w) <= W, Gamma*np.sqrt(1-(w*w)/(W*W)), 0.0)

    #set up the open quantum system bath object
    bath = oqs.FermionicBath(V, beta=beta)
   
    gf, wf = bath.discretise(oqs.OrthopolDiscretisation(Nb, *bath.estimate_bounds(wmax=W)), Ef = 0.0, sigma='+')
    ge, we = bath.discretise(oqs.OrthopolDiscretisation(Nb, *bath.estimate_bounds(wmax=W)), Ef = 0.0, sigma='-')

    Nbo = gf.shape[0]
    Nbe = ge.shape[0]

    #set up the total Hamiltonian
    N = Nb+1

    chi0 = chi
    chiU0 = chiU
    chiD0 = chiD
    if adaptive:
        chi0 = 8
        chiU0 = 8
        chiD0=8

    chi0 = min(chi0, chi)
    chiU0 = min(chiU0, chiU)
    chiD0 = min(chiD0, chiD)

    #set up the topology tree - this structure would ensure that the mode ordering in the Hamiltonian would be {c_d f^o_1d, f^o_2d, \dots, f^o_{Nd},  f^e_1d, f^e_2d, \dots, f^e_{Nd}, c_u, f^o_1u, f^o_2u, \dots, f^o_{Nu},  f^e_1u, f^e_2u, \dots, f^e_{Nu}}
    topo = dqd_tree(chi0, chiU0, chiD0, degree, Nbo, Nbe)
    capacity = dqd_tree(chi, chiU, chiD, degree, Nbo, Nbe)

    #In order to avoid long range JW strings we want the hamiltonian ordering to be  {f^e_Nd, \dots, f^e_{1d}, f^o_Nd, \dots, f^o_{1d}, c_d, c_u, f^o_1u, \dots, f^e_{Nu}, f^e_1u, \dots, f^e_{Nu}}.  
    #As such we will set up the optional mode ordering object allowed within the system_info class.  This allows us to specify a different ordering of modes
    #for the system information compared to tree structure - setup the mode ordering so that the down impurity ordering is flipped while the up ordering 
    #is the current order
    mode_ordering = [N - (x+1) for x in range(N)] + [N + x for x in range(N)] + [2*N+N - (x+1) for x in range(N)] + [2*N+N + x for x in range(N)]
    sysinf = system_modes(4*N)
    sysinf.mode_indices = mode_ordering
    for i in range(4*N):
        sysinf[i] = fermion_mode()

    H = setup_hamiltonian(epsd, U, Uc, Tc, gf, wf, ge, we)

    print(H)
    H.jordan_wigner(sysinf)
    print(H)

    A = ttn(topo, capacity, dtype=np.complex128)
    state = [0 for i in range(4*N)]

    #fill the filled orbitals
    for i in range(Nbo):
        state[1+i] = 1
        state[N+1+i] = 1
        state[2*N+1+i] = 1
        state[2*N+N+1+i] = 1
    state[N-1]=1
    state[2*N+N] = 1
    A.set_state(state)

    h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)
    mel = matrix_element(A)

    #compute the ground state of the system

    #now perform the dynamics and extract the Green's function


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
    parser.add_argument('--U', type = float, default=15)
    parser.add_argument('--Uc', type = float, default=10)
    parser.add_argument('--Tc', type = float, default=0)

    #number of bath modes
    parser.add_argument('--N', type=int, default=64)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='chain')

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=128)
    parser.add_argument('--chiU', type=int, default=36)
    parser.add_argument('--chiD', type=int, default=36)
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


    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    dqd_dynamics(args.N, args.Gamma, args.W, args.epsd, args.U, args.Tc, args.Uc, args.chi, args.dt, chiU=args.chiU, chiD=args.chiD, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree)
