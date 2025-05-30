# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import argparse
import h5py
from numba import jit

import pyttn
from pyttn import oqs

def output_results(ofname, t, n_u, n_d, n_ud, maxchi):
    h5 = h5py.File(ofname, "w")
    h5.create_dataset("t", data=t)
    h5.create_dataset("n_u", data=n_u)
    h5.create_dataset("n_d", data=n_d)
    h5.create_dataset("n_u n_d", data=n_ud)

    h5.create_dataset("maxchi", data=maxchi)
    h5.close()

def siam_tree(chi, chiU, degree, Nbo, Nbe):
    topo = pyttn.ntree(str("(1(chiU(2(2))(chiU))(chiU(2(2))(chiU)))").replace('chiU', str(chiU)))
    if(degree > 1):
        pyttn.ntreeBuilder.mlmctdh_subtree(topo()[0][1], [2 for i in range(Nbo)], degree, chi)
        pyttn.ntreeBuilder.mlmctdh_subtree(topo()[0][1], [2 for i in range(Nbe)], degree, chi)
        pyttn.ntreeBuilder.mlmctdh_subtree(topo()[1][1], [2 for i in range(Nbo)], degree, chi)
        pyttn.ntreeBuilder.mlmctdh_subtree(topo()[1][1], [2 for i in range(Nbe)], degree, chi)
    else:
        pyttn.ntreeBuilder.mps_subtree(topo()[0][1], [2 for i in range(Nbo)], chi, min(chi, 2))
        pyttn.ntreeBuilder.mps_subtree(topo()[0][1], [2 for i in range(Nbe)], chi, min(chi, 2))
        pyttn.ntreeBuilder.mps_subtree(topo()[1][1], [2 for i in range(Nbo)], chi, min(chi, 2))
        pyttn.ntreeBuilder.mps_subtree(topo()[1][1], [2 for i in range(Nbe)], chi, min(chi, 2))
    pyttn.ntreeBuilder.sanitise(topo)
    return topo

def siam_dynamics(Nb, Gamma, W, epsd, deps, U, chi, dt, chiU = None, beta = None, nstep = 1, geom='star', ofname='sbm.h5', degree = 1, adaptive=True, spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0, init_state = 'up'):
    if chiU is None:
        chiU = chi

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
    Nb = gf.shape[0]+ge.shape[0]

    #set up the total Hamiltonian
    N = Nb+1
    H = pyttn.SOP(2*N)

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
    sysinf = pyttn.system_modes(2*N)
    sysinf.mode_indices = mode_ordering
    for i in range(2*N):
        sysinf[i] = pyttn.fermion_mode()

    #add on the impurity Hamiltonian terms
    H += epsd*pyttn.fOP("n", N-1)
    H += (epsd+deps)*pyttn.fOP("n", N)    #the up impurity site has an energy of epsd + deps 
    H += U*pyttn.fOP("n", N-1)*pyttn.fOP("n", N)

    #add on spin down occupied chain
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, pyttn.fOP("cdag", N-1), pyttn.fOP("c", N-1), gf, wf, geom=geom, binds=modes_f_d)
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, pyttn.fOP("cdag", N-1), pyttn.fOP("c", N-1), ge, we, geom=geom, binds=modes_e_d)
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, pyttn.fOP("cdag", N), pyttn.fOP("c", N), gf, wf, geom=geom, binds=modes_f_u)
    H = oqs.unitary.add_fermionic_bath_hamiltonian(H, pyttn.fOP("cdag", N), pyttn.fOP("c", N), ge, we, geom=geom, binds=modes_e_u)

    #perform a Jordan Wigner mapping of the Fermionic Hamiltonian
    H.jordan_wigner(sysinf)

    A = pyttn.ttn(topo, capacity, dtype=np.complex128)
    state = [0 for i in range(2*N)]
    for i in range(Nbo):
        state[1+i] = 1
        state[N+1+i] = 1
    if init_state == 'up':
        state[N]=1
    else:
        state[N-1] = 1
    A.set_state(state)

    h = pyttn.sop_operator(H, A, sysinf, identity_opt=True, compress=True)

    mel = pyttn.matrix_element(A)

    ops = []
    Nu = None
    if Nu is None:
        op = pyttn.SOP(2*N)
        op += pyttn.fOP("n", N-1)
        ops.append(pyttn.sop_operator(op, A, sysinf))

        op = pyttn.SOP(2*N)
        op += pyttn.fOP("n", N)
        ops.append(pyttn.sop_operator(op, A, sysinf))

        op = pyttn.SOP(2*N)
        op += pyttn.fOP("n", N-1)*pyttn.fOP("n", N)
        ops.append(pyttn.sop_operator(op, A, sysinf))

    sweep = None
    if not adaptive:
        sweep = pyttn.tdvp(A, h, krylov_dim = 12)
    else:
        sweep = pyttn.tdvp(A, h, krylov_dim = 12, subspace_krylov_dim=10, subspace_neigs=2, expansion='subspace')
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


    t = np.arange(nstep+1)*dt

    for res, op in zip(results, ops):
        res[0] = np.real(mel(op, A, A))
    maxchi[0] = A.maximum_bond_dimension()
    for i in range(nstep):
        sweep.step(A, h)
        renorm = mel(A, A)
        for res, op in zip(results, ops):
            res[i+1] = np.real(mel(op, A, A)/renorm)
        maxchi[i+1] = A.maximum_bond_dimension()

        if(i % 10):
            output_results(ofname, t, *results, maxchi)

    output_results(ofname, t, *results, maxchi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the single impurity anderson model.')

    #exponential bath cutoff parameters
    parser.add_argument('--Gamma', type = float, default=0.01)
    parser.add_argument('--W', type = float, default=1)
    parser.add_argument('--epsd', type = float, default=-0.1875)
    parser.add_argument('--deps', type = float, default=0.0)
    parser.add_argument('--U', type = float, default=15)

    #number of bath modes
    parser.add_argument('--N', type=int, default=64)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=10)

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
