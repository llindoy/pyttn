import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import h5py
import scipy

sys.path.append("../../")
import pyttn
from pyttn import *
from pyttn import oqs, utils

from numba import jit


def siam_tree(chi, chiU, degree, bsysf, bsyse):
    topo = ntree(str("(1(chiU(4(4))(chiU))(chiU(4(4))(chiU)))").replace('chiU', str(chiU)))
    print(topo)
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[0][1], [bsysf[i] for i in range(len(bsysf))], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[0][1], [bsyse[i] for i in range(len(bsyse))], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[1][1], [bsysf[i] for i in range(len(bsysf))], degree, chi)
        ntreeBuilder.mlmctdh_subtree(topo()[1][1], [bsyse[i] for i in range(len(bsyse))], degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[0][1], [bsysf[i] for i in range(len(bsysf))], chi, min(chi, 4))
        ntreeBuilder.mps_subtree(topo()[0][1], [bsyse[i] for i in range(len(bsyse))], chi, min(chi, 4))
        ntreeBuilder.mps_subtree(topo()[1][1], [bsysf[i] for i in range(len(bsysf))], chi, min(chi, 4))
        ntreeBuilder.mps_subtree(topo()[1][1], [bsyse[i] for i in range(len(bsyse))], chi, min(chi, 4))
    ntreeBuilder.sanitise(topo)
    return topo

def observable_tree(obstree, dims):
    Opttn = ttn(obstree, dtype=np.complex128)
    prod_state = [op.flatten()]
    for i in range(len(dims)):
        state_vec = np.identity(int(np.sqrt(dims[i])), dtype=np.complex128).flatten()
        prod_state.append(state_vec)
    Opttn.set_product(prod_state)
    return Opttn


def bath_parameters(bath, K, nstep, dt, Ef, sigma):
    dkf, zkf = bath.expfit(oqs.ESPRITDecomposition(K=K, tmax=nstep*dt, Nt = nstep), Ef = 0.0, sigma=sigma)

    #set up the exp bath object this takes the dk and zk terms.  Truncate the modes and
    #extract the system information object from this.
    expbathf = oqs.ExpFitFermionicBath(dkf, zkf)
    bsysf = expbathf.system_information()


    import matplotlib.pyplot as plt

    t=np.linspace(0, dt*(nstep+1), nstep+1)

    plt.plot(t, oqs.FermionicBath.Ctexp(t, dkf, -1.0j*zkf, sigma=sigma))

    gkf = np.real(zkf)
    Ekf = np.imag(zkf)

    Vkf = np.real(np.sqrt(dkf))
    Mkf = -np.imag(np.sqrt(dkf))
    return gkf, Ekf, Vkf, Mkf, bsysf


def setup_system(bsysf, bsyse, use_mode_combination=True, nbmax=8, nhilbmax=1024):
    #setup the system information object
    sysinf = system_modes(1)
    sysinf[0] = [fermion_mode(), fermion_mode()]

    if use_mode_combination:
        mode_comb = utils.ModeCombination(nhilbmax, nbmax)
        bsyse = mode_comb(bsyse)
        bsysf = mode_comb(bsysf)

    Nf = bsysf.nprimitive_modes()
    Ne = bsyse.nprimitive_modes()
    N = 2+Ne+Nf

    bsys = combine_system(bsysf, bsyse)

    #set the spin up spins
    sysinfu = combine_systems(sysinf, bsys)

    #and the down spin results.  Now order the modes the otherway around for the jordan-wigner mapping
    sysinfd = system_modes(len(sysinfu))
    for i in range(sysinfu):
        sysinfd[i] = sysinfu[len(sysinfu)-(i+1)]


    mode_ordering = [len(sysinfd) - (x+1) for x in range(len(sysinfd))] 
    sysinfd.mode_indices=mode_ordering

    #but reverese the ordering on the tree structure
    dims=[sysinfu[i].lhd() for i in range(len(sysinfu))]+[sysinfu[i].lhd() for i in range(len(sysinfu))]

    sysinf = combine_systems(sysinfd, sysinfu)

    modes_f_d = [N - 2 - (x+1) for x in range(Nf)]
    modes_f_u = [N + 2 - (x+1) for x in range(Nf)]

    modes_e_d = [N - 2 - Nf - (x+1) for x in range(Ne)]
    modes_e_u = [N + 2 - Nf - (x+1) for x in range(Ne)]

    return sysinf, dims, bsysf, bsyse, modes_f_d, modes_f_u, modes_e_d, modes_e_u




def siam_dynamics(Gamma, W, epsd, deps, U, chi, K, dt, chiU = None, beta = None, nstep = 1, ofname='sbm.h5', degree = 1, adaptive=True, spawning_threshold=1e-5, unoccupied_threshold=1e-4, nunoccupied=0, init_state = 'up', use_mode_combination=True, nbmax=8, nhilbmax=1024):
    if chiU is None:
        chiU = chi

    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def V(w):
        return np.where(np.abs(w) <= W, Gamma*np.sqrt(1-(w*w)/(W*W)), 0.0)

    #set up the open quantum system bath object
    bath = oqs.FermionicBath(V, beta=beta)
   
    gkf, Ekf, Vkf, Mkf, bsysf = bath_parameters(bath, K, nstep, dt, 0.0, '+')
    gke, Eke, Vke, Mke, bsyse = bath_parameters(bath, K, nstep, dt, 0.0, '-')
    plt.show()

    sysinf, dims, bsysf, bsyse, modes_f_d, modes_f_u, modes_e_d, modes_e_u = setup_system(bsysf, bsyse, use_mode_combination, nbmax, nhilbmax)

    N = sysinf.nprimitive_modes()
    Nh = N//2

    #set up the total Hamiltonian
    H = SOP(N)

    #add on the system liouvillian - here we are using that sz^T = sz and "sx^T=sx"
    Lsys = epsd* ( fOP("n", Nn-2) - fOP("n", Nn-1))
    Lsys += (epsd+deps)* ( fOP("n", Nn) - fOP("n", Nn+1))
    Lsys += U* ( fOP("n", Nn-2)*fOP("n", Nn) - fOP("n", Nn-1)*fOP("n", Nn+1))

    H += Lsys

    #now add on the filled bath terms

    #and the empty bath terms

    for i in range(len(zk)):
        i1 = 2*(i+1)
        i2 = 2*(i+1)+1

        H += 2.0j*complex(gk[i])*sOP("a", i1)*sOP("a", i2)
        H += -1.0j*(complex(zk[i])*sOP("n", i1)  + complex(np.conj(zk[i]))*sOP("n", i2))
        H += 2.0j*(complex(np.conj(Mk[i]))*sOP("sz", 0)*sOP("a", i2) + complex(Mk[i])*sOP("sz", 1)*sOP("a", i1))
        H += (Vk[i]-1.0j*Mk[i])*sOP("sz", 0)*sOP("adag", i1) - (Vk[i]+1.0j*Mk[i])*sOP("sz", 1)*sOP("adag", i2)
        H += (Vk[i]-1.0j*Mk[i])*sOP("sz", 0)*sOP("a", i1) - (Vk[i]+1.0j*Mk[i])*sOP("sz", 1)*sOP("a", i2)

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 4

    topo = build_topology(sysinf[0].lhd(), chi0, L, b_mode_dims, degree)
    capacity = build_topology(sysinf[0].lhd(), chi, L, b_mode_dims, degree)

    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(len(bsys)+1)])

    h = sop_operator(H, A, sysinf)
    #set up ttns storing the observable to be measured.  Here these are just system observables
    #so we form a tree with the same topology as A but with all bath bond dimensions set to 1
    obstree = siam_tree(1,1, degree, bsysf, bsyse)
    id_ttn = observable_tree(obstree, dims)

    mel = matrix_element(A)

    sweep = None
    if not adaptive:
        sweep = tdvp(A, h, krylov_dim = 12)
    else:
        sweep = tdvp(A, h, krylov_dim = 12, expansion='subspace')
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold=unoccupied_threshold
        sweep.minimum_unoccupied=nunoccupied

    sweep.coefficient = -1.0j

    res = np.zeros(nstep+1)
    maxchi = np.zeros(nstep+1)
    res[0] = np.real(mel(Sz_ttn, A))
    maxchi[0] = A.maximum_bond_dimension()

    renorm = mel(id_ttn, A)
    i=0
    print((i)*dt, res[i], np.real(renorm), maxchi[i], np.real(mel(A, A)))

    tp = 0
    ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), 5)
    for i in range(5):
        dti = ts[i]-tp
        print(ts, dt)
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]

    i=1
    res[1] = np.real(mel(Sz_ttn, A))
    maxchi[1] = A.maximum_bond_dimension()
    print((i)*dt, res[i], np.real(renorm), maxchi[i], np.real(mel(A, A)))
    sweep.dt = dt

    for i in range(1,nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        renorm = mel(id_ttn, A)
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
    parser.add_argument('--K', type=int, default=4)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--Lmin', type=int, default=4)

    #mode combination parameters
    parser.add_argument('--nbmax', type=int, default=1)
    parser.add_argument('--nhilbmax', type=int, default=1000)


    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=0.5)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=64)
    parser.add_argument('--degree', type=int, default=1)

    

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--tmax', type=float, default=10)

    #output file name
    parser.add_argument('--fname', type=str, default='sbm.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-7)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    siam_dynamics(args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.L, args.K, args.dt, beta = args.beta, nstep = nstep, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, use_mode_combination=True, nbmax=args.nbmax, nhilbmax=args.nhilbmax)
