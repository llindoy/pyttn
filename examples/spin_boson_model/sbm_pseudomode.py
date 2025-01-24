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

def build_topology(nsys, chi, nbose, b_mode_dims, degree):
    topo = ntree("(1(%d(%d))(%d))"%(nsys, nsys, nsys))
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], b_mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[1], b_mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(topo)
    return topo

def observable_tree(obstree, op, b_mode_dims):
    Opttn = ttn(obstree, dtype=np.complex128)
    #setup the Sz tree state
    prod_state = [op.flatten()]
    for i in range(len(b_mode_dims)):
        state_vec = np.identity(int(np.sqrt(b_mode_dims[i])), dtype=np.complex128).flatten()
        prod_state.append(state_vec)
    Opttn.set_product(prod_state)
    return Opttn

def sbm_dynamics(alpha, wc, s, eps, delta, chi, L, K, dt, Lmin = None, beta = None, nstep = 1, ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, use_mode_combination=True, nbmax=2, nhilbmax=1024):
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)#np.where(np.abs(w) < 1, 1, 0)#

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)
    dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=K, tmax=nstep*dt, Nt = nstep))

    #set up the exp bath object this takes the dk and zk terms.  Truncate the modes and
    #extract the system information object from this.
    expbath = oqs.ExpFitBosonicBath(dk, zk)
    expbath.truncate_modes(utils.EnergyTruncation(10*wc, Lmax=L, Lmin=Lmin))
    bsys = expbath.system_information()

    gk = np.real(zk)
    Ek = np.imag(zk)

    Vk = np.real(np.sqrt(dk))
    Mk = -np.imag(np.sqrt(dk))

    Nb = bsys.nprimitive_modes()
    N = Nb+2

    #setup the system information object
    sysinf = system_modes(1)
    sysinf[0] = [tls_mode(), tls_mode()]

    #now attempt mode combination on the bath modes
    if use_mode_combination:
        mode_comb = utils.ModeCombination(nhilbmax, nbmax)
        bsys = mode_comb(bsys)

    #extract the bath mode dimensions
    b_mode_dims = np.zeros(len(bsys), dtype=int)
    for i in range(len(bsys)):
        b_mode_dims[i] = bsys[i].lhd()

    #construct the system information object by combining the system information with the bath
    #information
    sysinf = combine_systems(sysinf, bsys)
    print(sysinf.nprimitive_modes(), N)

    #set up the total Hamiltonian
    H = SOP(N)

    #add on the system liouvillian - here we are using that sz^T = sz and "sx^T=sx"
    Lsys = (eps*sOP("sz", 0) + delta*sOP("sx", 0)) - (eps*sOP("sz", 1)+delta*sOP("sx", 1))
    H += Lsys

    for i in range(len(zk)):
        i1 = 2*(i+1)
        i2 = 2*(i+1)+1

        H += complex(Ek[i])*(sOP("n", i1)-sOP("n", i2))
        H += 2.0j*complex(gk[i])*(sOP("a", i1)*sOP("a", i2)-0.5*(sOP("n", i1)+sOP("n", i2)))
        H += complex(Vk[i])*(sOP("sz", 0)*(sOP("adag", i1)+sOP("a", i1)) - sOP("sz", 1)*(sOP("adag", i2)+sOP("a", i2)))
        H += 2.0j*complex(Mk[i])*(sOP("sz", 1)*sOP("a", i1) - 0.5*(sOP("sz", 0)*sOP("a", i1) + sOP("sz", 1)*sOP("adag", i2)))
        H += 2.0j*complex(np.conj(Mk[i]))*(sOP("sz", 0)*sOP("a", i2) - 0.5*(sOP("sz", 0)*sOP("adag", i1) + sOP("sz", 1)*sOP("a", i2)))

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
    obstree = build_topology(sysinf[0].lhd(), 1, L, b_mode_dims, degree)
    id_ttn = observable_tree(obstree, np.identity(2), b_mode_dims)

    szop = site_operator(sOP("sz", 0), sysinf)

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
    res[0] = np.real(mel(szop, A, id_ttn))
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
    res[1] = np.real(mel(szop, A, id_ttn))
    maxchi[1] = A.maximum_bond_dimension()
    print((i)*dt, res[i], np.real(renorm), maxchi[i], np.real(mel(A, A)))
    sweep.dt = dt

    for i in range(1,nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        renorm = mel(id_ttn, A)
        res[i+1] = np.real(mel(szop, A, id_ttn))
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
    parser.add_argument('--K', type=int, default=6)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--L', type=int, default=30)
    parser.add_argument('--Lmin', type=int, default=6)

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
    parser.add_argument('--tmax', type=float, default=40)

    #output file name
    parser.add_argument('--fname', type=str, default='sbm_pseudomode.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=5e-7)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.L, args.K, args.dt, beta = args.beta, nstep = nstep, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, Lmin=args.Lmin, use_mode_combination=True, nbmax=args.nbmax, nhilbmax=args.nhilbmax)
