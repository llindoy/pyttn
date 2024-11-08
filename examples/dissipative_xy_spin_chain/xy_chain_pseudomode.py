import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
import pyttn
from pyttn import *
from pyttn import oqs, utils

from numba import jit

def build_topology(Ns, ds, chiS,  chi, nbose, b_mode_dims, degree):
    lchi = [chiS for i in range(Ns)]
    topo = ntreeBuilder.mps_tree(lchi, chiS)

    leaf_indices=topo.leaf_indices()
    for li in leaf_indices:
        topo.at(li).insert(ds)
        if degree == 1:
            ntreeBuilder.mps_subtree(topo.at(li), b_mode_dims, chi, min(chi, nbose))
        else:
            ntreeBuilder.mlmctdh_subtree(topo.at(li), b_mode_dims, degree, chi)

    ntreeBuilder.sanitise(topo)
    return topo

def observable_tree(Ns, obstree, op, b_mode_dims):
    Opttn = ttn(obstree, dtype=np.complex128)
    #setup the Sz tree state

    prod_state = []
    for i in range(Ns):
        prod_state.append(op.flatten())
        for i in range(len(b_mode_dims)):
            state_vec = np.identity(int(np.sqrt(b_mode_dims[i])), dtype=np.complex128).flatten()
            prod_state.append(state_vec)

    Opttn.set_product(prod_state)
    return Opttn

def xychain_dynamics(Ns, alpha, wc, eta, chiS, chi, L, K, dt, Lmin = None, beta = None, nstep = 1, ofname='xychain.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, use_mode_combination=True, nbmax=2, nhilbmax=1024):
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(2*np.pi*alpha*w*np.exp(-np.abs(w/wc))**2)*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta, wmax=wc*100)
    dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=K, tmax=nstep*dt, Nt = nstep))

    #set up the exp bath object this takes the dk and zk terms.  Truncate the modes and
    #extract the system information object from this.
    expbath = oqs.ExpFitBosonicBath(dk, zk)
    expbath.truncate_modes(utils.EnergyTruncation(15*wc, Lmax=L, Lmin=Lmin))
    bsys = expbath.system_information()

    gk = np.real(zk)
    Ek = np.imag(zk)

    Vk = np.real(np.sqrt(dk))
    Mk = -np.imag(np.sqrt(dk))

    Nb = bsys.nprimitive_modes()
    N = (Nb+2)*Ns

    #set up the system information object for a single spin
    #setup the system information object
    sysinf = system_modes(1)
    sysinf[0] = [spin_mode(2), spin_mode(2)]

    #now attempt mode combination on the bath modes
    if use_mode_combination:
        mode_comb = utils.ModeCombination(nhilbmax, nbmax)
        bsys = mode_comb(bsys)

    #extract the bath mode dimensions
    b_mode_dims = np.zeros(len(bsys), dtype=int)
    for i in range(len(bsys)):
        b_mode_dims[i] = bsys[i].lhd()

    sysinf = combine_systems(sysinf, bsys)

    #set the total system information object to just be a single spin
    sysinfo = copy.deepcopy(sysinf)

    #and add on the system information objects for the remaining spins
    for i in range(Ns-1):
        sysinfo = combine_systems(sysinfo, sysinf)

    #set up the total Hamiltonian
    H = SOP(sysinfo.nprimitive_modes())

    #set up the interactions for each spin and its bath
    for si in range(Ns):
        skip = si*(Nb+2)

        #the onsite energy terms
        H += sOP("sz", skip) - sOP("sz", skip+1)

        #add on the HEOM bath Hamiltonian
        for i in range(len(zk)):
            i1 = skip+2*(i+1)
            i2 = skip+2*(i+1)+1
            print(i1, i2, sysinfo.nprimitive_modes())

            H += complex(Ek[i])*(sOP("n", i1)-sOP("n", i2))
            H += 2.0j*complex(gk[i])*(sOP("a", i1)*sOP("a", i2)-0.5*(sOP("n", i1)+sOP("n", i2)))
            H += complex(Vk[i])*(sOP("sz", skip)*(sOP("adag", i1)+sOP("a", i1)) - sOP("sz", skip+1)*(sOP("adag", i2)+sOP("a", i2)))
            H += 2.0j*complex(Mk[i])*(sOP("sz", skip+1)*sOP("a", i1) - 0.5*(sOP("sz", skip)*sOP("a", i1) + sOP("sz", skip+1)*sOP("adag", i2)))
            H += 2.0j*complex(np.conj(Mk[i]))*(sOP("sz", skip)*sOP("a", i2) - 0.5*(sOP("sz", skip)*sOP("adag", i1) + sOP("sz", skip+1)*sOP("a", i2)))

    #now we add on the spin-spin coupling terms
    for si in range(Ns-1):
        s1 = si*(Nb+2)
        s2 = (si+1)*(Nb+2)

        H += (1.0-eta)*(sOP("sx", s1)*sOP("sx", s2) - sOP("sx", s1+1)*sOP("sx", s2+1))
        H += (1.0+eta)*(sOP("sy", s1)*sOP("sy", s2) - sOP("sy", s1+1)*sOP("sy", s2+1))


    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    chiS0 = chiS
    if adaptive:
        chi0 = 16
        chiS0 = 16

    topo = build_topology(Ns, sysinfo[0].lhd(), chiS0, chi0, L, b_mode_dims, degree)
    capacity = build_topology(Ns, sysinfo[0].lhd(), chiS, chi, L, b_mode_dims, degree)

    #utils.visualise_tree(topo)
    #import matplotlib.pyplot as plt
    #plt.show()
    #exit()
    A = ttn(topo, capacity, dtype=np.complex128)

    state = [0 for i in range(Ns*(len(bsys)+1))]
    state[(Ns-1)//2*(len(bsys)+1)]=3
    print(state)
    A.set_state(state)

    h = sop_operator(H, A, sysinfo)
    #set up ttns storing the observable to be measured.  Here these are just system observables
    #so we form a tree with the same topology as A but with all bath bond dimensions set to 1
    obstree = build_topology(Ns, sysinfo[0].lhd(), 1, 1, L, b_mode_dims, degree)
    id_ttn = observable_tree(Ns, obstree, np.identity(2), b_mode_dims)

    ops = []
    for si in range(Ns):
        skip = si*(Nb+2)
        ops.append(site_operator(sOP("sz", skip), sysinfo))

    mel = matrix_element(A)

    #set up the tdvp object
    sweep = None
    if not adaptive:
        sweep = tdvp(A, h, krylov_dim = 15)
    else:
        sweep = tdvp(A, h, krylov_dim = 16, expansion='subspace', subspace_neigs=8, subspace_krylov_dim=16)
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold=unoccupied_threshold
        sweep.minimum_unoccupied=nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j

    res = np.zeros((Ns, nstep+1), dtype=np.complex128)
    Rnorm = np.ones(nstep+1, dtype=np.complex128)
    maxchi = np.zeros(nstep+1)
    norm = np.zeros(nstep+1)

    for i in range(Ns):
        res[i, 0] = mel(ops[i], A, id_ttn)
    maxchi[0] = A.maximum_bond_dimension()


    #perform the dynamics
    renorm = mel(id_ttn, A)
    i=0
    print((i)*dt, res[Ns//2, i], np.real(renorm), maxchi[i], np.real(mel(A, A)))
    norm[i] = np.real(mel(A,A))

    tp = 0
    ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), 5)
    for i in range(5):
        dti = ts[i]-tp
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
        renorm = mel(id_ttn, A)

    i=1
    for si in range(Ns):
        res[si, 1] = mel(ops[si], A, id_ttn)
    maxchi[1] = A.maximum_bond_dimension()
    norm[i] = np.real(mel(A,A))
    Rnorm[1]=(1/renorm)
    print((i)*dt, res[Ns//2, i], np.real(renorm), maxchi[i], np.real(mel(A, A)))
    sweep.dt = dt

    for i in range(1, nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        renorm = mel(id_ttn, A)
        for si in range(Ns):
            res[si, i+1] = mel(ops[si], A, id_ttn)
        maxchi[i+1] = A.maximum_bond_dimension()
        norm[i+1] = np.real(mel(A,A))
        Rnorm[i+1]=(1/renorm)

        print((i+1)*dt, res[Ns//2, i+1], np.real(renorm), maxchi[i+1], np.real(mel(A, A)))

        h5 = h5py.File(ofname, 'w')
        h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
        for si in range(Ns):
            h5.create_dataset('Sz'+str(si), data=np.real(res[si, :]))
            h5.create_dataset('rSz'+str(si), data=np.real(res[si, :]*Rnorm))
        h5.create_dataset('maxchi', data=maxchi)
        h5.create_dataset('norm', data=norm)
        h5.close()

    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    for si in range(Ns):
        h5.create_dataset('Sz'+str(si), data=np.real(res[si, :]))
        h5.create_dataset('rSz'+str(si), data=np.real(res[si, :]*Rnorm))
    h5.create_dataset('maxchi', data=maxchi)
    h5.create_dataset('norm', data=norm)
    h5.close()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #number of spins in the system
    parser.add_argument('--Ns', type=int, default=21)

    #exponential bath cutoff parameters
    parser.add_argument('--alpha', type = float, default=0.32)
    parser.add_argument('--wc', type = float, default=4)

    #number of bath modes
    parser.add_argument('--K', type=int, default=4)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--Lmin', type=int, default=4)

    #mode combination parameters
    parser.add_argument('--nbmax', type=int, default=2)
    parser.add_argument('--nhilbmax', type=int, default=1000)


    #system hamiltonian parameters
    parser.add_argument('--eta', type = float, default=0.04)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--chiS', type=int, default=32)
    parser.add_argument('--degree', type=int, default=1)


    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--tmax', type=float, default=10)

    #output file name
    parser.add_argument('--fname', type=str, default='xychain_pm.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=2e-6)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-6)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1

    xychain_dynamics(args.Ns, args.alpha, args.wc, args.eta, args.chiS, args.chi, args.L, args.K, args.dt, beta = args.beta, nstep = nstep, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, Lmin=args.Lmin, use_mode_combination=True, nbmax=args.nbmax, nhilbmax=args.nhilbmax)
