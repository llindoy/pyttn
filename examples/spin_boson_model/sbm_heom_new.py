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

def sbm_dynamics(alpha, wc, s, eps, delta, chi, L, K, dt, Lmin = None, beta = None, nstep = 1, ofname='sbm.h5', method='heom', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, use_mode_combination=True, nbmax=2, nhilbmax=1024):
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)

    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)
    dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=K, tmax=nstep*dt, Nt = nstep))

    expbath = oqs.ExpFitBosonicBath(dk, zk)
    expbath.truncate_modes(utils.EnergyTruncation(10*wc, Lmax=L, Lmin=Lmin))

    if use_mode_combination:
        mode_comb = utils.ModeCombination(nbmax, nhilbmax)
        bsys = expbath.system_information(mode_comb)
    else:
        bsys = expbath.system_information()

    #setup the system information object
    sysinf = system_modes(1)
    sysinf[0] = [tls_mode(), tls_mode()]

    #now attempt mode combination on the bath modes
    if use_mode_combination:
        mode_comb = utils.ModeCombination(nhilbmax, nbmax)
        bsys = mode_comb(bsys)

    #construct the system information object by combining the system information with the bath
    #information
    sysinf = combine_systems(sysinf, bsys)

    #set up the total Hamiltonian
    H = SOP(sysinf.nprimitive_modes())

    #add on the system liouvillian - here we are using that sz^T = sz and "sx^T=sx"
    H += (eps*sOP("sz", 0) + delta*sOP("sx", 0)) - (eps*sOP("sz", 1)+delta*sOP("sx", 1))
    H = expbath.add_system_bath_generator(H, [sOP("sz", 0), sOP("sz", 1)], method=method)

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 4

    topo = ntree("(1(4(4)))")
    capacity = ntree("(1(4(4)))")
    linds = expbath.add_bath_tree(topo(), degree, chi0, chi0)
    expbath.add_bath_tree(capacity(), degree, chi, chi)

    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(len(bsys)+1)])

    h = sop_operator(H, A, sysinf)

    #set up ttns storing the observable to be measured.  Here these are just system observables
    #so we form a tree with the same topology as A but with all bath bond dimensions set to 1
    obstree = ntree("(1(4(4)))")
    expbath.add_bath_tree(obstree(), degree, 1, 1)
    id_ttn = ttn(obstree, dtype=np.complex128)
    id_ttn.set_product([np.identity(2).flatten()] + expbath.identity_product_state(method=method))

    szop = site_operator(sOP("sz", 0), sysinf)
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
    res[0] = np.real(mel(szop, A, id_ttn))
    maxchi[0] = A.maximum_bond_dimension()

    #perform the dynamics
    renorm = mel(id_ttn, A)
    t1 = time.time()
    i=0
    print((i)*dt, res[i], np.real(renorm), maxchi[i], np.real(mel(A, A)))
    for i in range(nstep):
        sweep.step(A, h)
        renorm = mel(id_ttn, A)
        #A*=(1/renorm)
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
    t2 = time.time()
    print(t2-t1)

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
    parser.add_argument('--delta', type = float, default=1.0)
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
    parser.add_argument('--method', type=str, default='heom')


    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-10)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    fname = args.fname
    if fname == 'sbm.h5':
        fname = 'sbm_'+args.method+'.h5'

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.L, args.K, args.dt, beta = args.beta, nstep = nstep, ofname = fname, method=args.method, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, Lmin=args.Lmin, use_mode_combination=True, nbmax=args.nbmax, nhilbmax=args.nhilbmax)
