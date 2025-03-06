import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../../")
from pyttn import *
from pyttn import oqs, utils
from numba import jit
import matplotlib.pyplot as plt
from cayley_helper import get_spin_connectivity, build_topology


def xychain_dynamics(Nl, Nb, alpha, wc, eta, chi, chiS, chiB, nbose, dt, nbose_min = None, beta = None, nstep = 1, Nw=4.0, geom='ipchain', ofname='xychain.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, use_mode_combination=True, nbmax=2, nhilbmax=1024, cayley_degree=3):
    t = np.arange(nstep+1)*dt

    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return 2*np.pi*alpha*w*np.exp(-np.abs(w/wc)**2)

    #setup the system information object.  Additionally work out the local hilbert space dimensions os that we can set up the system mode informmation
    tree_mode_dims = []
    sysinf = system_modes(1)
    sysinf[0] = spin_mode(2)

    Nbc = 0
    if Nb != 0:
        #set up the open quantum system bath object
        bath = oqs.BosonicBath(J, beta=beta)

        #discretise the bath correleation function using the orthonormal polynomial based cutoff 
        g,w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw*wc), Nw*wc))


    #set up the total Hamiltonian
    N = Nb+1

    hiterms, Ns = get_spin_connectivity(Nl, d=3)

    #build the system Hamiltonian
    H = SOP(Ns*N)

    frequencies = None
    #add on the system part of the system bath Hamiltonian
    for si in range(Ns):
        skip = si*(Nb+1)
        H += sOP("sz", skip)
        print("adding spin " + str(si))
    
        binds = [skip+1+i for i in range(Nb)]

        #add on the bath and system bath contributions of the bath hamiltonian
        if Nb != 0:
            H, frequencies = oqs.unitary.add_bosonic_bath_hamiltonian(H, sOP("sz", skip), g, w, geom=geom, return_frequencies=True, binds=binds)

    #add on the spin coupling terms
    for ind in hiterms:
        print(ind[0], ind[1], Nb+1)
        skip1 = (ind[0])*(Nb+1)
        skip2 = (ind[1])*(Nb+1)
        H += (1-eta)*sOP("sx", skip1)*sOP("sx", skip2) + (1+eta)*sOP("sy", skip1)*sOP("sy", skip2) 

    if Nb != 0:
        #set up the local hilbert space dimensions of the bosonic modes
        b_mode_dims = [min(max(4, int(wc*20/frequencies[i])), nbose) for i in range(Nb)]
        bsys = system_modes(Nb)
        for i in range(Nb):
            bsys[i] = boson_mode(b_mode_dims[i])

        #set up the mode combination informatino
        mode_comb = utils.ModeCombination(nbmax, nhilbmax)
        bsys = mode_comb(bsys)
        sysinf = combine_systems(sysinf, bsys)

        for ind in range(len(bsys)):
            tree_mode_dims.append(bsys[ind].lhd())

        Nbc = len(bsys)
    print("hamiltonian string setup")

    sysinfo = copy.deepcopy(sysinf)
    #and add on the system information objects for the remaining spins
    for i in range(Ns-1):
        sysinfo = combine_systems(sysinfo, sysinf)

    sysinf=sysinfo

    print("system built")

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    chiS0 = chiS
    chiB0 = chiB
    if adaptive:
        chi0 = 8
        chiS0 = 8
        chiB0 = 8
    chi0=min(chi0, chi)
    chiS0=min(chiS0, chiS)
    chiB0=min(chiB0, chiB)

    #now build the topology and capacity arrays
    topo = build_topology(Nl, 2, chi0, chiS0, chiB0, nbose, tree_mode_dims, degree)
    capacity = build_topology(Nl, 2, chi, chiS, chiB, nbose, tree_mode_dims, degree)

    print("topology built")


    #utils.visualise_tree(topo)
    #plt.show()

    #construct and initialise the ttn wavefunction
    A = ttn(topo, capacity, dtype=np.complex128)
    state = [0 for i in range(Ns*(Nbc+1))]
    state[0]=1
    A.set_state(state)

    print("psi0 built")
    #set up the Hamiltonian as a sop object
    h = sop_operator(H, A, sysinf)
    print("H built")

    #construct objects need for evaluating observables
    mel = matrix_element(A)

    #set up the observable to measure
    ops = []
    for si in range(Ns):
        skip = si*(Nb+1)
        ops.append(site_operator(sOP("sz", skip), sysinf))

    #set up tdvp sweeping algorithm parameters
    sweep = None
    if not adaptive:
        sweep = tdvp(A, h, krylov_dim = 12)
    else:
        sweep = tdvp(A, h, krylov_dim=12, subspace_neigs = 6, expansion='subspace')
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold=unoccupied_threshold
        sweep.minimum_unoccupied=nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j

    t1 = time.time()
    #run dynamics and measure properties storing them in a file
    res = np.zeros((Ns, nstep+1), dtype=np.complex128)
    maxchi = np.zeros(nstep+1)
    for i in range(Ns):
        res[i, 0] = mel(ops[i], A)
    maxchi[0] = A.maximum_bond_dimension()

    #perform the first timestep using a logarithmic discretisation of time over this period.  
    #This can be useful to allow for suitable adaptation of weakly occupied single particle 
    #functions through the initial time point.
    tp = 0
    ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), 5)
    for i in range(5):
        dti = ts[i]-tp
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
    i=1

    #set the values after the first timestep
    for si in range(Ns):
        res[si, 1] = mel(ops[si], A)
    maxchi[1] = A.maximum_bond_dimension()
    sweep.dt = dt

    #now perform standard time stepping
    for i in range(1,nstep):
        sweep.step(A, h)
        for si in range(Ns):
            res[si, i+1] = mel(ops[si], A)
        maxchi[i+1] = A.maximum_bond_dimension()
        print(i, res[(Ns-1)//2, i+1], A.maximum_bond_dimension())

        t2 = time.time()
        #outputting results to files every 10 steps
        if(i % 1 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            for si in range(Ns):
                h5.create_dataset('Sz'+str(si), data=np.real(res[si, :]))
            h5.create_dataset('maxchi', data=maxchi)
            h5.create_dataset('time', data=np.array([t2-t1]))
            h5.close()
                
    t2 = time.time()
    #and finally dump everything to file at the end of the simulation
    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    for si in range(Ns):
        h5.create_dataset('Sz'+str(si), data=np.real(res[si, :]))
    h5.create_dataset('maxchi', data=maxchi)
    h5.create_dataset('time', data=np.array([t2-t1]))
    h5.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    parser.add_argument('--N', type=int, default=50)
    #number of spins in the system
    parser.add_argument('--Nl', type=int, default=3)

    #exponential bath cutoff parameters
    parser.add_argument('--alpha', type = float, default=0.32)
    parser.add_argument('--wc', type = float, default=4)

    #number of bath modes
    parser.add_argument('--geom', type = str, default='star')

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=20)
    parser.add_argument('--nbose_min', type=int, default=5)

    #mode combination parameters
    parser.add_argument('--nbmax', type=int, default=4)
    parser.add_argument('--nhilbmax', type=int, default=1000)


    #system hamiltonian parameters
    parser.add_argument('--eta', type = float, default=0.04)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=32)
    parser.add_argument('--chiS', type=int, default=24)
    parser.add_argument('--chiB', type=int, default=24)
    parser.add_argument('--degree', type=int, default=1)


    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--tmax', type=float, default=10)

    #output file name
    parser.add_argument('--fname', type=str, default='xytree.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-5)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1

    xychain_dynamics(args.Nl, args.N, args.alpha, args.wc, args.eta, args.chi, args.chiS, args.chiB, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree, nbose_min=args.nbose_min, use_mode_combination=True, nbmax=args.nbmax, nhilbmax=args.nhilbmax)

