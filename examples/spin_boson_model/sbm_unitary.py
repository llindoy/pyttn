import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../")
from pyttn import *
from pyttn import oqs, utils
from numba import jit
from pyttn.utils import visualise_tree

import matplotlib.pyplot as plt

def setup_topology(chi, nbose, mode_dims, degree):
    topo = ntree("(1(2(2))(2))")

    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[1], mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(topo)
    return topo

def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta = None, Ncut = 20, nstep = 1, Nw = 10.0, geom='star', ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, use_mode_combination=True, nbmax=2, nhilbmax=1024):
    t = np.arange(nstep+1)*dt

    """
    Set up the system bath Hamiltonian
    """
    #setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi/2*alpha*wc*np.power(w/wc, s)*np.exp(-np.abs(w/wc)))*np.where(w > 0, 1.0, -1.0)


    #set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    #discretise the bath correleation function using the orthonormal polynomial based cutoff 
    g,w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw*wc), Nw*wc))

    #set up the total Hamiltonian
    N = Nb+1
    H = SOP(N)

    #add on the system part of the system bath Hamiltonian
    H += eps*sOP("sz", 0) + delta*sOP("sx", 0)

    #set up the discretised bath object
    discbath = oqs.DiscreteBosonicBath(g, w)

    #add the system bath Hamiltonian terms to the Hamiltonian
    H = discbath.add_system_bath_hamiltonian(H, sOP("sz", 0), geom=geom)

    #truncate the system bath modes
    discbath.truncate_modes(utils.EnergyTruncation(10*wc, Lmax=nbose, Lmin=4))

    #and get the bath information

    #set up the mode combination informatino
    if use_mode_combination:
        mode_comb = utils.ModeCombination(nbmax, nhilbmax)
        bsys = discbath.system_information(mode_comb)
    else:
        bsys = discbath.system_information()

    #set up the total system information object including both system information and bath information
    sysinf = system_modes(1)
    sysinf[0] = tls_mode()
    sysinf = combine_systems(sysinf, bsys)


    """
    Set up the TTN structures and initial state of the wavefunction. 
    Here we make use of the discrete bath add_bath_tree function to add each of the trees
    """
    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = min(4, chi)

    #build the trees for the system mode and 
    topo = ntree("(1(2(2)))")
    capacity = ntree("(1(2(2)))")
    linds = discbath.add_bath_tree(topo(), degree, chi0, min(chi0, nbose))
    discbath.add_bath_tree(capacity(), degree, chi, min(chi, nbose))

    #construct and initialise the ttn wavefunction
    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(len(bsys)+1)])


    """
    Set up the operator objects representing the Hamiltonian and observables of interest
    """
    #set up the Hamiltonian as a sop object
    h = sop_operator(H, A, sysinf)

    #set up the observable to measure
    op = site_operator(sOP("sz", 0), sysinf)


    """
    Set up objects used for computing matrix elements and performing the time evolution
    """
    #construct objects need for evaluating observables
    mel = matrix_element(A)

    #set up tdvp sweeping algorithm parameters
    sweep = None
    if not adaptive:
        sweep = tdvp(A, h, krylov_dim = 16)
    else:
        sweep = tdvp(A, h, krylov_dim=16, subspace_neigs = 2, expansion='subspace')
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold=unoccupied_threshold
        sweep.minimum_unoccupied=nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j


    """
    Perform the time evolution and measure the required observables dumping to a hdf5 file.
    """
    #run dynamics and measure properties storing them in a file
    Sz = np.zeros(nstep+1)
    maxchi = np.zeros(nstep+1)
    Sz[0] = np.real(mel(op, A, A))
    maxchi[0] = A.maximum_bond_dimension()

    #perform the first timestep using a logarithmic discretisation of time over this period.  
    #This can be useful to allow for suitable adaptation of weakly occupied single particle 
    #functions through the initial time point.
    tp = 0
    ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), 5)
    for i in range(5):
        dti = ts[i]-tp
        print(ts, dt)
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
    i=1

    #set the values after the first timestep
    Sz[1] =np.real(mel(op, A, A))
    maxchi[1] = A.maximum_bond_dimension()
    sweep.dt = dt

    #now perform standard time stepping
    for i in range(1,nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        Sz[i+1] = np.real(mel(op, A, A))
        maxchi[i+1] = A.maximum_bond_dimension()
        print(i, Sz[i+1], A.maximum_bond_dimension())


        #outputting results to files every 10 steps
        if(i % 10 == 0):
            h5 = h5py.File(ofname, 'w')
            h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
            h5.create_dataset('Sz', data=Sz)
            h5.create_dataset('maxchi', data=maxchi)
            h5.close()
                
    #and finally dump everything to file at the end of the simulation
    h5 = h5py.File(ofname, 'w')
    h5.create_dataset('t', data=(np.arange(nstep+1)*dt))
    h5.create_dataset('Sz', data=Sz)
    h5.create_dataset('maxchi', data=maxchi)
    h5.close()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Dynamics of the zero temperature spin boson model with')

    #exponential bath cutoff parameters
    parser.add_argument('alpha', type = float)
    parser.add_argument('--wc', type = float, default=5)
    parser.add_argument('--s', type = float, default=1)

    #number of bath modes
    parser.add_argument('--N', type=int, default=8)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='chain')

    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=1)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=32)
    parser.add_argument('--degree', type=int, default=2)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=30)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--tmax', type=float, default=10)

    #output file name
    parser.add_argument('--fname', type=str, default='sbm.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-5)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.N, args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree)
