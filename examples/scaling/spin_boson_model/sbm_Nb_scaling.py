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

def setup_topology(chi, nbose, mode_dims, degree):
    topo = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[1], mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(topo)
    return topo


def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta = None, Ncut = 20, nstep = 1, Nw = 10.0, geom='star', ofname='sbm.h5', degree = 2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, nbmax=1, nhilbmax=1024):
    t = np.arange(nstep+1)*dt

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

    #add on the bath and system bath contributions of the bath hamiltonian
    oqs.add_bath_hamiltonian(H, sOP("sz", 0), g, w, Nb)

    #set up the local hilbert space dimensions of the bosonic modes
    mode_dims = [min(max(4, int(wc*Ncut/w[i])), nbose) for i in range(Nb)]

    #set up the mode combination informatino
    mode_comb = utils.ModeCombination(nbmax, nhilbmax)
    composite_modes = mode_comb(mode_dims, [x+1 for x in range(Nb)])
    Nbc = len(composite_modes)

    #setup the system information object.  Additionally work out the local hilbert space dimensions os that we can set up the system mode informmation
    sysinf = system_modes(1+Nbc)
    sysinf[0] = tls_mode()
    tree_mode_dims = []
    for ind, cmode in enumerate(composite_modes):
        print(ind, cmode)
        sysinf[ind+1] = [boson_mode(mode_dims[x-1]) for x in cmode]
        tree_mode_dims.append(sysinf[ind+1].lhd())

    #construct the topology and capacity trees used for constructing 
    chi0 = chi
    if adaptive:
        chi0 = 4

    #now build the topology and capacity arrays
    topo = setup_topology(chi0, nbose, tree_mode_dims, degree)
    capacity = setup_topology(chi, nbose, tree_mode_dims, degree)

    #construct and initialise the ttn wavefunction
    A = ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(Nbc+1)])

    #set up the Hamiltonian
    h = sop_operator(H, A, sysinf, identity_opt=True, compress=True)

    #construct objects need for evaluating observables
    mel = matrix_element(A)

    #set up the observable to measure
    op = site_operator(sOP("sz", 0), sysinf)


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

    if(geom == 'ipchain'):
        sweep.use_time_dependent_hamiltonian = True

    #run dynamics and measure properties storing them in a file
    res = np.zeros(nstep+1)
    maxchi = np.zeros(nstep+1)
    res[0] = np.real(mel(op, A, A))
    maxchi[0] = A.maximum_bond_dimension()

    tp = 0
    ts = np.logspace(np.log10(dt*1e-5), np.log10(dt), 5)
    for i in range(5):
        dti = ts[i]-tp
        print(ts, dt)
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]
    i=1
    res[1] =np.real(mel(op, A, A))
    maxchi[1] = A.maximum_bond_dimension()
    sweep.dt = dt

    for i in range(1,nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        res[i+1] = np.real(mel(op, A, A))
        maxchi[i+1] = A.maximum_bond_dimension()
        print(i, res[i+1], A.maximum_bond_dimension())

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
    parser.add_argument('--N', type=int, default=16)

    #geometry to be used for bath dynamics
    parser.add_argument('--geom', type = str, default='star')

    #system hamiltonian parameters
    parser.add_argument('--delta', type = float, default=1)
    parser.add_argument('--eps', type = float, default=0)

    #bath inverse temperature
    parser.add_argument('--beta', type = float, default=None)

    #maximum bond dimension
    parser.add_argument('--chi', type=int, default=64)
    parser.add_argument('--degree', type=int, default=1)

    #maximum bosonic hilbert space dimension
    parser.add_argument('--nbose', type=int, default=20)

    #integration time parameters
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--tmax', type=float, default=40)

    #output file name
    parser.add_argument('--fname', type=str, default='sbm.h5')

    #the minimum number of unoccupied modes for the dynamics
    parser.add_argument('--subspace', type=bool, default = True)
    parser.add_argument('--nunoccupied', type=int, default=0)
    parser.add_argument('--spawning_threshold', type=float, default=1e-6)
    parser.add_argument('--unoccupied_threshold', type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax/args.dt)+1
    sbm_dynamics(args.N, args.alpha, args.wc, args.s, args.eps, args.delta, args.chi, args.nbose, args.dt, beta = args.beta, nstep = nstep, geom=args.geom, ofname = args.fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, unoccupied_threshold = args.unoccupied_threshold, adaptive = args.subspace, degree = args.degree)
