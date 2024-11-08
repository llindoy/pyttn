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

def setup_topology(chi, nbose, mode_dims, degree):
    topo = ntree("(1(2(2))(2))")
    if(degree > 1):
        ntreeBuilder.mlmctdh_subtree(topo()[1], mode_dims, degree, chi)
    else:
        ntreeBuilder.mps_subtree(topo()[1], mode_dims, chi, min(chi, nbose))
    ntreeBuilder.sanitise(topo)
    return topo

def sbm_dynamics(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta = None, nstep = 1, Nw = 10.0, degree = 2, compress=True, adaptive=False, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, nbmax=1, nhilbmax=1024):
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
    H = oqs.add_bosonic_bath_hamiltonian(H, sOP("sz", 0), g, w)

    #set up the local hilbert space dimensions of the bosonic modes
    mode_dims = [nbose for i in range(Nb)]

    #set up the mode combination informatino
    mode_comb = utils.ModeCombination(nbmax, nhilbmax)
    composite_modes = mode_comb(mode_dims, [x+1 for x in range(Nb)])
    Nbc = len(composite_modes)

    #setup the system information object.  Additionally work out the local hilbert space dimensions os that we can set up the system mode informmation
    sysinf = system_modes(1+Nbc)
    sysinf[0] = qubit_mode()
    tree_mode_dims = []
    for ind, cmode in enumerate(composite_modes):
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
    h = sop_operator(H, A, sysinf, compress=compress)
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

    timings = np.zeros(nstep)
    for i in range(nstep):
        t1 = time.time()
        sweep.step(A, h)
        t2 = time.time()
        timings[i] = t2-t1

    stdev=0
    if(nstep > 1):
        stdev = np.std(timings)
    return np.mean(timings), stdev


