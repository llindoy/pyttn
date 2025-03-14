import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import time
import sys
import h5py
import scipy
import copy

import pyttn
from pyttn import oqs, utils
from numba import jit


def sbm_dynamics_timing(Nb, alpha, wc, s, eps, delta, chi, nbose, dt, beta = None, nstep = 1, Nw = 10.0, degree = 2, compress=True, adaptive=False, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, nbmax=1, nhilbmax=1024):
    r"""A function for setting up and running dynamics of the spin boson model using the Star Hamiltonian and timing the results.
    Here we consider a Hamiltonian of the form

    .. math::
        \hat{H} = \frac{\epsilon}{2} \hat{\sigma}_z + \frac{\Delta}{2}\hat{\sigma}_x  + \sum_{k=1}^{N_b} \hat{\sigma}_z \left(\hat{a}_k^\dagger + \hat{a}_k \right) + \sum_{k=1}^{N_b} \omega_k \hat{a}_k^\dagger \hat{a}_k

    where the coupling constants and frequencies are obtained by discretising the continuous spectral density
    .. math::
        J(\omega) = \frac{\pi}{2} \frac{\alpha}{\omega_c^{s-1}} \exp\left(-\frac{\omega}{\omega_c}\right)


    :param Nb: The number of bath modes to use in the discretisation
    :type Nb: int
    :param alpha: The Kondo parameter associated with the bath
    :type alpha: float
    :param wc: The bath cutoff frequency
    :type wc: float
    :param s: The exponent in the bath spectral density
    :type s: float
    :param eps: The bias in the spin boson model Hamiltonian
    :type eps: float
    :param delta: The tunnelling matrix element in the spin boson model Hamiltonian
    :type delta: float
    :param chi: The fixed bond dimension used throughout the tensor network
    :type chi: int
    :param nbose: The fixed local Hilbert space dimension to use for all modes in the bath
    :type nbose: int
    :param dt: The timestep used for integration
    :type dt: float
    :param beta: The inverse temperature to use in the simulation.  For beta=None we perform a zero temperature simulation (default: None)
    :type beta: float, optional
    :param nstep: The number of timesteps to perform (default: 1)
    :type nstep: int, optional
    :param Nw: A factor (that multiplies wc) to define a hard frequency limit for the discretisation approach.  (default: 10)
    :type Nw: float, optional
    :param degree: The degree of the tree tensor network used to represent the bath (default: 2)
    :type degree: int, optional
    :param compress: Whether or not to compress the SOP operator into a hierarchical SOP form (default: True)
    :type compress: bool
    :param adaptive: Whether or not to use the adaptive subspace integration scheme (default: False)
    :type adaptive: bool
    :param spawning_threshold: The threshold parameter used to decide whether or not to spawn new basis functions (default: 2e-4)
    :type spawning_threshold: float, optional
    :param unoccupied_threshold: The threshold parameter used to decide whether or not a basis function is occupied (default: 1e-4)
    :type unoccupied_threshold: float, optional
    :param nunoccupied: The minimum number of unoccupied basis functions to have at all times (default: 0)
    :type nunoccupied: int, optional
    :param nbmax: The maximum number of modes to combine together for mode combination(default: 1)
    :type nbmax: int, optional
    :param nhilbmax: The maximum local Hilbert space dimension allowed with mode combination (default: 1024)
    :type nhilbmax: int, optional

    :return: The mean and standard deviation of the timings of each step
    :rtype: list
    """

    t = np.arange(nstep+1)*dt
    """
    Set up the system information
    """
    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi / 2 * alpha * wc * np.power(w / wc, s) * np.exp(-np.abs(w / wc))) * np.where(w > 0, 1.0, -1.0)

    # set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)

    # discretise the bath correleation function using the orthonormal polynomial based cutoff
    g, w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc) )

    # set up the discretised bath object
    discbath = oqs.DiscreteBosonicBath(g, w)

    # truncate the system bath modes
    discbath.truncate_modes(utils.DepthTruncation(nbose))

    # set up the total system information object including both system information and bath information
    sysinf = pyttn.system_modes(1)
    sysinf[0] = pyttn.tls_mode()

    # set up the mode combination informatino
    if nbmax > 1:
        mode_comb = utils.ModeCombination(nbmax, nhilbmax)
        bsys = discbath.system_information(mode_comb)
    else:
        bsys = discbath.system_information()
    sysinf = pyttn.combine_systems(sysinf, bsys)

    """
    Set up the Hamiltonian for the discretised model
    """
    # set up the total Hamiltonian
    H = pyttn.SOP(sysinf.nprimitive_modes())

    # add on the system part of the system bath Hamiltonian
    H += eps * pyttn.sOP("sz", 0) + delta * pyttn.sOP("sx", 0)

    # add the system bath Hamiltonian terms to the Hamiltonian
    H = discbath.add_system_bath_hamiltonian(H, pyttn.sOP("sz", 0), geom="star")

    """
    Set up the TTN structures and initial state of the wavefunction. 
    Here we make use of the discrete bath add_bath_tree function to add each of the trees
    """
    # build the trees for the system mode and
    topo = pyttn.ntree("(1(2(2)))")
    capacity = pyttn.ntree("(1(2(2)))")
    linds = discbath.add_bath_tree(topo(), degree, chi, min(chi, nbose))
    discbath.add_bath_tree(capacity(), degree, chi, min(chi, nbose))

    #construct and initialise the ttn wavefunction
    A = pyttn.ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(sysinf.nmodes())])

    #set up the Hamiltonian
    h = pyttn.sop_operator(H, A, sysinf, compress=compress)

    #set up tdvp sweeping algorithm parameters
    sweep = None
    if not adaptive:
        sweep = pyttn.tdvp(A, h, krylov_dim = 12)
    else:
        sweep = pyttn.tdvp(A, h, krylov_dim=12, subspace_neigs = 6, expansion='subspace')
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold=unoccupied_threshold
        sweep.minimum_unoccupied=nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j

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


