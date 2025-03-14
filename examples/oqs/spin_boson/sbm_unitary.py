import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import h5py

import pyttn
from pyttn import oqs, utils
from numba import jit

def output_results(ofname, t, Sz, maxchi):
    h5 = h5py.File(ofname, "w")
    h5.create_dataset("t", data=t)
    h5.create_dataset("Sz", data=Sz)
    h5.create_dataset("maxchi", data=maxchi)
    h5.close()

def run_first_step(sweep, A, h, dt, nstep=5, nscale=1e-5):
    tp = 0
    ts = np.logspace(np.log10(dt * nscale), np.log10(dt), nstep)
    for i in range(nstep):
        dti = ts[i] - tp
        print(ts, dt)
        sweep.dt = dti
        sweep.step(A, h)
        tp = ts[i]

def sbm_dynamics(Nb,alpha, wc, s, eps, delta, chi, nbose, dt, beta=None, nstep=1, Nw=10.0, geom="star", ofname="sbm_unitary.h5", degree=2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, nbmax=2, nhilbmax=1024):
    r"""A function for setting up and running dynamics of the spin boson model using the a Hamiltonian with a user specified geometry.
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
    :param geom: The geometry to use for the Hamiltonian.   (default: "star")
    :type geom: str, optional
    :param ofname: The output filename.   (default: "sbm_unitary.h5")
    :type ofname: str, optional
    :param degree: The degree of the tree tensor network used to represent the bath (default: 2)
    :type degree: int, optional
    :param adaptive: Whether or not to use the adaptive subspace integration scheme (default: False)
    :type adaptive: bool
    :param spawning_threshold: The threshold parameter used to decide whether or not to spawn new basis functions (default: 2e-4)
    :type spawning_threshold: float, optional
    :param unoccupied_threshold: The threshold parameter used to decide whether or not a basis function is occupied (default: 1e-4)
    :type unoccupied_threshold: float, optional
    :param nunoccupied: The minimum number of unoccupied basis functions to have at all times (default: 0)
    :type nunoccupied: int, optional
    :param nbmax: The maximum number of modes to combine together for mode combination(default: 2)
    :type nbmax: int, optional
    :param nhilbmax: The maximum local Hilbert space dimension allowed with mode combination (default: 1024)
    :type nhilbmax: int, optional
    """
    
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
    discbath.truncate_modes(utils.EnergyTruncation(10 * wc, Lmax=nbose, Lmin=4))

    # and get the bath information

    # set up the mode combination informatino
    if nbmax > 1:
        mode_comb = utils.ModeCombination(nbmax, nhilbmax)
        bsys = discbath.system_information(mode_comb)
    else:
        bsys = discbath.system_information()

    # set up the total system information object including both system information and bath information
    sysinf = pyttn.system_modes(1)
    sysinf[0] = pyttn.tls_mode()
    sysinf = pyttn.combine_systems(sysinf, bsys)

    """
    Set up the Hamiltonian for the discretised model
    """
    # set up the total Hamiltonian
    H = pyttn.SOP(sysinf.nprimitive_modes())

    # add on the system part of the system bath Hamiltonian
    H += eps/2 * pyttn.sOP("sz", 0) + delta/2 * pyttn.sOP("sx", 0)

    # add the system bath Hamiltonian terms to the Hamiltonian
    H = discbath.add_system_bath_hamiltonian(H, pyttn.sOP("sz", 0), geom=geom)

    """
    Set up the TTN structures and initial state of the wavefunction. 
    Here we make use of the discrete bath add_bath_tree function to add each of the trees
    """
    # construct the topology and capacity trees used for constructing
    chi0 = chi
    if adaptive:
        chi0 = min(4, chi)

    # build the trees for the system mode and
    topo = pyttn.ntree("(1(2(2)))")
    capacity = pyttn.ntree("(1(2(2)))")
    _ = discbath.add_bath_tree(topo(), degree, chi0, min(chi0, nbose))
    discbath.add_bath_tree(capacity(), degree, chi, min(chi, nbose))

    # construct and initialise the ttn wavefunction
    A = pyttn.ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(sysinf.nmodes())])

    """
    Set up the operator objects representing the Hamiltonian and observables of interest
    """
    # set up the Hamiltonian as a sop object
    h = pyttn.sop_operator(H, A, sysinf)

    # set up the observable to measure
    op = pyttn.site_operator(pyttn.sOP("sz", 0), sysinf)

    """
    Set up objects used for computing matrix elements and performing the time evolution
    """
    # construct objects need for evaluating observables
    mel = pyttn.matrix_element(A)

    # set up tdvp sweeping algorithm parameters
    sweep = None
    if not adaptive:
        sweep = pyttn.tdvp(A, h, krylov_dim=16)
    else:
        sweep = pyttn.tdvp(A, h, krylov_dim=16, subspace_neigs=2, expansion="subspace")
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold = unoccupied_threshold
        sweep.minimum_unoccupied = nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j

    """
    Perform the time evolution and measure the required observables dumping to a hdf5 file.
    """
    # run dynamics and measure properties storing them in a file
    t=(np.arange(nstep + 1) * dt)
    Sz = np.zeros(nstep + 1)
    maxchi = np.zeros(nstep + 1)
    Sz[0] = np.real(mel(op, A, A))
    maxchi[0] = A.maximum_bond_dimension()

    # perform the first timestep using a logarithmic discretisation of time over this period.
    # This can be useful to allow for suitable adaptation of weakly occupied single particle
    # functions through the initial time point.
    run_first_step(sweep, A, h, dt, 5)

    # set the values after the first timestep
    Sz[1] = np.real(mel(op, A, A))
    maxchi[1] = A.maximum_bond_dimension()
    sweep.dt = dt

    # now perform standard time stepping
    for i in range(1, nstep):
        sweep.step(A, h)
        Sz[i + 1] = np.real(mel(op, A, A))
        maxchi[i + 1] = A.maximum_bond_dimension()
        print(i, Sz[i + 1], A.maximum_bond_dimension())

        # outputting results to files every 10 steps
        if i % 10 == 0:
            output_results(ofname, t, Sz, maxchi)


    # and finally dump everything to file at the end of the simulation
    output_results(ofname, t, Sz, maxchi)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Dynamics of the spin boson model with unitary dynamics methods"
    )

    # exponential bath cutoff parameters
    parser.add_argument("alpha", type=float)
    parser.add_argument("--wc", type=float, default=5)
    parser.add_argument("--s", type=float, default=1)

    # number of bath modes
    parser.add_argument("--N", type=int, default=8)

    # geometry to be used for bath dynamics
    parser.add_argument("--geom", type=str, default="star")

    # system hamiltonian parameters
    parser.add_argument("--delta", type=float, default=1)
    parser.add_argument("--eps", type=float, default=0)

    # bath inverse temperature
    parser.add_argument("--beta", type=float, default=None)

    # maximum bond dimension
    parser.add_argument("--chi", type=int, default=32)
    parser.add_argument("--degree", type=int, default=2)

    # maximum bosonic hilbert space dimension
    parser.add_argument("--nbose", type=int, default=30)

    # integration time parameters
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--tmax", type=float, default=10)

    # output file name
    parser.add_argument("--fname", type=str, default=None)

    # the minimum number of unoccupied modes for the dynamics
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-5)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)

    args = parser.parse_args()

    fname = args.fname
    if fname is None:
        fname = "sbm_"+args.geom+".h5"

    nstep = int(args.tmax / args.dt) + 1
    sbm_dynamics(
        args.N,
        args.alpha,
        args.wc,
        args.s,
        args.eps,
        args.delta,
        args.chi,
        args.nbose,
        args.dt,
        beta=args.beta,
        nstep=nstep,
        geom=args.geom,
        ofname=fname,
        nunoccupied=args.nunoccupied,
        spawning_threshold=args.spawning_threshold,
        unoccupied_threshold=args.unoccupied_threshold,
        adaptive=args.subspace,
        degree=args.degree,
    )
