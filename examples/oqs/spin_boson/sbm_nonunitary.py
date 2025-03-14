import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import h5py
import argparse

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

def sbm_dynamics(alpha, wc, s, eps, delta, chi, L, K, dt, Lmin=None, beta=None, nstep=1, ofname="sbm_nonunitary.h5", degree=2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, nbmax=2, nhilbmax=1024, method = "heom"):
    r"""A function for setting up and running dynamics of the spin boson model using the non-unitary methods (HEOM or Pseudomode)
    Here we consider a Hamiltonian of the form

    .. math::
        \hat{H} = \frac{\epsilon}{2} \hat{\sigma}_z + \frac{\Delta}{2}\hat{\sigma}_x  + \sum_{k=1}^{N_b} \hat{\sigma}_z \left(\hat{a}_k^\dagger + \hat{a}_k \right) + \sum_{k=1}^{N_b} \omega_k \hat{a}_k^\dagger \hat{a}_k

    where the coupling constants and frequencies are obtained by discretising the continuous spectral density
    .. math::
        J(\omega) = \frac{\pi}{2} \frac{\alpha}{\omega_c^{s-1}} \exp\left(-\frac{\omega}{\omega_c}\right)


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
    :param L: The maximum depth of hierarchy per mode
    :type L: int
    :param K: The number of terms in the exponential decomposition
    :type K: int
    :param dt: The timestep used for integration
    :type dt: float
    :param L: The minimum depth of hierarchy per mode
    :type L: int
    :param beta: The inverse temperature to use in the simulation.  For beta=None we perform a zero temperature simulation (default: None)
    :type beta: float, optional
    :param nstep: The number of timesteps to perform (default: 1)
    :type nstep: int, optional
    :param ofname: The output filename.   (default: "sbm_nonunitary.h5")
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
    :param method: The method used to perform the dynamics.   (default: "orthopol")
    :type method: {"heom",  "pseudomode"}, optional
    """

    t = np.arange(nstep + 1) * dt

    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return np.abs(np.pi / 2 * alpha * wc * np.power(w / wc, s) * np.exp(-np.abs(w / wc))) * np.where(w > 0, 1.0, -1.0)

    # set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta)
    dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=K, tmax=nstep * dt, Nt=nstep))

    # setup the system information object
    sysinf = pyttn.system_modes(1)
    sysinf[0] = [pyttn.tls_mode(), pyttn.tls_mode()]

    #set up the bath information
    expbath = oqs.ExpFitBosonicBath(dk, zk)
    expbath.truncate_modes(utils.EnergyTruncation(10 * wc, Lmax=L, Lmin=Lmin))

    if nbmax>1:
        mode_comb = utils.ModeCombination(nhilbmax, nbmax)
        bsys = expbath.system_information(mode_comb)
    else:
        bsys = expbath.system_information()

    # construct the system information object by combining the system information with the bath
    # information
    sysinf = pyttn.combine_systems(sysinf, bsys)

    # set up the total Hamiltonian
    H = pyttn.SOP(sysinf.nprimitive_modes())

    # add on the system liouvillian - here we are using that sz^T = sz and "sx^T=sx"
    Lsys = (eps/2 * pyttn.sOP("sz", 0) + delta/2 * pyttn.sOP("sx", 0)) - (eps/2 * pyttn.sOP("sz", 1) + delta/2 * pyttn.sOP("sx", 1))
    H += Lsys
    H = expbath.add_system_bath_generator(H, [pyttn.sOP("sz", 0), pyttn.sOP("sz", 1)], method=method)

    # construct the topology and capacity trees used for constructing
    chi0 = chi
    if adaptive:
        chi0 = 4

    topo = pyttn.ntree("(1(4(4)))")
    capacity = pyttn.ntree("(1(4(4)))")
    _ = expbath.add_bath_tree(topo(), degree, chi0, chi0)
    expbath.add_bath_tree(capacity(), degree, chi, chi)

    A = pyttn.ttn(topo, capacity, dtype=np.complex128)
    A.set_state([0 for i in range(len(sysinf))])

    h = pyttn.sop_operator(H, A, sysinf)
    # set up ttns storing the observable to be measured.  Here these are just system observables
    # so we form a tree with the same topology as A but with all bath bond dimensions set to 1
    obstree = pyttn.ntree("(1(4(4)))")
    expbath.add_bath_tree(obstree(), degree, 1, 1)
    trace_ttn = pyttn.ttn(obstree, dtype=np.complex128)
    trace_ttn.set_product([np.identity(2).flatten()] + expbath.identity_product_state(method=method))


    szop = pyttn.site_operator(pyttn.sOP("sz", 0), sysinf)
    mel = pyttn.matrix_element(A)

    # set up the tdvp object
    sweep = None
    if not adaptive:
        sweep = pyttn.tdvp(A, h, krylov_dim=12)
    else:
        sweep = pyttn.tdvp(A, h, krylov_dim=12, expansion="subspace")
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold = unoccupied_threshold
        sweep.minimum_unoccupied = nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j

    Sz = np.zeros(nstep + 1)
    maxchi = np.zeros(nstep + 1)
    Sz[0] = np.real(mel(szop, A, trace_ttn))
    maxchi[0] = A.maximum_bond_dimension()

    #perform the first timestep using a logarithmic discretisation of time over this period.  
    #This can be useful to allow for suitable adaptation of weakly occupied single particle 
    #functions through the initial time point.
    run_first_step(sweep, A, h, dt, nstep=5)

    i=1
    #set the values after the first timestep
    Sz[i] =np.real(mel(szop, A, trace_ttn))
    sweep.dt = dt

    for i in range(1,nstep):
        sweep.step(A, h)
        Sz[i + 1] = np.real(mel(szop, A, trace_ttn))
        maxchi[i + 1] = A.maximum_bond_dimension()
        print((i + 1) * dt, Sz[i + 1], maxchi[i + 1], np.real(mel(A, A)))
        sys.stdout.flush()
        if i % 10 == 0:
            output_results(ofname, t, Sz, maxchi)

    output_results(ofname, t, Sz, maxchi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamics of the zero temperature spin boson model with"
    )

    # exponential bath cutoff parameters
    parser.add_argument("alpha", type=float)
    parser.add_argument("--wc", type=float, default=5)
    parser.add_argument("--s", type=float, default=1)

    # number of bath modes
    parser.add_argument("--K", type=int, default=6)

    # maximum bosonic hilbert space dimension
    parser.add_argument("--L", type=int, default=30)
    parser.add_argument("--Lmin", type=int, default=6)

    # mode combination parameters
    parser.add_argument("--nbmax", type=int, default=1)
    parser.add_argument("--nhilbmax", type=int, default=1000)

    # system hamiltonian parameters
    parser.add_argument("--delta", type=float, default=1)
    parser.add_argument("--eps", type=float, default=0)

    # bath inverse temperature
    parser.add_argument("--beta", type=float, default=None)

    # maximum bond dimension
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--degree", type=int, default=1)

    # integration time parameters
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--tmax", type=float, default=10)


    parser.add_argument("--fname", type=str, default=None)
    # the minimum number of unoccupied modes for the dynamics
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-7)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)
    parser.add_argument("--method", type=str, default="heom")

    args = parser.parse_args()

    # output file name
    fname = args.fname
    if fname is None:
        fname = "sbm_"+args.method+".h5"

    nstep = int(args.tmax / args.dt) + 1
    sbm_dynamics(
        args.alpha,
        args.wc,
        args.s,
        args.eps,
        args.delta,
        args.chi,
        args.L,
        args.K,
        args.dt,
        beta=args.beta,
        nstep=nstep,
        ofname=fname,
        nunoccupied=args.nunoccupied,
        spawning_threshold=args.spawning_threshold,
        unoccupied_threshold=args.unoccupied_threshold,
        adaptive=args.subspace,
        degree=args.degree,
        Lmin=args.Lmin,
        use_mode_combination=True,
        nbmax=args.nbmax,
        nhilbmax=args.nhilbmax,
        method = args.method
    )
