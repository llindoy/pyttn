import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import numpy as np
import time
import h5py
import copy

import pyttn
from pyttn import oqs, utils
from numba import jit


def build_topology(Ns, ds, chi, chiS, chiB, nbose, discbath, degree):
    lchi = [chiS for i in range(Ns)]
    topo = pyttn.ntreeBuilder.mps_tree(lchi, chi)

    leaf_indices = topo.leaf_indices()
    for li in leaf_indices:
        topo.at(li).insert(ds)

        #now if the discrete bath object has been created we add the bath modes to the tre
        if discbath is not None:
            _ = discbath.add_bath_tree(topo.at(li), degree, chiB, min(chiB, nbose))

    pyttn.ntreeBuilder.sanitise(topo)
    return topo


def output_results(ofname, t, res, maxchi, timing):
    h5 = h5py.File(ofname, "w")
    h5.create_dataset("t", data=t)
    for si in range(res.shape[0]):
        h5.create_dataset("Sz" + str(si), data=np.real(res[si, :]))
    h5.create_dataset("maxchi", data=maxchi)
    h5.create_dataset("time", data=np.array([timing]))
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

def xychain_dynamics(Ns, Nb, alpha, wc, eta, chi, chiS, chiB, nbose, dt, nbose_min=4, beta=None, 
                     nstep=1, Nw=4.0, Ecut=10.0, geom="ipchain", ofname="xychain.h5", degree=2, 
                     adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, nbmax=2, nhilbmax=1024):
    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return 2 * np.pi * alpha * w * np.exp(-np.abs(w / wc) ** 2)

    # setup the system information object.  Additionally work out the local hilbert space dimensions os that we can set up the system mode informmation
    sysinf = pyttn.system_modes(1)
    sysinf[0] = pyttn.spin_mode(2)

    discbath=None
    bsys=None

    if Nb != 0:
        # set up the open quantum system bath object
        bath = oqs.BosonicBath(J, beta=beta)

        # discretise the bath correleation function using the orthonormal polynomial based cutoff
        g, w = bath.discretise(oqs.OrthopolDiscretisation(Nb, bath.find_wmin(Nw * wc), Nw * wc))

        # set up the discretised bath object
        discbath = oqs.DiscreteBosonicBath(g, w)

        # truncate the system bath modes
        discbath.truncate_modes(utils.EnergyTruncation(Ecut * wc, Lmax=nbose, Lmin=nbose_min))

        # set up the mode combination informatino
        if nbmax > 1:
            mode_comb = utils.ModeCombination(nbmax, nhilbmax)
            bsys = discbath.system_information(mode_comb)
        else:
            bsys = discbath.system_information()

    # set up the total Hamiltonian
    N = Nb + 1
    H = pyttn.SOP(Ns * N)

    # add on the system part of the system bath Hamiltonian
    for si in range(Ns):
        skip = si * (Nb + 1)
        H += pyttn.sOP("sz", skip)

        if si + 1 != Ns:
            skip2 = (si + 1) * (Nb + 1)
            H += (1 - eta) * pyttn.sOP("sx", skip) * pyttn.sOP("sx", skip2) + (1 + eta) * pyttn.sOP("sy", skip) * pyttn.sOP("sy", skip2)

        if Nb != 0:
            # add on the bath and system bath contributions of the bath hamiltonian
            H = discbath.add_system_bath_hamiltonian(H, pyttn.sOP("sz", 0), geom=geom, bskip=skip+1)

    print("hamiltonian string setup")

    #set up the total system information object
    if Nb != 0:
        sysinf = pyttn.combine_systems(sysinf, bsys)

    site_info = copy.deepcopy(sysinf)
    sysinfo = copy.deepcopy(sysinf)
    # and add on the system information objects for the remaining spins
    for i in range(Ns - 1):
        sysinfo = pyttn.combine_systems(sysinfo, sysinf)

    sysinf = sysinfo

    print("system built")

    # construct the topology and capacity trees used for constructing
    chi0 = chi
    chiS0 = chiS
    chiB0 = chiB
    if adaptive:
        chi0 = min(8, chi0)
        chiS0 = min(8, chiS0)
        chiB0 = min(8, chiB0)

    # now build the topology and capacity arrays
    topo = build_topology(Ns, 2, chi0, chiS0, chiB0, nbose, discbath, degree)
    capacity = build_topology(Ns, 2, chi, chiS, chiB, nbose, discbath, degree)

    print("topology built")

    # construct and initialise the ttn wavefunction
    A = pyttn.ttn(topo, capacity, dtype=np.complex128)
    #set up the spin excitation on just the central site of the chain
    state = [0 for i in range(Ns * (site_info.nmodes()))]
    state[(Ns - 1) // 2 * (site_info.nmodes())] = 1
    A.set_state(state)

    print("psi0 built")
    # set up the Hamiltonian as a sop object
    h = pyttn.sop_operator(H, A, sysinf)
    print("H built")

    # construct objects need for evaluating observables
    mel = pyttn.matrix_element(A)

    # set up the observable to measure
    ops = []
    for si in range(Ns):
        skip = si * (site_info.nprimitive_modes())
        ops.append(pyttn.site_operator(pyttn.sOP("sz", skip), sysinf))

    # set up tdvp sweeping algorithm parameters
    sweep = None
    if not adaptive:
        sweep = pyttn.tdvp(A, h, krylov_dim=12)
    else:
        sweep = pyttn.tdvp(A, h, krylov_dim=12, subspace_neigs=6, expansion="subspace")
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold = unoccupied_threshold
        sweep.minimum_unoccupied = nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j

    t = np.arange(nstep + 1) * dt
    t1 = time.time()
    # run dynamics and measure properties storing them in a file
    res = np.zeros((Ns, nstep + 1), dtype=np.complex128)
    maxchi = np.zeros(nstep + 1)
    for i in range(Ns):
        res[i, 0] = mel(ops[i], A)
    maxchi[0] = A.maximum_bond_dimension()

    # perform the first timestep using a logarithmic discretisation of time over this period.
    # This can be useful to allow for suitable adaptation of weakly occupied single particle
    # functions through the initial time point.
    run_first_step(sweep, A, h, dt, nstep=5)

    # set the values after the first timestep
    for si in range(Ns):
        res[si, 1] = mel(ops[si], A)
    maxchi[1] = A.maximum_bond_dimension()
    sweep.dt = dt

    # now perform standard time stepping
    for i in range(1, nstep):
        sweep.step(A, h)
        for si in range(Ns):
            res[si, i + 1] = mel(ops[si], A)
        maxchi[i + 1] = A.maximum_bond_dimension()
        print(i, res[(Ns - 1) // 2, i + 1], A.maximum_bond_dimension())

        t2 = time.time()
        # outputting results to files every 10 steps
        if i % 1 == 0:
            output_results(ofname, t, res, maxchi, t2-t1)


    t2 = time.time()
    # and finally dump everything to file at the end of the simulation
    output_results(ofname, t, res, maxchi, t2-t1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamics of the zero temperature spin boson model with"
    )

    parser.add_argument("--N", type=int, default=40)
    # number of spins in the system
    parser.add_argument("--Ns", type=int, default=21)

    # exponential bath cutoff parameters
    parser.add_argument("--alpha", type=float, default=0.32)
    parser.add_argument("--wc", type=float, default=4)

    # number of bath modes
    parser.add_argument("--geom", type=str, default="chain")

    # maximum bosonic hilbert space dimension
    parser.add_argument("--nbose", type=int, default=30)
    parser.add_argument("--nbose_min", type=int, default=6)
    parser.add_argument("--ecut", type=float, default=10)

    # mode combination parameters
    parser.add_argument("--nbmax", type=int, default=4)
    parser.add_argument("--nhilbmax", type=int, default=1000)

    # system hamiltonian parameters
    parser.add_argument("--eta", type=float, default=0.04)

    # bath inverse temperature
    parser.add_argument("--beta", type=float, default=None)

    # maximum bond dimension
    parser.add_argument("--chi", type=int, default=36)
    parser.add_argument("--chiS", type=int, default=24)
    parser.add_argument("--chiB", type=int, default=36)
    parser.add_argument("--degree", type=int, default=1)

    # integration time parameters
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tmax", type=float, default=5)

    # output file name
    parser.add_argument("--fname", type=str, default="xychain.h5")

    # the minimum number of unoccupied modes for the dynamics
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-5)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)

    args = parser.parse_args()

    nstep = int(args.tmax / args.dt) + 1

    xychain_dynamics(args.Ns, args.N, args.alpha, args.wc, args.eta, args.chi, args.chiS, args.chiB, args.nbose, args.dt, 
                     beta=args.beta, nstep=nstep, Ecut=args.ecut, geom=args.geom, ofname=args.fname, nunoccupied=args.nunoccupied,
                     spawning_threshold=args.spawning_threshold, unoccupied_threshold=args.unoccupied_threshold, adaptive=args.subspace,
                     degree=args.degree, nbose_min=args.nbose_min, nbmax=args.nbmax, nhilbmax=args.nhilbmax)
