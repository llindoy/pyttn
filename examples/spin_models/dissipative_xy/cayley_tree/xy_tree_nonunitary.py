import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
import argparse
import h5py
import copy

import pyttn
from pyttn import oqs, utils
from cayley_helper import get_spin_connectivity, build_topology, get_mode_reordering

from numba import jit

def output_results(ofname, t, res, Rnorm, norm, maxchi, timing):
    h5 = h5py.File(ofname, "w")
    h5.create_dataset("t", data=t)
    for si in range(res.shape[0]):
        h5.create_dataset("Sz" + str(si), data=np.real(res[si, :]))
        h5.create_dataset("rSz" + str(si), data=res[si, :] / Rnorm)
    h5.create_dataset("rnorm", data=Rnorm)
    h5.create_dataset("norm", data=norm)
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


def xychain_dynamics(Nl, alpha, wc, eta, chi, chiS, chiB, L, K, dt, Lmin=None, Ecut = 10, beta=None, nstep=1, ofname="xychain.h5", method="heom",
                     degree=2, adaptive=True, spawning_threshold=2e-4, unoccupied_threshold=1e-4, nunoccupied=0, nbmax=2, nhilbmax=1024):

    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return 2 * np.pi * alpha * w * np.exp(-np.abs(w / wc) ** 2)

    # set up the system information object for a single spin
    # setup the system information object
    sysinf = pyttn.system_modes(1)
    sysinf[0] = [pyttn.spin_mode(2), pyttn.spin_mode(2)]

    expbath=None
    if K != 0:
        # set up the open quantum system bath object
        bath = oqs.BosonicBath(J, beta=beta, wmax=wc * Ecut)
        dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=K, tmax=nstep * dt, Nt=nstep))

        # set up the exp bath object this takes the dk and zk terms.  Truncate the modes and
        # extract the system information object from this.
        expbath = oqs.ExpFitBosonicBath(dk, zk)
        expbath.truncate_modes(utils.EnergyTruncation(Ecut * wc, Lmax=L, Lmin=Lmin))

        # now attempt mode combination on the bath modes
        bsys=None
        if nbmax>1:
            mode_comb = utils.ModeCombination(nhilbmax, nbmax)
            bsys = expbath.system_information(mode_comb)
        else:
            bsys = expbath.system_information()
        
        sysinf = pyttn.combine_systems(sysinf, bsys)


    hiterms, Ns = get_spin_connectivity(Nl, d=3)
    # set the total system information object to just be a single spin
    sysinfo = copy.deepcopy(sysinf)
    site_info = copy.deepcopy(sysinf)
    # and add on the system information objects for the remaining spins
    for i in range(Ns - 1):
        sysinfo = pyttn.combine_systems(sysinfo, sysinf)

    sysinfo.mode_indices = get_mode_reordering(Nl, bsys.nmodes(), site_size=1)

    # set up the total Hamiltonian
    H = pyttn.SOP(sysinfo.nprimitive_modes())

    # set up the interactions for each spin and its bath
    for si in range(Ns):
        skip = si * (site_info.nprimitive_modes())

        # the onsite energy terms
        H += pyttn.sOP("sz", skip) - pyttn.sOP("sz", skip + 1)

        if(bsys.nprimitive_modes() > 0):
            # add on the HEOM bath Hamiltonian
            H = expbath.add_system_bath_generator(H, [pyttn.sOP("sz", skip+0), pyttn.sOP("sz", skip+1)], method=method, bskip = skip+2)

    # now we add on the spin-spin coupling terms
    for ind in hiterms:
        s1 = (ind[0]) * (site_info.nprimitive_modes())
        s2 = (ind[1]) * (site_info.nprimitive_modes())

        H += (1.0 - eta) * (pyttn.sOP("sx", s1) * pyttn.sOP("sx", s2) - pyttn.sOP("sx", s1 + 1) * pyttn.sOP("sx", s2 + 1))
        H += (1.0 + eta) * (pyttn.sOP("sy", s1) * pyttn.sOP("sy", s2) - pyttn.sOP("sy", s1 + 1) * pyttn.sOP("sy", s2 + 1))

    # construct the topology and capacity trees used for constructing
    chi0 = chi
    chiS0 = chiS
    chiB0 = chiB
    if adaptive:
        chi0 = min(chi0, 4)
        chiS0 = min(chiS0, 4)
        chiB0 = min(chiB0, 4)

    topo = build_topology(Nl, sysinfo[0].lhd(), chi0, chiS0, chiB0, L, expbath, degree)
    capacity = build_topology(Nl, sysinfo[0].lhd(), chi, chiS, chiB, L, expbath, degree)

    A = pyttn.ttn(topo, capacity, dtype=np.complex128)

    state = [0 for i in range(Ns * (site_info.nmodes()))]
    state[0] = 3
    print(site_info.nmodes(), len(state), A.nmodes())
    A.set_state(state)

    print("building Hamiltonian")
    h = pyttn.sop_operator(H, A, sysinfo)
    print("Hamiltonian built")
    # set up ttns storing the observable to be measured.  Here these are just system observables
    # so we form a tree with the same topology as A but with all bath bond dimensions set to 1
    obstree = build_topology(Nl, sysinfo[0].lhd(), chi0, chiS0, chiB0, L, expbath, degree)
    trace_ttn = pyttn.ttn(obstree, dtype=np.complex128)

    #now set this up 
    prod_state = []
    for i in range(Ns):
        prod_state += [np.identity(2).flatten()] 
        #add on the product state that projects us onto the bath trace.  This depends on the method used.
        if K != 0:
            prod_state += expbath.identity_product_state(method=method)
    trace_ttn.set_product(prod_state)

    ops = []
    for si in range(Ns):
        skip = si * (site_info.nprimitive_modes() )
        ops.append(pyttn.site_operator(pyttn.sOP("sz", skip), sysinfo))

    mel = pyttn.matrix_element(A)

    # set up the tdvp object
    sweep = None
    if not adaptive:
        sweep = pyttn.tdvp(A, h, krylov_dim=16)
    else:
        sweep = pyttn.tdvp(A, h, krylov_dim=16, expansion="subspace")
        sweep.spawning_threshold = spawning_threshold
        sweep.unoccupied_threshold = unoccupied_threshold
        sweep.minimum_unoccupied = nunoccupied

    sweep.dt = dt
    sweep.coefficient = -1.0j

    res = np.zeros((Ns, nstep + 1), dtype=np.complex128)
    Rnorm = np.ones(nstep + 1, dtype=np.complex128)
    maxchi = np.zeros(nstep + 1)
    norm = np.zeros(nstep + 1)

    for i in range(Ns):
        res[i, 0] = mel(ops[i], A, trace_ttn)
    maxchi[0] = A.maximum_bond_dimension()

    t1 = time.time()
    # perform the dynamics
    renorm = mel(trace_ttn, A)
    i = 0
    print((i) * dt, res[0, i], np.real(renorm), maxchi[i], np.real(mel(A, A)))
    norm[i] = np.real(mel(A, A))

    # perform the first timestep using a logarithmic discretisation of time over this period.
    # This can be useful to allow for suitable adaptation of weakly occupied single particle
    # functions through the initial time point.
    run_first_step(sweep, A, h, dt, nstep=5)
    sweep.dt = dt

    #evaluate all properties
    for si in range(Ns):
        res[si, 1] = mel(ops[si], A, trace_ttn)
    maxchi[1] = A.maximum_bond_dimension()
    norm[i] = np.real(mel(A, A))
    Rnorm[1] = 1 / renorm

    #print result
    i=1
    print((i) * dt, res[0, i], renorm, maxchi[i], np.real(mel(A, A)))

    t = np.arange(nstep + 1) * dt
    for i in range(1, nstep):
        #perform timestep
        sweep.step(A, h)

        #evaluate all properties
        renorm = mel(trace_ttn, A)
        for si in range(Ns):
            res[si, i + 1] = mel(ops[si], A, trace_ttn)
        maxchi[i + 1] = A.maximum_bond_dimension()
        norm[i + 1] = np.real(mel(A, A))
        Rnorm[i + 1] = 1 / renorm
        
        #print result
        print((i + 1) * dt, res[0, i + 1], np.real(renorm), maxchi[i + 1], np.real(mel(A, A)))

        t2 = time.time()
        output_results(ofname, t, res, Rnorm, norm, maxchi, t2-t1)

        #if i % 10 ==0 :
        #    import matplotlib.pyplot as plt
        #    utils.visualise_tree(A, prog="twopi", bond_prop="bond dimension")
        #    plt.show()
    t2 = time.time()
    output_results(ofname, t, res, Rnorm, norm, maxchi, t2-t1)


def run_from_inputs():
    parser = argparse.ArgumentParser(
        description="Dynamics of the zero temperature spin boson model with"
    )

    # number of spins in the system
    parser.add_argument("--Nl", type=int, default=3)

    # exponential bath cutoff parameters
    parser.add_argument("--alpha", type=float, default=0.32)
    parser.add_argument("--wc", type=float, default=4)

    # number of bath modes
    parser.add_argument("--K", type=int, default=4)

    # maximum bosonic hilbert space dimension
    parser.add_argument("--L", type=int, default=25)
    parser.add_argument("--Lmin", type=int, default=4)
    parser.add_argument("--ecut", type=float, default=10)

    # mode combination parameters
    parser.add_argument("--nbmax", type=int, default=2)
    parser.add_argument("--nhilbmax", type=int, default=1000)

    # system hamiltonian parameters
    parser.add_argument("--eta", type=float, default=0.04)

    # bath inverse temperature
    parser.add_argument("--beta", type=float, default=None)

    # maximum bond dimension
    parser.add_argument("--chi", type=int, default=16)
    parser.add_argument("--chiS", type=int, default=12)
    parser.add_argument("--chiB", type=int, default=8)
    parser.add_argument("--degree", type=int, default=1)

    # integration time parameters
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--tmax", type=float, default=10)

    parser.add_argument("--method", type=str, default="heom")

    # output file name
    parser.add_argument("--fname", type=str, default=None)

    # the minimum number of unoccupied modes for the dynamics
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=1)
    parser.add_argument("--spawning_threshold", type=float, default=5e-7)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)

    args = parser.parse_args()

    # output file name
    fname = args.fname
    if fname is None:
        fname = "xychain_"+args.method+".h5"

    nstep = int(args.tmax / args.dt) + 1

    xychain_dynamics(args.Nl, args.alpha, args.wc, args.eta, args.chi, args.chiS, args.chiB, args.L, args.K, args.dt, beta=args.beta, nstep=nstep, 
                     Ecut=args.ecut, method=args.method, ofname=fname, nunoccupied=args.nunoccupied, spawning_threshold=args.spawning_threshold, 
                     unoccupied_threshold=args.unoccupied_threshold, adaptive=args.subspace, degree=args.degree, Lmin=args.Lmin, 
                     nbmax=args.nbmax, nhilbmax=args.nhilbmax)


if __name__ == "__main__":
    run_from_inputs()
