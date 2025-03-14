import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
import h5py
import copy
import argparse


import pyttn
from pyttn import oqs, utils
from numba import jit


def build_topology(Ns, ds, chi, chiS, chiB, nbose, expbath, degree):
    lchi = [chiS for i in range(Ns)]
    topo = pyttn.ntreeBuilder.mps_tree(lchi, chi)

    leaf_indices = topo.leaf_indices()
    for li in leaf_indices:
        topo.at(li).insert(ds)

        #now if the discrete bath object has been created we add the bath modes to the tre
        if expbath is not None:
            _ = expbath.add_bath_tree(topo.at(li), degree, chiB, min(chiB, nbose))

    pyttn.ntreeBuilder.sanitise(topo)
    return topo


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


def xychain_dynamics(Ns, alpha, wc, eta, chi, chiS, chiB, L, K, dt, Lmin=None, Ecut=10, beta=None, nstep=1, 
                     ofname="xychain.h5", degree=2, adaptive=True, spawning_threshold=2e-4, 
                     unoccupied_threshold=1e-4, nunoccupied=0, nbmax=2, nhilbmax=1024, method="heom"):
    t = np.arange(nstep + 1) * dt

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
        bsys = expbath.system_information()

        dk = expbath.dk
        zk = expbath.zk

        # now attempt mode combination on the bath modes
        if nbmax>1:
            mode_comb = utils.ModeCombination(nhilbmax, nbmax)
            bsys = mode_comb(bsys)

        sysinf = pyttn.combine_systems(sysinf, bsys)


    # set the total system information object to just be a single spin
    sysinfo = copy.deepcopy(sysinf)
    site_info = copy.deepcopy(sysinf)
    # and add on the system information objects for the remaining spins
    for i in range(Ns - 1):
        sysinfo = pyttn.combine_systems(sysinfo, sysinf)

    # set up the total Hamiltonian
    H = pyttn.SOP(sysinfo.nprimitive_modes())

    # set up the interactions for each spin and its bath
    for si in range(Ns):
        skip = si * (site_info.nprimitive_modes())

        # the onsite energy terms
        H += pyttn.sOP("sz", skip) - pyttn.sOP("sz", skip + 1)

        # add on the HEOM bath Hamiltonian
        H = expbath.add_system_bath_generator(H, [pyttn.sOP("sz", skip+0), pyttn.sOP("sz", skip+1)], method=method, bskip = skip+2)

    # now we add on the spin-spin coupling terms
    for si in range(Ns - 1):
        s1 = si * (site_info.nprimitive_modes())
        s2 = (si + 1) * (site_info.nprimitive_modes())

        H += (1.0 - eta) * (pyttn.sOP("sx", s1) * pyttn.sOP("sx", s2) - pyttn.sOP("sx", s1 + 1) * pyttn.sOP("sx", s2 + 1))
        H += (1.0 + eta) * (pyttn.sOP("sy", s1) * pyttn.sOP("sy", s2) - pyttn.sOP("sy", s1 + 1) * pyttn.sOP("sy", s2 + 1))

    # construct the topology and capacity trees used for constructing
    chi0 = chi
    chiS0 = chiS
    chiB0 = chiB
    if adaptive:
        chi0 = min(16, chi)
        chiS0 = min(16, chiS)
        chiB0 = min(16, chiB)

    topo = build_topology(Ns, sysinfo[0].lhd(), chi0, chiS0, chiB0, L, expbath, degree)
    capacity = build_topology(Ns, sysinfo[0].lhd(), chi, chiS, chiB, L, expbath, degree)

    A = pyttn.ttn(topo, capacity, dtype=np.complex128)

    state = [0 for i in range(Ns * (site_info.nmodes()))]
    state[(Ns - 1) // 2 * (site_info.nmodes())] = 3
    A.set_state(state)

    h = pyttn.sop_operator(H, A, sysinfo)
    # set up ttns storing the observable to be measured.  Here these are just system observables
    # so we form a tree with the same topology as A but with all bath bond dimensions set to 1
    obstree = build_topology(Ns, sysinfo[0].lhd(), 1, 1, 1, L, expbath, degree)
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
        skip = si * (site_info.nprimitive_modes())
        ops.append(pyttn.site_operator(pyttn.sOP("sz", skip), sysinfo))

    mel = pyttn.matrix_element(A)

    # set up the tdvp object
    sweep = None
    if not adaptive:
        sweep = pyttn.tdvp(A, h, krylov_dim=16)
    else:
        sweep = pyttn.tdvp(A, h, krylov_dim=16, expansion="subspace", subspace_neigs=6, subspace_krylov_dim=12)
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
    print((i) * dt, res[Ns // 2, i], renorm, maxchi[i], np.real(mel(A, A)))
    norm[i] = np.real(mel(A, A))

    #perform the first timestep using a logarithmic discretisation of time over this period.  
    #This can be useful to allow for suitable adaptation of weakly occupied single particle 
    #functions through the initial time point.
    run_first_step(sweep, A, h, dt, nstep=5)
    sweep.dt = dt

    for si in range(Ns):
        res[si, 1] = mel(ops[si], A, trace_ttn)
    maxchi[1] = A.maximum_bond_dimension()
    norm[1] = np.real(mel(A, A))
    Rnorm[1] = 1 / renorm
    print((i) * dt, res[Ns // 2, i], renorm, maxchi[i], np.real(mel(A, A)))

    t=(np.arange(nstep + 1) * dt)
    for i in range(1, nstep):
        print(mel(h, A, trace_ttn))
        sweep.step(A, h)
        renorm = mel(trace_ttn, A)
        for si in range(Ns):
            res[si, i + 1] = mel(ops[si], A, trace_ttn)
        maxchi[i + 1] = A.maximum_bond_dimension()
        norm[i + 1] = np.real(mel(A, A))
        Rnorm[i + 1] = renorm

        print((i + 1) * dt, res[Ns // 2, i + 1], renorm, maxchi[i + 1], np.real(mel(A, A)))

        t2 = time.time()
        output_results(ofname, t, res, Rnorm, norm, maxchi, t2-t1)


    t2 = time.time()
    output_results(ofname, t, res, Rnorm, norm, maxchi, t2-t1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamics of the zero temperature spin boson model with"
    )

    # number of spins in the system
    parser.add_argument("--Ns", type=int, default=21)

    # exponential bath cutoff parameters
    parser.add_argument("--alpha", type=float, default=0.32)
    parser.add_argument("--wc", type=float, default=4)

    # number of bath modes
    parser.add_argument("--K", type=int, default=6)

    # maximum bosonic hilbert space dimension
    parser.add_argument("--L", type=int, default=30)
    parser.add_argument("--Lmin", type=int, default=6)

    # mode combination parameters
    parser.add_argument("--nbmax", type=int, default=2)
    parser.add_argument("--nhilbmax", type=int, default=1000)

    # system hamiltonian parameters
    parser.add_argument("--eta", type=float, default=0.02)

    # bath inverse temperature
    parser.add_argument("--beta", type=float, default=None)

    # maximum bond dimension
    parser.add_argument("--chi", type=int, default=32)
    parser.add_argument("--chiS", type=int, default=32)
    parser.add_argument("--chiB", type=int, default=32)
    parser.add_argument("--degree", type=int, default=1)

    # integration time parameters
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--tmax", type=float, default=40)
    parser.add_argument("--fname", type=str, default=None)

    # output file name
    parser.add_argument("--method", type=str, default="heom")

    # the minimum number of unoccupied modes for the dynamics
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-6)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)

    args = parser.parse_args()

    # output file name
    fname = args.fname
    if fname is None:
        fname = "xychain_"+args.method+".h5"

    nstep = int(args.tmax / args.dt) + 1

    xychain_dynamics(
        args.Ns,
        args.alpha,
        args.wc,
        args.eta,
        args.chi,
        args.chiS,
        args.chiB,
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
        nbmax=args.nbmax,
        nhilbmax=args.nhilbmax,
        method=args.method
    )
