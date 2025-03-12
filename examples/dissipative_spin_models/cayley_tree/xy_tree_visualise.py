import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
import sys
import h5py
import scipy
import copy

sys.path.append("../../../")
import pyttn
from pyttn import *
from pyttn import oqs, utils
from cayley_helper import get_spin_connectivity, build_topology, build_system_topology
from pyttn.utils import visualise_tree

import matplotlib.pyplot as plt
from numba import jit


def observable_tree(Ns, obstree, op, b_mode_dims):
    Opttn = ttn(obstree, dtype=np.complex128)
    # setup the Sz tree state

    prod_state = []
    for i in range(Ns):
        prod_state.append(op.flatten())
        for i in range(len(b_mode_dims)):
            state_vec = np.zeros(b_mode_dims[i], dtype=np.complex128)
            state_vec[0] = 1.0
            prod_state.append(state_vec)

    Opttn.set_product(prod_state)
    return Opttn


def xychain_dynamics(
    Nl,
    alpha,
    wc,
    eta,
    chi,
    chiS,
    chiB,
    L,
    K,
    dt,
    Lmin=None,
    beta=None,
    nstep=1,
    ofname="xychain.h5",
    degree=2,
    adaptive=True,
    spawning_threshold=2e-4,
    unoccupied_threshold=1e-4,
    nunoccupied=0,
    use_mode_combination=True,
    nbmax=2,
    nhilbmax=1024,
):
    t = np.arange(nstep + 1) * dt

    # setup the function for evaluating the exponential cutoff spectral density
    @jit(nopython=True)
    def J(w):
        return 2 * np.pi * alpha * w * np.exp(-np.abs(w / wc) ** 2)

    # set up the open quantum system bath object
    bath = oqs.BosonicBath(J, beta=beta, wmax=wc * 100)
    dk, zk = bath.expfit(oqs.ESPRITDecomposition(K=K, tmax=nstep * dt, Nt=nstep))

    # set up the exp bath object this takes the dk and zk terms.  Truncate the modes and
    # extract the system information object from this.
    expbath = oqs.ExpFitBosonicBath(dk, zk)
    expbath.truncate_modes(utils.EnergyTruncation(15 * wc, Lmax=L, Lmin=Lmin))
    bsys = expbath.system_information()

    dk = expbath.dk
    zk = expbath.zk

    hiterms, Ns = get_spin_connectivity(Nl, d=3)

    Nb = bsys.nprimitive_modes()
    N = (Nb + 2) * Ns

    # set up the system information object for a single spin
    # setup the system information object
    sysinf = system_modes(1)
    sysinf[0] = [spin_mode(2), spin_mode(2)]

    # now attempt mode combination on the bath modes
    if use_mode_combination:
        mode_comb = utils.ModeCombination(nhilbmax, nbmax)
        bsys = mode_comb(bsys)

    # extract the bath mode dimensions
    b_mode_dims = np.zeros(len(bsys), dtype=int)
    for i in range(len(bsys)):
        b_mode_dims[i] = bsys[i].lhd()

    sysinf = combine_systems(sysinf, bsys)

    # set the total system information object to just be a single spin
    sysinfo = copy.deepcopy(sysinf)

    # and add on the system information objects for the remaining spins
    for i in range(Ns - 1):
        sysinfo = combine_systems(sysinfo, sysinf)

    # set up the total Hamiltonian
    H = SOP(sysinfo.nprimitive_modes())
    zktot = np.sum(zk)

    # set up the interactions for each spin and its bath
    for si in range(Ns):
        skip = si * (Nb + 2)

        # the onsite energy terms
        H += sOP("sz", skip) - sOP("sz", skip + 1)

        # add on the HEOM bath Hamiltonian
        for i in range(Nb):
            bind = skip + i + 2
            H += -1.0j * zk[i] * sOP("n", bind)
            H += (
                complex(dk[i])
                * (sOP("sz", skip + 0) - sOP("sz", skip + 1))
                * sOP("a", bind)
            )
            if i % 2 == 0:
                H += complex(dk[i]) * sOP("sz", skip + 0) * sOP("adag", bind)
            else:
                H += -complex(dk[i]) * sOP("sz", skip + 1) * sOP("adag", bind)

    # now we add on the spin-spin coupling terms
    for ind in hiterms:
        s1 = (ind[0]) * (Nb + 2)
        s2 = (ind[1]) * (Nb + 2)

        H += (1.0 - eta) * (
            sOP("sx", s1) * sOP("sx", s2) - sOP("sx", s1 + 1) * sOP("sx", s2 + 1)
        )
        H += (1.0 + eta) * (
            sOP("sy", s1) * sOP("sy", s2) - sOP("sy", s1 + 1) * sOP("sy", s2 + 1)
        )

    # construct the topology and capacity trees used for constructing
    chi0 = chi
    chiS0 = chiS
    chiB0 = chiB
    if adaptive:
        chi0 = 16
        chiS0 = 16
        chiB0 = 16
    chi0 = min(chi0, chi)
    chiS0 = min(chiS0, chiS)
    chiB0 = min(chiB0, chiB)

    topo = build_system_topology(
        Nl, sysinfo[0].lhd(), chi0, chiS0, chiB0, L, b_mode_dims, degree
    )
    capacity = build_topology(
        Nl, sysinfo[0].lhd(), chi, chiS, chiB, L, b_mode_dims, degree
    )

    visualise_tree(topo, prog="twopi", add_labels=False)
    plt.show()
    exit()


import argparse


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
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--Lmin", type=int, default=4)

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
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--tmax", type=float, default=10)

    # output file name
    parser.add_argument("--fname", type=str, default="xytree_pm_8.h5")

    # the minimum number of unoccupied modes for the dynamics
    parser.add_argument("--subspace", type=bool, default=True)
    parser.add_argument("--nunoccupied", type=int, default=0)
    parser.add_argument("--spawning_threshold", type=float, default=1e-6)
    parser.add_argument("--unoccupied_threshold", type=float, default=1e-4)

    args = parser.parse_args()

    xychain_dynamics(
        args.Nl,
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
        ofname=args.fname,
        nunoccupied=args.nunoccupied,
        spawning_threshold=args.spawning_threshold,
        unoccupied_threshold=args.unoccupied_threshold,
        adaptive=args.subspace,
        degree=args.degree,
        Lmin=args.Lmin,
        use_mode_combination=True,
        nbmax=args.nbmax,
        nhilbmax=args.nhilbmax,
    )


def main():
    Nl = 3
    alpha = 0.32
    wc = 4
    eta = 0.04
    L = 20
    K = 4
    dt = 0.05
    beta = None
    tmax = 10
    nstep = int(tmax / dt) + 1
    nunoccupied = 0
    spawning_threshold = 1e-6
    unoccupied_threshold = 1e-4
    subspace = True
    degree = 2
    Lmin = 4
    use_mode_combination = True
    nbmax = 2
    nhilbmax = 1000

    chiSs = [4, 8, 12, 16, 20, 24, 32]

    for chiS in chiSs:
        chi = 32
        chiB = int(1.5 * chiS)
        fname = "xytree_heom_" + str(chi) + "_" + str(chiS) + "_" + str(chiB) + ".h5"

        xychain_dynamics(
            Nl,
            alpha,
            wc,
            eta,
            chi,
            chiS,
            chiB,
            L,
            K,
            dt,
            beta=beta,
            nstep=nstep,
            ofname=fname,
            nunoccupied=nunoccupied,
            spawning_threshold=spawning_threshold,
            unoccupied_threshold=unoccupied_threshold,
            adaptive=subspace,
            degree=degree,
            Lmin=Lmin,
            use_mode_combination=True,
            nbmax=nbmax,
            nhilbmax=nhilbmax,
        )


if __name__ == "__main__":
    main()
