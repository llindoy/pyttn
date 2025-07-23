# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import copy
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from cayley_helper import build_system_topology, get_spin_connectivity
from numba import jit

import pyttn
from pyttn import oqs, utils
from pyttn.utils import visualise_tree


def observable_tree(Ns, obstree, op, b_mode_dims):
    Opttn = pyttn.ttn(obstree, dtype=np.complex128)
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
    degree=2,
    adaptive=True,
    use_mode_combination=True,
    nbmax=2,
    nhilbmax=1024,
):

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

    # set up the system information object for a single spin
    # setup the system information object
    sysinf = pyttn.system_modes(1)
    sysinf[0] = [pyttn.spin_mode(2), pyttn.spin_mode(2)]

    # now attempt mode combination on the bath modes
    if use_mode_combination:
        mode_comb = utils.ModeCombination(nhilbmax, nbmax)
        bsys = mode_comb(bsys)

    # extract the bath mode dimensions
    b_mode_dims = np.zeros(len(bsys), dtype=int)
    for i in range(len(bsys)):
        b_mode_dims[i] = bsys[i].lhd()

    sysinf = pyttn.combine_systems(sysinf, bsys)

    # set the total system information object to just be a single spin
    sysinfo = copy.deepcopy(sysinf)

    # and add on the system information objects for the remaining spins
    for _ in range(Ns - 1):
        sysinfo = pyttn.combine_systems(sysinfo, sysinf)

    # set up the total Hamiltonian
    H = pyttn.SOP(sysinfo.nprimitive_modes())

    # set up the interactions for each spin and its bath
    for si in range(Ns):
        skip = si * (Nb + 2)

        # the onsite energy terms
        H += pyttn.sOP("sz", skip) - pyttn.sOP("sz", skip + 1)

        # add on the HEOM bath Hamiltonian
        for i in range(Nb):
            bind = skip + i + 2
            H += -1.0j * zk[i] * pyttn.sOP("n", bind)
            H += (
                complex(dk[i])
                * (pyttn.sOP("sz", skip + 0) - pyttn.sOP("sz", skip + 1))
                * pyttn.sOP("a", bind)
            )
            if i % 2 == 0:
                H += complex(dk[i]) * pyttn.sOP("sz", skip + 0) * pyttn.sOP("adag", bind)
            else:
                H += -complex(dk[i]) * pyttn.sOP("sz", skip + 1) * pyttn.sOP("adag", bind)

    # now we add on the spin-spin coupling terms
    for ind in hiterms:
        s1 = (ind[0]) * (Nb + 2)
        s2 = (ind[1]) * (Nb + 2)

        H += (1.0 - eta) * (
            pyttn.sOP("sx", s1) * pyttn.sOP("sx", s2) - pyttn.sOP("sx", s1 + 1) * pyttn.sOP("sx", s2 + 1)
        )
        H += (1.0 + eta) * (
            pyttn.sOP("sy", s1) * pyttn.sOP("sy", s2) - pyttn.sOP("sy", s1 + 1) * pyttn.sOP("sy", s2 + 1)
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

    visualise_tree(topo, prog="twopi", add_labels=False)
    plt.show()
    return

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
