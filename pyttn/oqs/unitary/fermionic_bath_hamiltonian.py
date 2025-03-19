# This files is part of the pyTTN package.
# (C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np
from .chain_map import chain_map
from pyttn import fOP


def __generate_binds(binds, bskip, Nb):
    if not isinstance(binds, np.ndarray):
        if binds is None:
            binds = [i + bskip for i in range(Nb)]
    return binds


# setup the star Hamiltonian for the spin boson model
def add_fermionic_star_bath_hamiltonian(H, Sp, Sm, g, w, binds=None, bskip=1):
    Nb = g.shape[0]
    binds = __generate_binds(binds, bskip, Nb)

    for i in range(Nb):
        H += g[i] * Sp * fOP("c", binds[i])
        H += g[i] * fOP("cdag", binds[i]) * Sm
        H += w[i] * fOP("n", binds[i])

    return H


# setup the chain hamiltonian for the spin boson model - this is the tedopa method
def add_fermionic_chain_bath_hamiltonian(H, Sp, Sm, t, e, binds=None, bskip=1):
    Nb = e.shape[0]
    binds = __generate_binds(binds, bskip, Nb)

    for i in range(Nb):
        if i == 0:
            H += t[i] * Sp * fOP("c", binds[i])
            H += t[i] * fOP("cdag", binds[i]) * Sm
        else:
            H += t[i] * fOP("cdag", binds[i - 1]) * fOP("c", binds[i])
            H += t[i] * fOP("cdag", binds[i]) * fOP("c", binds[i - 1])
        H += e[i] * fOP("n", binds[i])

    return H


def add_fermionic_bath_hamiltonian(
    H, Sp, Sm, g, w, binds=None, geom="star", bskip=1, return_frequencies=False
):
    if geom == "star":
        if not return_frequencies:
            return add_fermionic_star_bath_hamiltonian(
                H, Sp, Sm, g, w, binds=binds, bskip=bskip
            )
        else:
            return add_fermionic_star_bath_hamiltonian(
                H, Sp, Sm, g, w, binds=binds, bskip=bskip
            ), w

    elif geom == "chain":
        t, e = chain_map(g, w)
        if not return_frequencies:
            return add_fermionic_chain_bath_hamiltonian(
                H, Sp, Sm, t, e, binds=binds, bskip=bskip
            )
        else:
            return add_fermionic_chain_bath_hamiltonian(
                H, Sp, Sm, t, e, binds=binds, bskip=bskip
            ), e
    else:
        raise RuntimeError("Cannot add bath Hamiltonian geometry not recognised.")
