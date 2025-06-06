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

import numpy as np
import copy

from .chain_map import chain_map
from pyttn import sOP, coeff

# functions for setting up the bath hamiltonian in several different geometries.
def __generate_binds(binds, bskip, Nb):
    if not isinstance(binds, np.ndarray):
        if binds is None:
            binds = [i + bskip for i in range(Nb)]
    return binds


# setup the star Hamiltonian for the spin boson model
def add_star_bath_hamiltonian(H, Sp, g, w, Sm=None, binds=None, bskip=1):
    Nb = g.shape[0]
    binds = __generate_binds(binds, bskip, Nb)

    for i in range(Nb):
        if Sm is None:
            H += np.sqrt(2.0) * g[i] * Sp * sOP("q", binds[i])
        else:
            H += g[i] * Sp * sOP("a", binds[i])
            H += g[i] * Sm * sOP("adag", binds[i])
        H += w[i] * sOP("n", binds[i])

    return H


# setup the chain hamiltonian for the spin boson model - this is the tedopa method
def add_chain_bath_hamiltonian(H, Sp, t, e, Sm=None, binds=None, bskip=1):
    Nb = e.shape[0]
    binds = __generate_binds(binds, bskip, Nb)

    for i in range(Nb):
        if i == 0:
            if Sm is None:
                H += np.sqrt(2.0) * t[i] * Sp * sOP("q", binds[i])
            else:
                H += t[i] * Sp * sOP("a", binds[i])
                H += t[i] * Sm * sOP("adag", binds[i])
        else:
            H += t[i] * sOP("adag", binds[i - 1]) * sOP("a", binds[i])
            H += t[i] * sOP("a", binds[i - 1]) * sOP("adag", binds[i])
        H += e[i] * sOP("n", binds[i])

    return H


# setup the chain hamiltonian for the spin boson model - that is this implements the method described in Nuomin, Beratan, Zhang, Phys. Rev. A 105, 032406
def add_ipchain_bath_hamiltonian(H, Sp, t0, w, P, Sm=None, binds=None, bskip=1):
    Nb = w.shape[0]
    binds = __generate_binds(binds, bskip, Nb)

    class func_class:
        def __init__(self, i, t0, e0, U0, conj=False):
            self.i = copy.deepcopy(i)
            self.conj = conj
            self.t0 = copy.deepcopy(t0)
            self.e = copy.deepcopy(e0)
            self.U = copy.deepcopy(U0)

        def __call__(self, ti):
            val = (
                self.t0
                * np.conj(self.U[:, 0])
                @ (np.exp(-1.0j * ti * self.e) * self.U[:, self.i])
            )

            if self.conj:
                val = np.conj(val)

            return val

    for i in range(Nb):
        if Sm is None:
            H += coeff(func_class(i, t0, w, P, conj=False)) * Sp * sOP("a", binds[i])
            H += coeff(func_class(i, t0, w, P, conj=True)) * Sp * sOP("adag", binds[i])
        else:
            H += coeff(func_class(i, t0, w, P, conj=False)) * Sp * sOP("a", binds[i])
            H += coeff(func_class(i, t0, w, P, conj=True)) * Sm * sOP("adag", binds[i])

    return H


def add_bosonic_bath_hamiltonian(
    H, Sp, g, w, Sm=None, binds=None, bskip=1, geom="star", return_frequencies=False
):
    if geom == "star":
        if not return_frequencies:
            return add_star_bath_hamiltonian(
                H, Sp, g, w, Sm=Sm, binds=binds, bskip=bskip
            )
        else:
            return (
                add_star_bath_hamiltonian(H, Sp, g, w, Sm=Sm, binds=binds, bskip=bskip),
                w,
            )
    elif geom == "chain":
        t, e = chain_map(g, w)
        if not return_frequencies:
            return add_chain_bath_hamiltonian(
                H, Sp, t, e, Sm=Sm, binds=binds, bskip=bskip
            )
        else:
            return (
                add_chain_bath_hamiltonian(
                    H, Sp, t, e, Sm=Sm, binds=binds, bskip=bskip
                ),
                e,
            )
    elif geom == "ipchain":
        w2 = copy.deepcopy(w)
        t, e, U = chain_map(g, w, return_unitary=True)
        if not return_frequencies:
            return add_ipchain_bath_hamiltonian(
                H, Sp, t[0], w2, U, Sm=Sm, binds=binds, bskip=bskip
            )
        else:
            return (
                add_ipchain_bath_hamiltonian(
                    H, Sp, t[0], w2, U, Sm=Sm, binds=binds, bskip=bskip
                ),
                e,
            )
    else:
        raise RuntimeError("Cannot add bath Hamiltonian geometry not recognised.")
