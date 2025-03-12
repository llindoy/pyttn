import numpy as np
import copy

from .chain_map import chain_map
from pyttn import fOP, coeff


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


def add_fermionic_bath_hamiltonian(H, Sp, Sm, g, w, binds=None, geom="star", bskip=1):
    if geom == "star":
        return (
            add_fermionic_star_bath_hamiltonian(
                H, Sp, Sm, g, w, binds=binds, bskip=bskip
            ),
            w,
        )
    elif geom == "chain":
        t, e = chain_map(g, w)
        return (
            add_fermionic_chain_bath_hamiltonian(
                H, Sp, Sm, t, e, binds=binds, bskip=bskip
            ),
            e,
        )
    else:
        raise RuntimeError("Cannot add bath Hamiltonian geometry not recognised.")
