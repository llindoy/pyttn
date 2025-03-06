import numpy as np
import copy

from pyttn import sOP, coeff

def add_fermionic_bath_generator(H, Sp, dks, zks, Sm=None, binds=None, bskip=1, method="heom"):
    raise RuntimeError("Fermionic HEOM and Pseudofermion not yet implemented.")

"""
from .chain_map import chain_map
from pyttn import fOP, coeff

#setup the star Hamiltonian for the spin boson model
def add_fermionic_star_bath_hamiltonian(H, Sp, Sm, g, w, binds = None):
    Nb = g.shape[0]
    if not isinstance(binds, np.ndarray):
        if binds is None:
            binds = [i+1 for i in range(Nb)]

    for i in range(Nb):
        H += g[i] * Sp * fOP("c", binds[i])
        H += g[i] * fOP("cdag", binds[i]) * Sm
        H += w[i] * fOP("n", binds[i])

    return H

#setup the chain hamiltonian for the spin boson model - this is the tedopa method
def add_fermionic_chain_bath_hamiltonian(H, Sp, Sm, t, e, binds = None):
    Nb = e.shape[0]
    if not isinstance(binds, np.ndarray):
        if binds is None:
            binds = [i+1 for i in range(Nb)]

    for i in range(Nb):
        if i == 0:
            H += t[i]*Sp * fOP("c", binds[i])
            H += t[i] * fOP("cdag", binds[i])*Sm
        else:
            H += t[i]*fOP("cdag", binds[i-1])*fOP("c", binds[i])  
            H += t[i]*fOP("cdag", binds[i])*fOP("c", binds[i-1])
        H += e[i] * fOP("n", binds[i])

    return H

#setup the chain hamiltonian for the spin boson model - that is this implements the method described in Nuomin, Beratan, Zhang, Phys. Rev. A 105, 032406
def add_fermionic_ipchain_bath_hamiltonian(H, Sp, Sm, Nb, t0, w, P, binds = None):
    if not isinstance(binds, np.ndarray):
        if binds is None:
            binds = [i+1 for i in range(Nb)]

    class func_class:
        def __init__(self, i, t0, e0, U0, conj = False):
            self.i = copy.deepcopy(i)
            self.conj=conj
            self.t0 = copy.deepcopy(t0)
            self.e = copy.deepcopy(e0)
            self.U = copy.deepcopy(U0)

        def __call__(self, ti):
            val = self.t0*np.conj(self.U[:, 0])@(np.exp(-1.0j*ti*self.e)*self.U[:, self.i])

            if(self.conj):
                val = np.conj(val)

            return val

    for i in range(Nb):
        H += coeff(func_class(i, t0, w, P, conj=False))*Sp*fOP("c", binds[i]) 
        H += coeff(func_class(i, t0, w, P, conj=True ))*fOP("cdag", binds[i])*Sm

    return H

def add_fermionic_bath_hamiltonian(H, Sp, Sm, g, w, binds = None, geom='star', return_frequencies=False):
    if geom == 'star':
        if not return_frequencies:
            return add_fermionic_star_bath_hamiltonian(H, Sp, Sm, g, w, binds=binds)
        else:
            return add_fermionic_star_bath_hamiltonian(H, Sp, Sm, g, w, binds=binds), w
    elif geom == 'chain':
        t, e = chain_map(g, w)
        if not return_frequencies:
            return add_fermionic_chain_bath_hamiltonian(H, Sp, Sm, t, e, binds=binds)
        else:
            return add_fermionic_star_bath_hamiltonian(H, Sp, Sm, g, w, binds=binds), e
    elif geom == 'ipchain':
        w2 = copy.deepcopy(w)
        t, e, U = chain_map(g, w, return_unitary = True)
        if not return_frequencies:
            return add_fermionic_ipchain_bath_hamiltonian(H, Sp, Sm, e.shape[0], t[0], w2, U, binds=binds)
        else:
            return add_fermionic_star_bath_hamiltonian(H, Sp, Sm, g, w, binds=binds), e
    else:
        raise RuntimeError("Cannot add bath Hamiltonian geometry not recognised.")

"""
