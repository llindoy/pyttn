import numpy as np
import copy

from .chain_map import chain_map
from pyttn import sOP, coeff

#functions for setting up the bath hamiltonian in several different geometries. 

#setup the star Hamiltonian for the spin boson model
def add_star_bath_hamiltonian(H, Sp, g, w, Sm = None, binds = None):
    Nb = g.shape[0]
    if not isinstance(binds, np.ndarray):
        if binds is None:
            binds = [i+1 for i in range(Nb)]

    for i in range(Nb):
        if Sm is None:
            H += g[i] * Sp * (sOP("adag", binds[i])+sOP("a", binds[i]))
        else:
            H += g[i] * Sp * sOP("a", binds[i])
            H += g[i] * Sm * sOP("adag", binds[i])
        H += w[i] * sOP("n", binds[i])

    return H

#setup the chain hamiltonian for the spin boson model - this is the tedopa method
def add_chain_bath_hamiltonian(H, Sp, t, e, Sm=None, binds = None):
    Nb = e.shape[0]
    if not isinstance(binds, np.ndarray):
        if binds is None:
            binds = [i+1 for i in range(Nb)]

    for i in range(Nb):
        if i == 0:
            if Sm is None:
                H += t[i] * Sp * (sOP("adag", binds[i])+sOP("a", binds[i]))
            else:
                H += t[i]*Sp * sOP("a", binds[i])
                H += t[i]*Sm * sOP("adag", binds[i])
        else:
            H += t[i]*sOP("adag", binds[i-1])*sOP("a", binds[i])  
            H += t[i]*sOP("a", binds[i-1])*sOP("adag", binds[i]) 
        H += e[i] * sOP("n", binds[i])

    return H

#setup the chain hamiltonian for the spin boson model - that is this implements the method described in Nuomin, Beratan, Zhang, Phys. Rev. A 105, 032406
def add_ipchain_bath_hamiltonian(H, Sp, Nb, t0, w, P, Sm = None, binds = None):
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
        if Sm is None:
            H += coeff(func_class(i, t0, w, P, conj=False))*Sp*sOP("a", binds[i]) 
            H += coeff(func_class(i, t0, w, P, conj=True))*Sp*sOP("adag", binds[i]) 
        else:
            H += coeff(func_class(i, t0, w, P, conj=False))*Sp*sOP("a", binds[i]) 
            H += coeff(func_class(i, t0, w, P, conj=True ))*Sm*sOP("adag", binds[i])  

    return H

def add_bath_hamiltonian(H, Sp, g, w, Sm = None, binds = None, geom='star'):
    if geom == 'star':
        return add_star_bath_hamiltonian(H, Sp, g, w, Sm=Sm, binds=binds), w
    elif geom == 'chain':
        t, e = chain_map(g, w)
        return add_chain_bath_hamiltonian(H, Sp, t, e, Sm=Sm, binds=binds), e
    elif geom == 'ipchain':
        w2 = copy.deepcopy(w)
        t, e, U = chain_map(g, w, return_unitary = True)
        return add_ipchain_bath_hamiltonian(H, Sp, e.shape[0], t[0], w2, U, Sm = Sm, binds=binds), e
    else:
        raise RuntimeError("Cannot add bath Hamiltonian geometry not recognised.")


def add_heom_operator(H, Sp, alpha, nu, sys, Sm = None, binds = None, opdict=None, Lopdict=None):
    from pyttn import liouville_space_superoperator
    Nb = alpha.shape[0]
        
    Lsp_m = liouville_space_superoperator(Sp, sys, '-', opdict=opdict, Lopdict=Lopdict)
    Lsp_p = liouville_space_superoperator(Sp, sys, '+', opdict=opdict, Lopdict=Lopdict)

    Lsm_f = None
    Lsm_b = None
    if Sm is None:
        Lsm_f = liouville_space_superoperator(Sp, sys, 'l', opdict=opdict, Lopdict=Lopdict)
        Lsm_b = liouville_space_superoperator(Sp, sys, 'r', opdict=opdict, Lopdict=Lopdict)
    else:
        Lsm_f = liouville_space_superoperator(Sm, sys, 'l', opdict=opdict, Lopdict=Lopdict)
        Lsm_b = liouville_space_superoperator(Sm, sys, 'r', opdict=opdict, Lopdict=Lopdict)

    for i in range(Nb):
        gk = 0
        hk = 0
        wk = 0

        if(i%2 == 0):
            wk = -1.0j*zk[i//2]
            gk = dk[i//2]/np.sqrt(dk[i//2])
            hk = np.sqrt(dk[i//2])
        else:
            wk = -1.0j*np.conj(zk[i//2])
            gk = -np.conj(dk[i//2])/np.sqrt(np.conj(dk[i//2]))
            hk = np.sqrt(np.conj(dk[i//2]))


