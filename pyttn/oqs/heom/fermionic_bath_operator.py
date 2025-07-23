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

from typing import Optional, Union

import numpy as np

from pyttn import SOP, OP_type, sOP, sSOP

from .utils import generate_binds


def add_bosonic_heom_bath_generator(
    H: Union[sSOP, SOP],
    Sp: OP_type,
    dks: np.ndarray,
    zks: np.ndarray,
    Sm: Optional[OP_type] = None,
    binds: Optional[list[int]] = None,
    bskip: Optional[int] = 2,
) -> Union[sSOP, SOP]:
    """A function for adding the HEOM dynamics system+bath terms to the SOP object generator.  This function
    optionally allows for separate system raising and lowering operators, however, in the instance that the
    lowering operator is not defined it uses the raising operator for both.

    Additionally this function allows for user specified indices for the bath modes, or by default uses
    contiguous set of bath modes starting at index bskip.

    :param H: The input sum-of-product operator that the generator is to be added to
    :type H: Union[sSOP, SOP]
    :param Sp: The system part of the system bath coupling term corresponding to the system raising operator
    :type Sp: OP_type
    :param dks: The coefficients in the bath correlation function expansion
    :type dks: np.ndarray
    :param zks:  The exponents in the bath correlation function expansion
    :type zks: np.ndarray
    :param Sm: The system part of the system bath coupling term corresponding to the system lowering operator, defaults to None
    :type Sm: OP_type, optional
    :param binds: The indices of the HEOM bath modes, defaults to None
    :type binds: list, optional
    :param bskip: The number of sites to skip when define a contiguous set of bath mode indices, defaults to 2
    :type bskip: int, optional
    :return: The HEOM system bath generator
    :rtype: Union[sSOP, SOP]
    """
    Nb = 0
    for dk in dks:
        Nb = Nb + len(dk)
        if not (len(dk) == 1 or len(dk) == 2):
            raise Exception("Cannot add HEOM  bath unless each unexpected mode size")
    binds = generate_binds(binds, bskip, Nb)

    # set up the system bath operator
    c = 0
    for dk, zk in zip(dks, zks):
        # if we are dealing with a single terms.  This corresponds to the case that we have a single real value frequency
        if len(dk) == 1:
            # add on the bath terms
            H += -1.0j * zk[0] * sOP("n", binds[c])

            # add on the bath annihilation terms

            # add on the bath creation terms

            c = c + 1
        # otherwise we need to add on modes corresponding to forward and backward paths
        elif len(dk) == 2:
            # add on the bath terms
            H += -1.0j * zk[0] * sOP("n", binds[c])
            H += -1.0j * zk[1] * sOP("n", binds[c + 1])

            # add on the bath annihilation terms
            H += complex(dk[0]) * (Sp[0] - Sp[1]) * sOP("a", binds[c])
            H += complex(dk[1]) * (Sp[0] - Sp[1]) * sOP("a", binds[c + 1])

            # add on the bath creation terms
            # if the Sm operator is correctly defined use it
            if isinstance(Sm, list) and len(Sm) == 2:
                H += dk[0] * Sm[0] * sOP("adag", binds[c])
                H += -dk[1] * Sm[1] * sOP("adag", binds[c + 1])

            # otherwise just use the Sp operators
            else:
                H += dk[0] * Sp[0] * sOP("adag", binds[c])
                H += -dk[1] * Sp[1] * sOP("adag", binds[c + 1])
            c = c + 2

    return H


def add_fermionic_pseudomode_bath_generator(
    H: Union[sSOP, SOP],
    Sp: OP_type,
    Sm: OP_type,
    dks: np.ndarray,
    zks: np.ndarray,
    binds: Optional[list[int]] = None,
    bskip: Optional[int] = 2,
) -> Union[sSOP, SOP]:
    """A function for adding the pseudomode dynamics system+bath terms to the SOP object generator.  This function
    optionally allows for separate system raising and lowering operators, however, in the instance that the
    lowering operator is not defined it uses the raising operator for both.

    Additionally this function allows for user specified indices for the bath modes, or by default uses
    contiguous set of bath modes starting at index bskip.

    :param H: The input sum-of-product operator that the generator is to be added to
    :type H: Union[sSOP, SOP]
    :param Sp: The system part of the system bath coupling term corresponding to the system raising operator
    :type Sp: OP_type
    :param Sm: The system part of the system bath coupling term corresponding to the system lowering operator
    :type Sm: OP_type    
    :param dks: The coefficients in the bath correlation function expansion
    :type dks: np.ndarray
    :param zks:  The exponents in the bath correlation function expansion
    :type zks: np.ndarray
    :param binds: The indices of the HEOM bath modes, defaults to None
    :type binds: list , optional
    :param bskip: The number of sites to skip when define a contiguous set of bath mode indices, defaults to 2
    :type bskip: int, optional
    :return: The pseudomode system bath generator
    :rtype: Union[sSOP, SOP]
    """

    Nb = 0
    for dk in dks:
        Nb = Nb + len(dk)
        if not (len(dk) == 2):
            raise Exception(
                "Cannot add pseudomode bath unless each mode corresponds to forward and backward paths"
            )

    binds = generate_binds(binds, bskip, Nb)

    c = 0
    for dk, zk in zip(dks, zks):
        gk = np.real(zk[0])
        Ek = np.imag(zk[0])
        Mk = -np.imag(dk[0])

        i1 = binds[c]
        i2 = binds[c + 1]

        # add on the bath only terms
        H += complex(Ek) * (sOP("n", i1) - sOP("n", i2))  # the energy terms
        H += (2.0j* complex(gk)* (sOP("a", i1) * sOP("a", i2) - 0.5 * (sOP("n", i1) + sOP("n", i2))))  # the dissipators

        # now add on the system bath coupling terms
        H += 2.0j * complex(Mk) * (Sp[1] * sOP("a", i1))
        H += 2.0j * complex(np.conj(Mk)) * (Sp[0] * sOP("a", i2))

        H += complex(dk[0]) * Sm[0] * sOP("adag", i1) - complex(dk[1]) * Sm[1] * sOP("adag", i2)
        H += complex(dk[0]) * Sp[0] * sOP("a", i1) - complex(dk[1]) * Sp[1] * sOP("a", i2)

        c = c + 2

    return H


def add_fermionic_bath_generator(
    H: Union[sSOP, SOP],
    Sp: OP_type,
    Sm: OP_type,
    dks: np.ndarray,
    zks: np.ndarray,
    binds: Optional[list[int]] = None,
    bskip: Optional[int] = 2,
    method: str = "heom",
) -> Union[sSOP, SOP]:
    """A function for adding either HEOM or pseudomode dynamics system+bath terms to the SOP object generator.  Specifalised

    Additionally this function allows for user specified indices for the bath modes, or by default uses
    contiguous set of bath modes starting at index bskip.

    :param H: The input sum-of-product operator that the generator is to be added to
    :type H: SOP
    :param Sp: The system part of the system bath coupling term corresponding to the system raising operator
    :type Sp: OP_type
    :param Sm: The system part of the system bath coupling term corresponding to the system lowering operator
    :type Sm: OP_type   
    :param dks: The coefficients in the bath correlation function expansion
    :type dks: np.ndarray
    :param zks:  The exponents in the bath correlation function expansion
    :type zks: np.ndarray
    :param binds: The indices of the HEOM bath modes, defaults to None
    :type binds: list , optional
    :param bskip: The number of sites to skip when define a contiguous set of bath mode indices, defaults to 2
    :type bskip: int, optional
    :param method: The method to use, defaults to "heom"
    :type method: {"heom", "pseudomode"}, optional
    :return: The HEOM/pseudomode system bath generator
    :rtype: Union[sSOP, SOP]
    """
    if not isinstance(Sp, list):
        raise RuntimeError("Invalid Sp operator for heom.add_bosonic_bath_generator")
    if method == "heom":
        pass
        #return add_fermionic_heom_bath_generator(H, Sp, Sm, dks, zks, binds=binds, bskip=bskip)
    elif method == "pseudomode":
        return add_fermionic_pseudomode_bath_generator(
            H, Sp, Sm, dks, zks, binds=binds, bskip=bskip
        )
    else:
        raise RuntimeError("Pseudomode based bath method not recognised.")


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
