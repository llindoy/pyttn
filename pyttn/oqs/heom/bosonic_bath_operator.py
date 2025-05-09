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
from pyttn import sOP


def __generate_binds(binds, bskip, Nb):
    if not isinstance(binds, np.ndarray):
        if binds is None:
            binds = [i + bskip for i in range(Nb)]
    return binds


def add_heom_bath_generator(H, Sp, dks, zks, Sm=None, binds=None, bskip=2):
    """A function for adding the HEOM dynamics system+bath terms to the SOP object generator.  This function
    optionally allows for separate system raising and lowering operators, however, in the instance that the 
    lowering operator is not defined it uses the raising operator for both.

    Additionally this function allows for user specified indices for the bath modes, or by default uses 
    contiguous set of bath modes starting at index bskip.

    :param H: The input sum-of-product operator that the generator is to be added to
    :type H: SOP_dtype
    :param Sp: The system part of the system bath coupling term corresponding to the system raising operator
    :type Sp: {sOP, sPOP, sNBO, sSOP}
    :param dks: The coefficients in the bath correlation function expansion
    :type dks: np.ndarray
    :param zks:  The exponents in the bath correlation function expansion
    :type zks: np.ndarray
    :param Sm: The system part of the system bath coupling term corresponding to the system lowering operator, defaults to None
    :type Sm: {sOP, sPOP, sNBO, sSOP}, optional
    :param binds: The indices of the HEOM bath modes, defaults to None
    :type binds: {list, np.ndarray}, optional
    :param bskip: The number of sites to skip when define a contiguous set of bath mode indices, defaults to 2
    :type bskip: int, optional
    :return: The HEOM system bath generator
    :rtype: SOP_dtype
    """
    Nb = 0
    for dk in dks:
        Nb = Nb + len(dk)
        if not (len(dk) == 1 or len(dk) == 2):
            raise Exception("Cannot add HEOM  bath unless each unexpected mode size")
    binds = __generate_binds(binds, bskip, Nb)

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


def add_pseudomode_bath_generator(H, Sp, dks, zks, Sm=None, binds=None, bskip=2):
    """A function for adding the pseudomode dynamics system+bath terms to the SOP object generator.  This function
    optionally allows for separate system raising and lowering operators, however, in the instance that the 
    lowering operator is not defined it uses the raising operator for both.

    Additionally this function allows for user specified indices for the bath modes, or by default uses 
    contiguous set of bath modes starting at index bskip.

    :param H: The input sum-of-product operator that the generator is to be added to
    :type H: SOP_dtype
    :param Sp: The system part of the system bath coupling term corresponding to the system raising operator
    :type Sp: {sOP, sPOP, sNBO, sSOP}
    :param dks: The coefficients in the bath correlation function expansion
    :type dks: np.ndarray
    :param zks:  The exponents in the bath correlation function expansion
    :type zks: np.ndarray
    :param Sm: The system part of the system bath coupling term corresponding to the system lowering operator, defaults to None
    :type Sm: {sOP, sPOP, sNBO, sSOP}, optional
    :param binds: The indices of the HEOM bath modes, defaults to None
    :type binds: {list, np.ndarray}, optional
    :param bskip: The number of sites to skip when define a contiguous set of bath mode indices, defaults to 2
    :type bskip: int, optional
    :return: The pseudomode system bath generator
    :rtype: SOP_dtype
    """

    Nb = 0
    for dk in dks:
        Nb = Nb + len(dk)
        if not (len(dk) == 2):
            raise Exception(
                "Cannot add pseudomode bath unless each mode corresponds to forward and backward paths"
            )

    binds = __generate_binds(binds, bskip, Nb)

    c = 0
    for dk, zk in zip(dks, zks):
        gk = np.real(zk[0])
        Ek = np.imag(zk[0])
        Mk = -np.imag(dk[0])

        i1 = binds[c]
        i2 = binds[c + 1]

        # add on the bath only terms
        H += complex(Ek) * (sOP("n", i1) - sOP("n", i2))  # the energy terms
        H += (
            2.0j
            * complex(gk)
            * (sOP("a", i1) * sOP("a", i2) - 0.5 * (sOP("n", i1) + sOP("n", i2)))
        )  # the dissipators

        # now add on the system bath coupling terms
        # if the Sm operator is correctly defined use it
        if isinstance(Sm, list) and len(Sm) == 2:
            H += 2.0j * complex(Mk) * (Sp[1] * sOP("a", i1))
            H += 2.0j * complex(np.conj(Mk)) * (Sp[0] * sOP("a", i2))

            H += complex(dk[0]) * Sm[0] * sOP("adag", i1) - complex(dk[1]) * Sm[
                1
            ] * sOP("adag", i2)
            H += complex(dk[0]) * Sp[0] * sOP("a", i1) - complex(dk[1]) * Sp[1] * sOP(
                "a", i2
            )

        # otherwise just use the Sp operators
        else:
            H += 2.0j * complex(Mk) * (Sp[1] * sOP("a", i1))
            H += 2.0j * complex(np.conj(Mk)) * (Sp[0] * sOP("a", i2))

            H += complex(dk[0]) * Sp[0] * sOP("adag", i1) - complex(dk[1]) * Sp[
                1
            ] * sOP("adag", i2)
            H += complex(dk[0]) * Sp[0] * sOP("a", i1) - complex(dk[1]) * Sp[1] * sOP(
                "a", i2
            )

        c = c + 2

    return H


def add_bosonic_bath_generator(
    H, Sp, dks, zks, Sm=None, binds=None, bskip=2, method="heom"
):
    """A function for adding either HEOM or pseudomode dynamics system+bath terms to the SOP object generator.  This function
    optionally allows for separate system raising and lowering operators, however, in the instance that the 
    lowering operator is not defined it uses the raising operator for both.

    Additionally this function allows for user specified indices for the bath modes, or by default uses 
    contiguous set of bath modes starting at index bskip.

    :param H: The input sum-of-product operator that the generator is to be added to
    :type H: SOP_dtype
    :param Sp: The system part of the system bath coupling term corresponding to the system raising operator
    :type Sp: {sOP, sPOP, sNBO, sSOP}
    :param dks: The coefficients in the bath correlation function expansion
    :type dks: np.ndarray
    :param zks:  The exponents in the bath correlation function expansion
    :type zks: np.ndarray
    :param Sm: The system part of the system bath coupling term corresponding to the system lowering operator, defaults to None
    :type Sm: {sOP, sPOP, sNBO, sSOP}, optional
    :param binds: The indices of the HEOM bath modes, defaults to None
    :type binds: {list, np.ndarray}, optional
    :param bskip: The number of sites to skip when define a contiguous set of bath mode indices, defaults to 2
    :type bskip: int, optional
    :param method: The method to use, defaults to "heom"
    :type method: {"heom", "pseudomode"}, optional
    :return: The HEOM/pseudomode system bath generator
    :rtype: SOP_dtype
    """
    if not isinstance(Sp, list):
        raise RuntimeError("Invalid Sp operator for heom.add_bosonic_bath_generator")
    if method == "heom":
        return add_heom_bath_generator(H, Sp, dks, zks, Sm=Sm, binds=binds, bskip=bskip)
    elif method == "pseudomode":
        return add_pseudomode_bath_generator(
            H, Sp, dks, zks, Sm=Sm, binds=binds, bskip=bskip
        )
    else:
        raise RuntimeError("Pseudomode based bath method not recognised.")
