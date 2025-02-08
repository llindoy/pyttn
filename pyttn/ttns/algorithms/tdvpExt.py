from pyttn.ttnpp import one_site_tdvp_complex, ttn_complex, sop_operator_complex, adaptive_one_site_tdvp_complex
from pyttn.ttnpp import multiset_one_site_tdvp_complex, ms_ttn_complex, multiset_sop_operator_complex
import numpy as np


def single_set_tdvp(A, H, expansion='onesite', **kwargs):
    """A factory method for constructing an object used for performing single set TDVP calculations

    :param A: Tree Tensor Network that the DMRG algorithm will act on
    :type A: ttn_complex
    :param H: The Hamiltonian sop operator object
    :type H: sop_operator_complex
    :param expansion: A string determining the type of bond dimension expansion to be used.  Either no subspace expansion ('onesite') or energy variance based ('subspace').  (Default: 'onesite')
    :type expansion: {'onesite', 'subspace'}, optional
    :param \*\*kwargs: Keyword arguments to pass to the DMRG engine constructor.  For details see one_site_dmrg_complex or adaptive_one_site_tdvp_complex

    :returns: The TDVP evaluation object
    :rtype: one_site_tdvp_complex or adaptive_one_site_tdvp_complex
    """
    if isinstance(A, ttn_complex) and isinstance(H, sop_operator_complex):
        if expansion == 'onesite':
            return one_site_tdvp_complex(A, H, **kwargs)
        elif expansion == 'subspace':
            return adaptive_one_site_tdvp_complex(A, H, **kwargs)
    else:
        raise RuntimeError("Invalid input types for single set tdvp.")


def multiset_tdvp(A, H, expansion='onesite', **kwargs):
    """A factory method for constructing an object used for performing multiset tdvp calculations

    :param A: Tree Tensor Network that the TDVP algorithm will act on
    :type A: ttn_complex
    :param H: The Hamiltonian sop operator object
    :type H: sop_operator_complex
    :param expansion: A string determining the type of bond dimension expansion to be used.  (Default: 'onesite')
    :type expansion: {'onesite'}, optional
    :param \*\*kwargs: Keyword arguments to pass to the DMRG engine constructor.  For details see multiset_one_site_dmrg_complex 

    :returns: The TDVP evaluation object
    :rtype: multiset_one_site_tdvp_complex 
    """
    if isinstance(A, ms_ttn_complex) and isinstance(H, multiset_sop_operator_complex):
        if expansion == 'onesite':
            return multiset_one_site_tdvp_complex(A, H, **kwargs)
        elif expansion == 'subspace':
            raise ValueError(
                "subspace expansion algorithm has not yet been implemented for multiset TTNs.")
    else:
        raise RuntimeError("Invalid input types for multiset tdvp.")


def tdvp(A, H, expansion='onesite', **kwargs):
    """A factory method for constructing an object used for performing either single or multi set tdvp calculations. 
    Which type to construct is determined by the types of the input A and h matrices.

    :param A: Tree Tensor Network that the DMRG algorithm will act on
    :type A: ttn_complex or ms_ttn_complex
    :param H: The Hamiltonian sop operator object
    :type H: sop_operator_complex
    :param expansion: A string determining the type of bond dimension expansion to be used.  Either no subspace expansion ('onesite') or energy variance based ('subspace').  (Default: 'onesite')
    :type expansion: {'onesite', 'subspace'}, optional
    :param \*\*kwargs: Keyword arguments to pass to the DMRG engine constructor.  For details see one_site_tdvp_complex or adaptive_one_site_tdvp_complex

    :returns: The TDVP evaluation object
    :rtype: one_site_tdvp_complex or adaptive_one_site_tdvp_complex or multiset_one_site_tdvp_complex
    """
    if isinstance(A, ttn_complex) and isinstance(H, sop_operator_complex):
        return single_set_tdvp(A, H, expansion=expansion, **kwargs)
    elif isinstance(A, ms_ttn_complex) and isinstance(H, multiset_sop_operator_complex):
        return multiset_tdvp(A, H, expansion=expansion, **kwargs)
    else:
        raise RuntimeError("Invalid input types for tdvp.")
