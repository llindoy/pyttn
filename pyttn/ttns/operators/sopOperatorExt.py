import numpy as np

from pyttn.ttnpp import sop_operator_complex, ttn_complex, SOP_complex, system_modes
from pyttn.ttnpp import multiset_sop_operator_complex, ms_ttn_complex, multiset_SOP_complex
try:
    from pyttn.ttnpp import sop_operator_real, ttn_real, SOP_real
    from pyttn.ttnpp import multiset_sop_operator_real, ms_ttn_real, multiset_SOP_real

except ImportError:
    sop_operator_real = None
    ttn_real = None 
    SOP_real = None
    multiset_sop_operator_real = None
    ms_ttn_real = None 
    multiset_SOP_real = None

def sop_operator(h: SOP_real|SOP_complex, A : ttn_real|ttn_complex, sysinf : system_modes, *args, **kwargs):
    """Function for constructing the hierarchical sum of product operator of a string operator

    :param h: The sum of product operator representation of the Hamiltonian
    :type h: SOP_real or SOP_complex
    :param A: A TTN object with defining the topology of output hierarchical SOP object
    :type A: ttn_real or ttn_complex
    :param sysinf: The composition of the system defining the default dictionary to be considered for each node
    :type sysinf: system_modes
    :type \*args: Variable length list of arguments. See sop_operator_complex/sop_operator_real for options
    :type \*\*kwargs: Additional keyword arguments. See sop_operator_complex/sop_operator_real for options
    """
    if SOP_real is None:
        if isinstance(A, ttn_complex) and isinstance(h, SOP_complex):
            return sop_operator_complex(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError("Invalid argument for the creation of a sop_operator.")
    else:
        if isinstance(A, ttn_real) and isinstance(h, SOP_real):
            return sop_operator_real(h, A, sysinf, *args, **kwargs)
        elif isinstance(A, ttn_complex) and isinstance(h, SOP_complex):
            return sop_operator_complex(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError("Invalid argument for the creation of a sop_operator.")

def multiset_sop_operator(h, A, sysinf, *args, **kwargs):
    """Function for constructing the multiset hierarchical sum of product operator of a string operator

    :param h: The sum of product operator representation of the Hamiltonian
    :type h: multiset_SOP_real or multiset_SOP_complex
    :param A: A TTN object with defining the topology of output hierarchical SOP object
    :type A: ms_ttn_real or ms_ttn_complex
    :param sysinf: The composition of the system defining the default dictionary to be considered for each node
    :type sysinf: system_modes
    :type \*args: Variable length list of arguments. See multiset_sop_operator_complex/multiset_sop_operator_real for options
    :type \*\*kwargs: Additional keyword arguments. See /multiset_sop_operator_complex/multiset_sop_operator_real for options
    """
    if multiset_SOP_real is None:
        if isinstance(A, ms_ttn_complex) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError("Invalid argument for the creation of a sop_operator.")
    else:
        if isinstance(A, ms_ttn_real) and isinstance(h, multiset_SOP_real) and not multiset_SOP_real is None:
            return multiset_sop_operator_real(h, A, sysinf, *args, **kwargs)
        elif isinstance(A, ms_ttn_complex) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError("Invalid argument for the creation of a sop_operator.")
