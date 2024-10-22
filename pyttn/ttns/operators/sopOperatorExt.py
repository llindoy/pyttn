import numpy as np

from pyttn._pyttn import sop_operator_complex, ttn_complex, SOP_complex, system_modes
from pyttn._pyttn import multiset_sop_operator_complex, ms_ttn_complex, multiset_SOP_complex
try:
    from pyttn._pyttn import sop_operator_real, ttn_real, SOP_real
    from pyttn._pyttn import multiset_sop_operator_real, ms_ttn_real, multiset_SOP_real

except ImportError:
    sop_operator_real = None
    ttn_real = None 
    SOP_real = None
    multiset_sop_operator_real = None
    ms_ttn_real = None 
    multiset_SOP_real = None

def sop_operator(h: SOP_real|SOP_complex, A : ttn_real|ttn_complex, sysinf : system_modes, *args, **kwargs):
    r"""Function for constructing the hierarchical sum of product operator of a string operator

    Parameters:
    h (SOP_real or SOP_complex): The sum of product operator representation of the Hamiltonian
    A (ttn_real or ttn_complex): A TTN object with defining the topology of output hierarchical SOP object
    sysinf (system_modes): The composition of the system defining the default dictionary to be considered for each node

    Other Parameters:
    opdict (operator_dictionary_real or operator_dictionary_complex): User defined operator dictionary

    Keyword Args:
    compress (bool): Whether to perform a hierarchical SOP compression
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
