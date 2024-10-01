from pyttn._pyttn import sop_operator_real, sop_operator_complex, ttn_real, ttn_complex, SOP_real, SOP_complex, system_modes
from pyttn._pyttn import multiset_sop_operator_real, multiset_sop_operator_complex, ms_ttn_real, ms_ttn_complex, multiset_SOP_real, multiset_SOP_complex, system_modes
import numpy as np


def sop_operator(h, A, sysinf, *args, **kwargs):
    if isinstance(A, ttn_real) and isinstance(h, SOP_real):
        return sop_operator_real(h, A, sysinf, *args, **kwargs)
    elif isinstance(A, ttn_complex) and isinstance(h, SOP_complex):
        return sop_operator_complex(h, A, sysinf, *args, **kwargs)
    else:
        raise RuntimeError("Invalid argument for the creation of a sop_operator.")

def multiset_sop_operator(h, A, sysinf, *args, **kwargs):
    if isinstance(A, ms_ttn_real) and isinstance(h, multiset_SOP_real):
        return multiset_sop_operator_real(h, A, sysinf, *args, **kwargs)
    elif isinstance(A, ms_ttn_complex) and isinstance(h, multiset_SOP_complex):
        return multiset_sop_operator_complex(h, A, sysinf, *args, **kwargs)
    else:
        raise RuntimeError("Invalid argument for the creation of a sop_operator.")
