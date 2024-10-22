from pyttn._pyttn import system_modes
from .sSOPExt import is_sSOP, is_sNBO, is_sPOP, is_sOP
import numpy as np

def SOP(*args, dtype = np.complex128):
    from pyttn._pyttn import SOP_complex
    try:
        from pyttn._pyttn import SOP_real
        if(dtype == np.complex128):
            return SOP_complex(*args)
        elif(dtype == np.float64):
            return SOP_real(*args)
        else:
            raise RuntimeError("Invalid dtype for SOP")
    except ImportError:
        if(dtype == np.complex128):
            return SOP_complex(*args)
        else:
            raise RuntimeError("Invalid dtype for SOP")

def multiset_SOP(*args, dtype = np.complex128):
    from pyttn._pyttn import multiset_SOP_complex
    try:
        from pyttn._pyttn import multiset_SOP_real
        if(dtype == np.complex128):
            return multiset_SOP_complex(*args)
        elif(dtype == np.float64):
            return multiset_SOP_real(*args)
        else:
            raise RuntimeError("Invalid dtype for multiset_SOP")
    except ImportError:
        if(dtype == np.complex128):
            return multiset_SOP_complex(*args)
        else:
            raise RuntimeError("Invalid dtype for multiset_SOP")

def sum_of_product(nset, *args, dtype=np.complex128):
    if(nset == 1):
        return SOP(*args,dtype=dtype)
    elif nset > 1:
        return multiset_SOP(nset, *args,dtype=dtype)
    else:
        raise RuntimeError("Failed to construct sum of product operator object.")


def is_SOP_real(a):
    try:
        from pyttn._pyttn import SOP_real, multiset_SOP_real
        return isinstance(a, SOP_real) or isinstance(a, multiset_SOP_real) 
    except ImportError:
        return False


def is_SOP_complex(a):
    from pyttn._pyttn import SOP_complex, multiset_SOP_complex
    return isinstance(a, SOP_complex) or isinstance(a, multiset_SOP_complex)

def is_SOP(a):
    return is_SOP_real(a) or is_SOP_complex(a)

def validate_SOP(Op, sys):
    if is_SOP(Op):
        if(Op.nmodes() != sys.nprimitive_nmodes()):
            raise RuntimeError("Operator size is not compatible with system size.")
        return Op
    elif is_sSOP(Op):
        op = SOP(sys.nprimitive_modes())
        for term in Op:
            op += term
        return op
    elif is_sNBO(Op) or is_sPOP(Op) or is_sOP(Op):
        op = SOP(sys.nprimitive_modes())
        op += Op
        return op
    else:
        raise RuntimeError("Invalid operator type.")
