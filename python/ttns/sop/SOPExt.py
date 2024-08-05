from pyttn._pyttn import SOP_real, SOP_complex, multiset_SOP_real, multiset_SOP_complex
import numpy as np

def SOP(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return SOP_complex(*args)
    elif(dtype == np.float64):
        return SOP_real(*args)
    else:
        raise RuntimeError("Invalid dtype for SOP")


def multiset_SOP(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return multiset_SOP_complex(*args)
    elif(dtype == np.float64):
        return multiset_SOP_real(*args)
    else:
        raise RuntimeError("Invalid dtype for ms_SOP")


def sum_of_product(nset, *args, dtype=np.complex128):
    if(nset == 1):
        return SOP(*args,dtype=dtype)
    elif nset > 1:
        return ms_SOP(nset, *args,dtype=dtype)
    else:
        raise RuntimeError("Failed to construct sum of product operator object.")
