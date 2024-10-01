from pyttn._pyttn import fermion_operator, fOP
from pyttn._pyttn import sNBO_real, sNBO_complex, sSOP_real, sSOP_complex, coeff_real, coeff_complex
import numpy as np

def coeff(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return coeff_complex(*args)
    elif(dtype == np.float64):
        return coeff_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sNBO")


def sNBO(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return sNBO_complex(*args)
    elif(dtype == np.float64):
        return sNBO_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sNBO")


def sSOP(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return sSOP_complex(*args)
    elif(dtype == np.float64):
        return sSOP_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sSOP")
