from pyttn._pyttn import SOP_real, SOP_complex
import numpy as np

def SOP(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return SOP_complex(*args)
    elif(dtype == np.float64):
        return SOP_real(*args)
    else:
        raise RuntimeError("Invalid dtype for SOP")

