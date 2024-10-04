from pyttn._pyttn import operator_dictionary_real, operator_dictionary_complex
import numpy as np

def operator_dictionary(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return operator_dictionary_complex(*args)
    elif(dtype == np.float64):
        return operator_dictionary_real(*args)
    else:
        raise RuntimeError("Invalid dtype for operator_dictionary")

def is_operator_dictionary(a):
    return isinstance(a, operator_dictionary_complex) or isinstance(a, operator_dictionary_real)
