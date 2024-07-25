from pyttn._pyttn import matrix_element_real, matrix_element_complex, ttn_real, ttn_complex
import numpy as np

def matrix_element(*args, dtype = np.complex128):
    if(args):
        if isinstance(args[0], ttn_complex):
            return matrix_element_complex(*args)
        elif isinstance(args[0], ttn_real):
            return matrix_element_real(*args)
        else:
            raise RuntimeError("Invalid dtype for matrix_element")
    else:
        if(dtype == np.complex128):
            return matrix_element_complex()
        elif(dtype == np.float64):
            return matrix_element_real()
        else:
            raise RuntimeError("Invalid dtype for matrix_element")
