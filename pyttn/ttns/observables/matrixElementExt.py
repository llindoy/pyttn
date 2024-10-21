from pyttn._pyttn import matrix_element_real, matrix_element_complex, ttn_real, ttn_complex, ms_ttn_real, ms_ttn_complex
import numpy as np

def matrix_element(*args, dtype = np.complex128, **kwargs):
    if(args):
        if isinstance(args[0], ttn_complex) or isinstance(args[0], ms_ttn_complex):
            return matrix_element_complex(*args, **kwargs)
        elif isinstance(args[0], ttn_real) or isinstance(args[0], ms_ttn_real):
            return matrix_element_real(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for matrix_element")
    else:
        if(dtype == np.complex128):
            return matrix_element_complex(**kwargs)
        elif(dtype == np.float64):
            return matrix_element_real(**kwargs)
        else:
            raise RuntimeError("Invalid dtype for matrix_element")
