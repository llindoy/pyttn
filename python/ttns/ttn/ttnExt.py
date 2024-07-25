from pyttn._pyttn import ttn_real, ttn_complex, ttn_data_real, ttn_data_complex
import numpy as np

def ttn_data(*args, dtype=np.complex128):
    if(args):
        if isinstance(args[0], ttn_data_complex):
            return ttn_data_complex(*args)
        elif isinstance(args[0], ttn_data_real):
            if(dtype == np.complex128):
                return ttn_data_complex(*args)
            else:
                return ttn_data_real(*args)
        else:
            raise RuntimeError("Invalid args for ttn_data")
    else:
        if(dtype == np.complex128):
            return ttn_data_complex()
        elif(dtype == np.float64):
            return ttn_data_real()
        else:
            raise RuntimeError("Invalid dtype for ttn_data")

def ttn(*args, dtype = np.complex128):
    if(args):
        if isinstance(args[0], ttn_complex):
            return ttn_complex(*args)
        elif isinstance(args[0], ttn_real):
            if(dtype == np.complex128):
                return ttn_complex(*args)
            else:
                return ttn_real(*args)
        else:
            if(dtype == np.complex128):
                return ttn_complex(*args)
            elif(dtype == np.float64):
                return ttn_real(*args)
            else:
                raise RuntimeError("Invalid dtype for ttn")
    else:
        if(dtype == np.complex128):
            return ttn_complex(*args)
        elif(dtype == np.float64):
            return ttn_real(*args)
        else:
            raise RuntimeError("Invalid dtype for ttn")
