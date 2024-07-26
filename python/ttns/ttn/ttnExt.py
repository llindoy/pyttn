from pyttn._pyttn import ttn_real, ttn_complex, ttn_data_real, ttn_data_complex
import numpy as np

def ttn(*args, dtype = np.complex128, **kwargs):
    if(args):
        if isinstance(args[0], ttn_complex):
            return ttn_complex(*args, **kwargs)
        elif isinstance(args[0], ttn_real):
            if(dtype == np.complex128):
                return ttn_complex(*args, **kwargs)
            else:
                return ttn_real(*args, **kwargs)
        else:
            if(dtype == np.complex128):
                return ttn_complex(*args, **kwargs)
            elif(dtype == np.float64):
                return ttn_real(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ttn")
    else:
        if(dtype == np.complex128):
            return ttn_complex(*args, **kwargs)
        elif(dtype == np.float64):
            return ttn_real(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for ttn")
