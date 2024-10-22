import numpy as np

def ttn(*args, dtype = np.complex128, **kwargs):
    from pyttn._pyttn import ttn_complex

    try:
        from pyttn._pyttn import ttn_real
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
    except ImportError:
        if(args):
            if isinstance(args[0], ttn_complex):
                return ttn_complex(*args, **kwargs)
            else:
                if(dtype == np.complex128):
                    return ttn_complex(*args, **kwargs)
                else:
                    raise RuntimeError("Invalid dtype for ttn")
        else:
            if(dtype == np.complex128):
                return ttn_complex(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ttn")


def multiset_ttn(*args, dtype = np.complex128, **kwargs):
    from pyttn._pyttn import ms_ttn_complex

    try:
        from pyttn._pyttn import ms_ttn_real
        if(args):
            if isinstance(args[0], ms_ttn_complex):
                return ms_ttn_complex(*args, **kwargs)
            elif isinstance(args[0], ms_ttn_real):
                if(dtype == np.complex128):
                    return ms_ttn_complex(*args, **kwargs)
                else:
                    return ms_ttn_real(*args, **kwargs)
            else:
                if(dtype == np.complex128):
                    return ms_ttn_complex(*args, **kwargs)
                elif(dtype == np.float64):
                    return ms_ttn_real(*args, **kwargs)
                else:
                    raise RuntimeError("Invalid dtype for ms_ttn")
        else:
            if(dtype == np.complex128):
                return ms_ttn_complex(*args, **kwargs)
            elif(dtype == np.float64):
                return ms_ttn_real(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ms_ttn")
            
    except ImportError:
        if(args):
            if isinstance(args[0], ms_ttn_complex):
                return ms_ttn_complex(*args, **kwargs)
            else:
                if(dtype == np.complex128):
                    return ms_ttn_complex(*args, **kwargs)
                else:
                    raise RuntimeError("Invalid dtype for ms_ttn")
        else:
            if(dtype == np.complex128):
                return ms_ttn_complex(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ms_ttn")

def ms_ttn(*args, dtype=np.complex128, **kwargs):
    return multiset_ttn(*args, dtype=dtype, **kwargs)
