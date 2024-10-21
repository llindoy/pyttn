from pyttn._pyttn.ops import site_operator_real, site_operator_complex
from .opsExt import ops
from .opsExt import __site_op_dict__
import numpy as np

def site_operator(*args, mode = None, optype=None, dtype=np.complex128, **kwargs):
    ret = None
    if(optype is None):
        if(args and len(args) == 1):
            if(args[0].complex_dtype()):
                ret = site_operator_complex(args[0])
            else:
                ret = site_operator_real(args[0])
        elif args and len(args) <= 3:
            if(dtype == np.complex128):
                ret = site_operator_complex(*args, **kwargs)
            else:
                ret = site_operator_real(*args, **kwargs)
        else:
            raise RuntimeError("Failed to construct site_operator object invalid arguments.")
    else:
        if optype in __site_op_dict__:
            M = __site_op_dict__[optype](*args, dtype=dtype, **kwargs)
            if(M.complex_dtype()):
                ret = site_operator_complex(M)
            else:
                ret = site_operator_real(M)
        else:
            raise RuntimeError("Failed to construct site_operator object.  optype not recognized.")
    if not mode is None:
        ret.mode = mode

    return ret
