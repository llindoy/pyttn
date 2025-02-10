from .opsExt import ops
from .opsExt import __site_op_dict__
import numpy as np


from pyttn.ttnpp.ops import site_operator_complex
try:
    from pyttn.ttnpp.ops import site_operator_real
    __real_ttn_import = True

except ImportError:
    __real_ttn_import = False
    site_operator_real = None


#and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda.ops import site_operator_complex as site_operator_complex_cuda
    __cuda_import = True

    #and if we have imported real ttns we import the cuda versions
    if __real_ttn_import:
        from pyttn.ttnpp.cuda.ops import site_operator_real as site_operator_real_cuda
    else:
        site_operator_real_cuda = None

except ImportError:
    __cuda_import = False
    site_operator_real_cuda = None
    site_operator_complex_cuda = None


def __site_operator_blas(*args, mode=None, optype=None, dtype=np.complex128, **kwargs):
    ret = None
    if (optype is None):
        if (args and len(args) == 1):
            if (args[0].complex_dtype() or not __real_ttn_import):
                ret = site_operator_complex(args[0])
            else:
                ret = site_operator_real(args[0])
        elif args and len(args) <= 3:
            if (dtype == np.complex128 or not __real_ttn_import ):
                ret = site_operator_complex(*args, **kwargs)
            else:
                ret = site_operator_real(*args, **kwargs)
        else:
            raise RuntimeError(
                "Failed to construct site_operator object invalid arguments.")
    else:
        if optype in __site_op_dict__:
            M = __site_op_dict__[optype](*args, dtype=dtype, **kwargs)
            if (M.complex_dtype() or not __real_ttn_import ):
                ret = site_operator_complex(M)
            else:
                ret = site_operator_real(M)
        else:
            raise RuntimeError(
                "Failed to construct site_operator object.  optype not recognized.")
    if not mode is None:
        ret.mode = mode
    return ret


def __site_operator_cuda(*args, mode=None, optype=None, dtype=np.complex128, **kwargs):
    ret = None
    if (optype is None):
        if (args and len(args) == 1):
            if (args[0].complex_dtype() or not __real_ttn_import):
                ret = site_operator_complex_cuda(args[0])
            else:
                ret = site_operator_real_cuda(args[0])
        elif args and len(args) <= 3:
            if (dtype == np.complex128 or not __real_ttn_import ):
                ret = site_operator_complex_cuda(*args, **kwargs)
            else:
                ret = site_operator_real_cuda(*args, **kwargs)
        else:
            raise RuntimeError(
                "Failed to construct site_operator object invalid arguments.")
    else:
        if optype in __site_op_dict__:
            M = __site_op_dict__[optype](*args, dtype=dtype, **kwargs)
            if (M.complex_dtype() or not __real_ttn_import ):
                ret = site_operator_complex_cuda(M)
            else:
                ret = site_operator_real_cuda(M)
        else:
            raise RuntimeError(
                "Failed to construct site_operator object.  optype not recognized.")
    if not mode is None:
        ret.mode = mode
    return ret

def site_operator(*args, mode=None, optype=None, dtype=np.complex128, backend = "blas", **kwargs):
    """Factory function for constructing a one site operator.

    :param \*args: Variable length list of arguments. There are several valid options for the \*args parameters.  If the optype variable is None the allowed options are

        - site_op (site_operator_real or site_operato_complex) - Construct a new site_operator object from the existing object
        - op (sOP), sysinf (system_modes) - Construct a new site_operator from the string operator and system information
        - op (sOP), sysinf (system_modes), opdict (operator_dictionary_real or operator_dictionary_complex) -  Construct a new site_operator from the string operator, system information and used defined operator dictionary.

        Otherwise, if the optype variable has been set then the valid arguments are determined by the specified optype see opsExt.py for details.

    :param mode: The mode the site operator is acting on. (Default: None)
    :type mode: int or None, optional
    :param optype: The type of the operator to be constructed. (Default: None)
    :type optype: {'identity', 'matrix', 'sparse_matrix', 'diagonal_matrix'} or None, optional
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional
    :param backend: The computational backend to use for the product operator  (Default: "blas") 
    :type backend: {"blas", "cuda"}, optional
    :param \*\*kwargs: Additional keyword arguments. To construct the site_operator object
    """
    if backend == 'blas':
        return __site_operator_blas(*args, mode=mode, optype=optype, dtype=dtype, **kwargs)
    elif __cuda_import and backend == 'cuda':
        return __site_operator_cuda(*args, mode=mode, optype=optype, dtype=dtype, **kwargs)
    else:
        raise RuntimeError("Invalid backend type for site_operator")
