import numpy as np

from pyttn.ttnpp import sop_operator_complex
from pyttn.ttnpp import ttn_complex
from pyttn.ttnpp import SOP_complex
try:
    from pyttn.ttnpp import sop_operator_real
    from pyttn.ttnpp import ttn_real
    from pyttn.ttnpp import SOP_real

    __real_ttn_import = True

except ImportError:
    __real_ttn_import = False
    sop_operator_real = None
    ttn_real = None
    SOP_real = None

#and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import sop_operator_complex as sop_operator_complex_cuda
    from pyttn.ttnpp.cuda import ttn_complex as ttn_complex_cuda

    __cuda_import = True

    #and if we have imported real ttns we import the cuda versions
    if __real_ttn_import:
        from pyttn.ttnpp.cuda import sop_operator_real as sop_operator_real_cuda
        from pyttn.ttnpp.cuda import ttn_real as ttn_real_cuda

    else:
        sop_operator_real_cuda = None
        ttn_real_cuda = None

except ImportError:
    __cuda_import = False
    sop_operator_real_cuda = None
    sop_operator_complex_cuda = None
    ttn_real_cuda = None
    ttn_complex_cuda = None


def __sop_operator_blas(h, A, sysinf, *args, **kwargs):
    if not __real_ttn_import:
        if isinstance(A, ttn_complex) and isinstance(h, SOP_complex):
            print(h)
            return sop_operator_complex(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a sop_operator.")
    else:
        if isinstance(A, ttn_real) and isinstance(h, SOP_real):
            return sop_operator_real(h, A, sysinf, *args, **kwargs)
        elif isinstance(A, ttn_complex) and isinstance(h, SOP_complex):
            return sop_operator_complex(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a sop_operator.")

def __sop_operator_cuda(h, A, sysinf, *args, **kwargs):

    if not __real_ttn_import:
        if isinstance(A, ttn_complex_cuda) and isinstance(h, SOP_complex):
            return sop_operator_complex_cuda(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a sop_operator.")
    else:
        if isinstance(A, ttn_real_cuda) and isinstance(h, SOP_real):
            return sop_operator_real_cuda(h, A, sysinf, *args, **kwargs)
        elif isinstance(A, ttn_complex_cuda) and isinstance(h, SOP_complex):
            return sop_operator_complex_cuda(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a sop_operator.")

def sop_operator(h, A, sysinf, *args, **kwargs):
    """Function for constructing the hierarchical sum of product operator of a string operator

    :param h: The sum of product operator representation of the Hamiltonian
    :type h: SOP_real or SOP_complex
    :param A: A TTN object with defining the topology of output hierarchical SOP object
    :type A: ttn_real or ttn_complex
    :param sysinf: The composition of the system defining the default dictionary to be considered for each node
    :type sysinf: system_modes
    :type \\*args: Variable length list of arguments. See sop_operator_complex/sop_operator_real for options
    :type \\*\\*kwargs: Additional keyword arguments. See sop_operator_complex/sop_operator_real for options
    """

    if len(args) > 0:
        if(args[0].backend() != A.backend()):
            raise RuntimeError("Attempted to construct sop_operator with opdict but opdict backend is not compatible with ttn backend.")
    if(A.backend() == "blas"):
        return __sop_operator_blas(h, A, sysinf, *args, **kwargs)
    elif __cuda_import and A.backend()== "cuda":
        return __sop_operator_cuda(h, A, sysinf, *args, **kwargs)
    else:
        raise RuntimeError("Invalid backend type for sop_operator")
