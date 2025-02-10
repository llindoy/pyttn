import numpy as np

from pyttn.ttnpp import ttn_complex, ms_ttn_complex
#and attempt to import the real ttns
try:
    from pyttn.ttnpp import ttn_real, ms_ttn_real
    __real_ttn_import = True
except ImportError:
    ttn_real = None
    ms_ttn_real = None
    __real_ttn_import = False

#and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import ttn_complex as ttn_complex_cuda
    from pyttn.ttnpp.cuda import ms_ttn_complex as ms_ttn_complex_cuda

    __cuda_import = True

    #and if we have imported real ttns we import the cuda versions
    if __real_ttn_import:
        from pyttn.ttnpp.cuda import ttn_real as ttn_real_cuda
        from pyttn.ttnpp.cuda import ms_ttn_real as ms_ttn_real_cuda    
    else:
        ttn_real_cuda = None
        ms_ttn_real_cuda = None
except ImportError:
    __cuda_import = False
    ttn_complex_cuda = None
    ms_ttn_complex_cuda = None
    ttn_real_cuda = None
    ms_ttn_real_cuda = None


def __ttn_blas(*args, dtype=np.complex128, **kwargs):
    if (args):
        if isinstance(args[0], ttn_complex):
            return ttn_complex(*args, **kwargs)
        elif __real_ttn_import and isinstance(args[0], ttn_real):
            if (dtype == np.complex128):
                return ttn_complex(*args, **kwargs)
            else:
                return ttn_real(*args, **kwargs)
        else:
            if dtype == np.complex128  or not __real_ttn_import:
                return ttn_complex(*args, **kwargs)
            elif dtype == np.float64:
                return ttn_real(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ttn")
    else:
        if (dtype == np.complex128 or not __real_ttn_import):
            return ttn_complex(*args, **kwargs)
        elif dtype == np.float64:
            return ttn_real(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for ttn")
        
def __ttn_cuda(*args, dtype = np.complex128, **kwargs):
    if (args):
        if isinstance(args[0], ttn_complex_cuda):
            return ttn_complex_cuda(*args, **kwargs)
        elif __real_ttn_import and isinstance(args[0], ttn_real_cuda):
            if (dtype == np.complex128):
                return ttn_complex_cuda(*args, **kwargs)
            else:
                return ttn_real_cuda(*args, **kwargs)
        else:
            if dtype == np.complex128  or not __real_ttn_import:
                return ttn_complex_cuda(*args, **kwargs)
            elif dtype == np.float64:
                return ttn_real_cuda(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ttn")
    else:
        if (dtype == np.complex128 or not __real_ttn_import):
            return ttn_complex_cuda(*args, **kwargs)
        elif dtype == np.float64:
            return ttn_real_cuda(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for ttn")

def ttn(*args, dtype=np.complex128, backend="blas", **kwargs):
    """Factory function for constructing a tree tensor network state operator

    :param \\*args: Variable length list of arguments. This function can handle two possible lists of arguments

        - Default construct TTN object
        - ttn (ttn_dtype) - Copy construct TTN object
        - slice (ms_ttn_slice_dtype) - Construct TTN object from slice of multiset ttn
        - tree (ntree) - Construct TTN from an Ntree object
        - string (str) - Construct TTN from an string defining an Ntree object 

    :param dtype: The dtype to use for the ttn.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional
    :param backend: The computational backend to use for the ttn.  (Default: "blas") 
    :type backend: {"blas", "cuda"}, optional
    :param \\*\\*kwargs: Additional keyword arguments that are based to the ttn object constructor

    :returns: The Tree Tensor Network State object
    :rtype: ttn_dtype (dtype=complex or real)
    """

    if backend == 'blas':
        return __ttn_blas(*args, dtype=dtype, **kwargs)
    elif __cuda_import and backend == 'cuda':
        return __ttn_cuda(*args, dtype=dtype, **kwargs)
    else:
        raise RuntimeError("Invalid backend type for ttn")


def __ms_ttn_blas(*args, dtype=np.complex128, **kwargs):
    if (args):
        if isinstance(args[0], ms_ttn_complex):
            return ms_ttn_complex(*args, **kwargs)
        elif __real_ttn_import and isinstance(args[0], ms_ttn_real):
            if (dtype == np.complex128):
                return ms_ttn_complex(*args, **kwargs)
            else:
                return ms_ttn_real(*args, **kwargs)
        else:
            if (dtype == np.complex128  or not __real_ttn_import):
                return ms_ttn_complex(*args, **kwargs)
            elif dtype == np.float64:
                return ms_ttn_real(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ms_ttn")
    else:
        if (dtype == np.complex128  or not __real_ttn_import):
            return ms_ttn_complex(*args, **kwargs)
        elif dtype == np.float64:
            return ms_ttn_real(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for ms_ttn")
        
def __ms_ttn_cuda(*args, dtype=np.complex128, **kwargs):
    if (args):
        if isinstance(args[0], ms_ttn_complex_cuda):
            return ms_ttn_complex_cuda(*args, **kwargs)
        elif __real_ttn_import and isinstance(args[0], ms_ttn_real):
            if (dtype == np.complex128):
                return ms_ttn_complex_cuda(*args, **kwargs)
            else:
                return ms_ttn_real_cuda(*args, **kwargs)
        else:
            if (dtype == np.complex128  or not __real_ttn_import):
                return ms_ttn_complex_cuda(*args, **kwargs)
            elif dtype == np.float64:
                return ms_ttn_real_cuda(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for ms_ttn")
    else:
        if (dtype == np.complex128  or not __real_ttn_import):
            return ms_ttn_complex_cuda(*args, **kwargs)
        elif dtype == np.float64:
            return ms_ttn_real_cuda(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for ms_ttn")

def multiset_ttn(*args, dtype=np.complex128, backend="blas", **kwargs):
    """Factory function for constructing a multiset tree tensor network state operator

    :param \\*args: Variable length list of arguments. This function can handle two possible lists of arguments

        - Default construct a multiset TTN object
        - msttn (ms_ttn_dtype) - Copy construct a multiset TTN object
        - tree (ntree), nset (int) - Construct a multiset TTN from an Ntree object and the number of set variables
        - string (str), nset (int) - Construct multiset TTN from an string defining an Ntree object and the number of set variables

    :param dtype: The dtype to use for the ttn.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional
    :param backend: The computational backend to use for the ttn.  (Default: "blas") 
    :type backend: {"blas", "cuda"}, optional
    :param \\*\\*kwargs: Additional keyword arguments that are based to the ttn object constructor

    :returns: The Multiset Tree Tensor Network State object
    :rtype: ms_ttn_dtype (dtype=complex or real)
    """
    if backend == 'blas':
        return __ms_ttn_blas(*args, dtype=dtype, **kwargs)
    elif __cuda_import and backend == 'cuda':
        return __ms_ttn_cuda(*args, dtype=dtype, **kwargs)
    else:
        raise RuntimeError("Invalid backend type for multiset_ttn")




def ms_ttn(*args, dtype=np.complex128, backend="blas", **kwargs):
    """Factory function for constructing a multiset tree tensor network state operator

    :param \\*args: Variable length list of arguments. This function can handle two possible lists of arguments

        - Default construct a multiset TTN object
        - msttn (ms_ttn_dtype) - Copy construct a multiset TTN object
        - tree (ntree), nset (int) - Construct a multiset TTN from an Ntree object and the number of set variables
        - string (str), nset (int) - Construct multiset TTN from an string defining an Ntree object and the number of set variables

    :param dtype: The dtype to use for the ttn.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional
    :param backend: The computational backend to use for the ttn.  (Default: "blas") 
    :type backend: {"blas", "cuda"}, optional
    :param \\*\\*kwargs: Additional keyword arguments that are based to the ttn object constructor

    :returns: The Multiset Tree Tensor Network State object
    :rtype: ms_ttn_dtype (dtype=complex or real)
    """

    return multiset_ttn(*args, dtype=dtype, backend=backend, **kwargs)
