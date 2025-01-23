import numpy as np

def ttn(*args, dtype = np.complex128, **kwargs):
    """Factory function for constructing a tree tensor network state operator

    :param \*args: Variable length list of arguments. This function can handle two possible lists of arguments

        - Default construct TTN object
        - ttn (ttn_dtype) - Copy construct TTN object
        - slice (ms_ttn_slice_dtype) - Construct TTN object from slice of multiset ttn
        - tree (ntree) - Construct TTN from an Ntree object
        - string (str) - Construct TTN from an string defining an Ntree object 

    :param dtype: The dtype to use for the site operator.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional
    :param \*\*kwargs: Additional keyword arguments that are based to the ttn object constructor

    :returns: The Tree Tensor Network State object
    :rtype: ttn_dtype (dtype=complex or real)
    """

    from pyttn.ttnpp import ttn_complex

    try:
        from pyttn.ttnpp import ttn_real
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
    """Factory function for constructing a multiset tree tensor network state operator

    :param \*args: Variable length list of arguments. This function can handle two possible lists of arguments

        - Default construct a multiset TTN object
        - msttn (ms_ttn_dtype) - Copy construct a multiset TTN object
        - tree (ntree), nset (int) - Construct a multiset TTN from an Ntree object and the number of set variables
        - string (str), nset (int) - Construct multiset TTN from an string defining an Ntree object and the number of set variables

    :param dtype: The dtype to use for the site operator.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional
    :param \*\*kwargs: Additional keyword arguments that are based to the ttn object constructor

    :returns: The Multiset Tree Tensor Network State object
    :rtype: ms_ttn_dtype (dtype=complex or real)
    """

    from pyttn.ttnpp import ms_ttn_complex

    try:
        from pyttn.ttnpp import ms_ttn_real
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
    """Factory function for constructing a multiset tree tensor network state operator

    :param \*args: Variable length list of arguments. This function can handle two possible lists of arguments

        - Default construct a multiset TTN object
        - msttn (ms_ttn_dtype) - Copy construct a multiset TTN object
        - tree (ntree), nset (int) - Construct a multiset TTN from an Ntree object and the number of set variables
        - string (str), nset (int) - Construct multiset TTN from an string defining an Ntree object and the number of set variables

    :param dtype: The dtype to use for the site operator.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional
    :param \*\*kwargs: Additional keyword arguments that are based to the ttn object constructor

    :returns: The Multiset Tree Tensor Network State object
    :rtype: ms_ttn_dtype (dtype=complex or real)
    """

    return multiset_ttn(*args, dtype=dtype, **kwargs)
