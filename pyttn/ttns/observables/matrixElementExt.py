import numpy as np


def matrix_element(*args, dtype=np.complex128, **kwargs):
    r"""A factory method for constructing an object used for evaluating matrix elements from a TTN. If this function is passed a TTN object it uses
    this to construct a matrix_element suitable for evaluating matrix elements of this class.  Otherwise this will construct an empty matrix_element
    object with the required dtype.

    :param *args: A variable length set of arguments that is either empty or contains the TTN that we will want to evaluate matrix elements of. 
    :type *args: empty or single ttn_complex, ttn_real, ms_ttn_complex, ms_ttn_real
    :param dtype: The type to be stored in the matrix element object.  This is ignored if the a TTN object is passed in the first argument.
    :type dtype: {np.float64, np.complex128}, optional
    :param **kwargs: Keyword arguments to pass to the matrix element engine constructor.  For details see matrix_element_real or matrix_element_complex

    :returns: The matrix_element object
    :rtype: matrix_element_real or matrix_element_complex
    """
    from pyttn.ttnpp import matrix_element_complex, ttn_complex, ms_ttn_complex

    try:
        from pyttn.ttnpp import matrix_element_real, ttn_real, ms_ttn_real
        if (args):
            if isinstance(args[0], ttn_complex) or isinstance(args[0], ms_ttn_complex):
                return matrix_element_complex(*args, **kwargs)
            elif isinstance(args[0], ttn_real) or isinstance(args[0], ms_ttn_real):
                return matrix_element_real(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for matrix_element")
        else:
            if (dtype == np.complex128):
                return matrix_element_complex(**kwargs)
            elif (dtype == np.float64):
                return matrix_element_real(**kwargs)
            else:
                raise RuntimeError("Invalid dtype for matrix_element")

    except ImportError:
        if (args):
            if isinstance(args[0], ttn_complex) or isinstance(args[0], ms_ttn_complex):
                return matrix_element_complex(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for matrix_element")
        else:
            if (dtype == np.complex128):
                return matrix_element_complex(**kwargs)
            else:
                raise RuntimeError("Invalid dtype for matrix_element")
