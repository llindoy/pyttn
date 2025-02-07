import numpy as np

def operator_dictionary(*args, dtype = np.complex128):
    """Factory function for constructing a user defined operator dictionary.

    :param \*args: Variable length list of arguments. 
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The operator dictionary object
    :rtype: operator_dictionary_real or operator_dictionary_complex
    """
    from pyttn.ttnpp import operator_dictionary_complex
    try:
        from pyttn.ttnpp import operator_dictionary_real
        if(dtype == np.complex128):
            return operator_dictionary_complex(*args)
        elif(dtype == np.float64):
            return operator_dictionary_real(*args)
        else:
            raise RuntimeError("Invalid dtype for operator_dictionary")

    except ImportError:
        if(dtype == np.complex128):
            return operator_dictionary_complex(*args)
        else:
            raise RuntimeError("Invalid dtype for operator_dictionary")


def __is_operator_dictionary(a):
    from pyttn.ttnpp import operator_dictionary_complex
    try:
        from pyttn.ttnpp import operator_dictionary_real
        return isinstance(a, operator_dictionary_complex) or isinstance(a, operator_dictionary_real)

    except ImportError:
        return isinstance(a, operator_dictionary_complex)
