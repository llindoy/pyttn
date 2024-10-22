import numpy as np

def operator_dictionary(*args, dtype = np.complex128):
    from pyttn._pyttn import operator_dictionary_complex
    try:
        from pyttn._pyttn import operator_dictionary_real
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


def is_operator_dictionary(a):
    from pyttn._pyttn import operator_dictionary_complex
    try:
        from pyttn._pyttn import operator_dictionary_real
        return isinstance(a, operator_dictionary_complex) or isinstance(a, operator_dictionary_real)

    except ImportError:
        return isinstance(a, operator_dictionary_complex)
