from pyttn.ttnpp import fermion_operator, fOP
from pyttn.ttnpp import sOP, sPOP, sNBO_real, sNBO_complex, sSOP_real, sSOP_complex, coeff_real, coeff_complex
import numpy as np


def coeff(coeff, dtype=np.complex128):
    """A function for constructing the coeff type for Hamiltonian specification

    :param coeff: A variable list for specifying the coefficient.  Valid options are

        - Default construct the coefficient
        - value (dtype) - Set the coefficient to a constant value
        - func (callable) - set the coefficient to a time-dependent value
    :param dtype: The internal variable type for the product operator.
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The coefficient object
    :rtype: coeff_real or coeff_complex
    """
    if (dtype == np.complex128):
        return coeff_complex(*args)
    elif (dtype == np.float64):
        return coeff_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sNBO")


def sNBO(*args, dtype=np.complex128):
    """A function for constructing an n-body operator string

    :param \*args: A variable list for specifying the coefficient.  Valid options are

        -  Default construct the sNBO
        - op (sOP) - Construct NBO from single site operator
        - pop (sPOP) - Construct NBO from product operator
        - arg (dtype), op (sOP) - Construct NBO as a product of a constant and a single site operator
        - arg (dtype), pop (sPOP) - Construct NBO as a product of a constant and a product operator
        - arg (coeff), op (sOP) - Construct NBO as a product of a coefficient and a single site operator
        - arg (coeff), pop (sPOP) - Construct NBO as a product of a coefficient and a product operator
        - nbo (sNBO) - Construct NBO from another NBO

    :param dtype: The internal variable type for the n-body operator string.
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The n-body operator object
    :rtype: sNBO_real or sNBO_complex
    """
    if (dtype == np.complex128):
        return sNBO_complex(*args)
    elif (dtype == np.float64):
        return sNBO_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sNBO")


def sSOP(*args, dtype=np.complex128):
    """A function for constructing a sum-of-product string operator

    :param \*args: A variable list for specifying the coefficient.  Valid options are

        -  Default construct the sSOP
        - op (str) - Construct the sSOP from a string defining a sOP
        - op (sOP) - Construct sSOP from single site operator
        - pop (sPOP) - Construct sSOP from product operator
        - nbo (sNBO) - Construct sSOP from an sNBO
        - sop (sSOP) - Construct sSOP from another sSOP

    :param dtype: The internal variable type for the sum-of-product string operator.
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The sum-of-product string operator
    :rtype: sSOP_real or sSOP_complex
    """
    if (dtype == np.complex128):
        return sSOP_complex(*args)
    elif (dtype == np.float64):
        return sSOP_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sSOP")


def __is_sOP(a):
    return isinstance(a, sOP)


def __is_sPOP(a):
    return isinstance(a, sPOP)


def __is_sNBO(a):
    return isinstance(a, sNBO_complex) or isinstance(a, sNBO_real)


def __is_sSOP(a):
    return isinstance(a, sSOP_complex) or isinstance(a, sSOP_real)
