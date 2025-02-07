from ..ttnpp.linalg import *
import numpy as np

def vector(v):
    """
    A function for converting from a 1 dimensional numpy array to a C++ linalg::tensor<T,1> type
    used by the C++ layer of pyTTN.

    :param M: The numpy array 
    :type M: np.ndarray

    :returns: A pybind11 wrapped linalg::tensor<T, 1> object
    :rtype: vector_real, vector_complex
    """

    if v.dtype == np.float64:
        return vector_real(v)
    elif v.dtype == np.complex128:
        return vector_complex(v)
    else:
        raise RuntimeError("Invalid dtype for linalg.vector");

def matrix(M):
    """
    A function for converting from a numpy array to a C++ linalg::tensor<T,2> type
    used by the C++ layer of pyTTN.

    :param M: The numpy matrix 
    :type M: np.ndarray

    :returns: A pybind11 wrapped linalg::tensor<T, 2> object
    :rtype: matrix_real, matrix_complex
    """
    if M.dtype == np.float64:
        return matrix_real(M)
    elif M.dtype == np.complex128:
        return matrix_complex(M)
    else:
        raise RuntimeError("Invalid dtype for linalg.matrix");

def tensor(T):
    """
    A function for converting from a numpy array to a C++ linalg::tensor<T,D> type
    for D<=4 used by the C++ layer of pyTTN.

    :param M: The numpy tensor 
    :type M: np.ndarray

    :returns: A pybind11 wrapped linalg::tensor<T, D> object
    :rtype: pybind11 wrapped linalg::tensor<T,D> object
    """
    if(T.ndim == 1):
        return vector(T)
    elif (T.ndim == 2):
        return matrix(T)
    elif (T.ndim == 3):
        if T.dtype == np.float64:
            return tensor_3_real(T)
        elif T.dtype == np.complex128:
            return tensor_3_complex(T)
        else:
            raise RuntimeError("Invalid dtype for linalg.tensor");
    elif (T.ndim == 4):
        if T.dtype == np.float64:
            return tensor_4_real(T)
        elif T.dtype == np.complex128:
            return tensor_4_complex(T)
        else:
            raise RuntimeError("Invalid dtype for linalg.tensor");
    else:
        raise RuntimeError("Incompatible matrix dimensions");
