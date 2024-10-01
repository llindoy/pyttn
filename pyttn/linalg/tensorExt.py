from .._pyttn.linalg import *
import numpy as np

def vector(M):
    if M.dtype == np.float64:
        return vector_real(M)
    elif M.dtype == np.complex128:
        return vector_complex(M)
    else:
        raise RuntimeError("Invalid dtype for linalg.vector");

def matrix(M):
    if M.dtype == np.float64:
        return matrix_real(M)
    elif M.dtype == np.complex128:
        return matrix_complex(M)
    else:
        raise RuntimeError("Invalid dtype for linalg.matrix");

def tensor(M):
    if(M.ndim == 1):
        return vector(M)
    elif (M.ndim == 2):
        return matrix(M)
    elif (M.ndim == 3):
        if M.dtype == np.float64:
            return tensor_3_real(M)
        elif M.dtype == np.complex128:
            return tensor_3_complex(M)
        else:
            raise RuntimeError("Invalid dtype for linalg.tensor");
    elif (M.ndim == 4):
        if M.dtype == np.float64:
            return tensor_4_real(M)
        elif M.dtype == np.complex128:
            return tensor_4_complex(M)
        else:
            raise RuntimeError("Invalid dtype for linalg.tensor");
    else:
        raise RuntimeError("Incompatible matrix dimensions");
