import pyttn._pyttn.ops as ops
import numpy as np
import scipy as sp

def identity(*args, dtype=np.complex128):
    """
    """
    if(dtype == np.complex128):
        return ops.identity_complex(*args)
    elif(dtype == np.float64):
        return ops.identity_real(*args)
    else:
        raise RuntimeError("Invalid dtype for identity operator")

def matrix(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.matrix_complex(*args)
    elif(dtype == np.float64):
        return ops.matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for matrix operator")

def sparse_matrix_default(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.sparse_matrix_complex(*args)
    elif(dtype == np.float64):
        return ops.sparse_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sparse_matrix operator")

def sparse_matrix(*args, dtype=np.complex128):
    if(len(args) == 1):
        if isinstance(args[0], sp.sparse.csr_matrix) or isinstance(args[0], sp.sparse.coo_matrix):
            m2 = None
            if isinstance(args[0], sp.sparse.csr_matrix):
                m2 = args[0]
            else:
                m2 = args[0].tocsr()

            if(m2.dtype == np.complex128 or dtype==np.complex128):
                return ops.sparse_matrix_complex(m2.data, m2.indices, m2.indptr, ncols=m2.shape[1])
            else:
                return ops.sparse_matrix_real(m2.data, m2.indices, m2.indptr, ncols=m2.shape[1])
        else:
            sparse_matrix_default(*args, dtype=dtype)
    else:
        sparse_matrix_default(*args, dtype=dtype)

def diagonal_matrix(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.diagonal_matrix_complex(*args)
    elif(dtype == np.float64):
        return ops.diagonal_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for diagonal_matrix operator")



__site_op_dict__ = {
        "identity": identity, 
        "matrix": matrix, 
        "sparse_matrix": sparse_matrix, 
        "diagonal_matrix": diagonal_matrix, 
}
