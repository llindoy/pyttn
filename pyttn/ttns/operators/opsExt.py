import pyttn.ttnpp.ops as ops
import numpy as np
import scipy as sp

def identity(*args, dtype=np.complex128):
    """Factory function for constructing an identity matrix site operator

    :type \*args: Variable length list of arguments. Allowed options are
        - empty - Default construct identity operator
        - size (int) - Construct an identity operator of a specified size
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional

    :returns: identity operator
    :rtype: ops.identity_complex or ops.identity_real
    """
    if(dtype == np.complex128):
        return ops.identity_complex(*args)
    elif(dtype == np.float64):
        return ops.identity_real(*args)
    else:
        raise RuntimeError("Invalid dtype for identity operator")

def matrix(*args, dtype=np.complex128):
    """Factory function for constructing a matrix site operator

    :type \*args: Variable length list of arguments. For details see the dense matrix constructors.
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional

    :returns: matrix operator
    :rtype: ops.matrix_complex or ops.matrix_real
    """
    if(dtype == np.complex128):
        return ops.matrix_complex(*args)
    elif(dtype == np.float64):
        return ops.matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for matrix operator")

def __sparse_matrix_default(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.sparse_matrix_complex(*args)
    elif(dtype == np.float64):
        return ops.sparse_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sparse_matrix operator")

def sparse_matrix(*args, dtype=np.complex128):
    """Factory function for constructing a sparse matrix site operator

    :type \*args: Variable length list of arguments. For details see the sparse matrix constructors
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional

    :returns: sparse operator
    :rtype: ops.sparse_complex or ops.sparse_real
    """

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
            __sparse_matrix_default(*args, dtype=dtype)
    else:
        __sparse_matrix_default(*args, dtype=dtype)

def diagonal_matrix(*args, dtype=np.complex128):
    """Factory function for constructing an diagonal matrix site operator

    :type \*args: Variable length list of arguments. For details see the diagonal matrix constructors
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional

    :returns: diagonal operator
    :rtype: ops.diagonal_complex or ops.diagonal_real
    """
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
