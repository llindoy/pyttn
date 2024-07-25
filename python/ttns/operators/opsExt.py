import pyttn._pyttn.ops as ops
import numpy as np

def identity(*args, dtype=np.complex128):
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


def adjoint_matrix(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.adjoint_matrix_complex(*args)
    elif(dtype == np.float64):
        return ops.adjoint_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for adjoint_matrix operator")

def sparse_matrix(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.sparse_matrix_complex(*args)
    elif(dtype == np.float64):
        return ops.sparse_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sparse_matrix operator")

def diagonal_matrix(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.diagonal_matrix_complex(*args)
    elif(dtype == np.float64):
        return ops.diagonal_matrix_real(*args)
    else:
        raise RuntimeError("Invalid dtype for diagonal_matrix operator")

def commutator(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.commutator_complex(*args)
    elif(dtype == np.float64):
        return ops.commutator_real(*args)
    else:
        raise RuntimeError("Invalid dtype for commutator operator")

def anti_commutator(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.anti_commutator_complex(*args)
    elif(dtype == np.float64):
        return ops.anti_commutator_real(*args)
    else:
        raise RuntimeError("Invalid dtype for anti_commutator operator")


def dvr(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.dvr_complex(*args)
    elif(dtype == np.float64):
        return ops.dvr_real(*args)
    else:
        raise RuntimeError("Invalid dtype for dvr operator")

def direct_product(*args, dtype=np.complex128):
    if(dtype == np.complex128):
        return ops.direct_product_complex(*args)
    elif(dtype == np.float64):
        return ops.direct_product_real(*args)
    else:
        raise RuntimeError("Invalid dtype for direct_product operator")


__site_op_dict__ = {
        "identity": identity, 
        "matrix": matrix, 
        "adjoint_matrix": adjoint_matrix, 
        "sparse_matrix": sparse_matrix, 
        "commutator": commutator, 
        "anti_commutator": anti_commutator, 
        "dvr": dvr, 
        "direct_product": direct_product 
}
