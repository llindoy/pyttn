from pyttn import ops
from pyttn import site_operator
from pyttn.linalg import vector, matrix
import pytest
import numpy as np
import os
from scipy.sparse import csr_matrix as spcsr


os.environ["OMP_NUM_THREADS"] = "1"


@pytest.mark.parametrize("N, mode", [(2, 0), (3, 4), (4, 2), (5, 6), (6, 0), (120, 2)])
def test_identity(N, mode):
    # identity constructor
    siteop = site_operator(ops.identity(N))
    assert siteop.is_identity()
    assert siteop.size() == N
    assert str(siteop) == "I_" + str(N)
    assert siteop.mode == 0

    # check that the matrix we have constructed is all close to the identity matrix
    assert np.allclose(np.array(siteop.todense()), np.identity(N))

    # identity constructor with mode info
    siteop = site_operator(ops.identity(N), mode)
    assert siteop.is_identity()
    assert siteop.size() == N
    assert str(siteop) == "I_" + str(N)
    assert siteop.mode == mode

    def apply_test(op):
        # apply to vector
        m1 = np.random.uniform(size=(N)) * (1.0 + 0.0j)
        m2 = vector(m1)
        m3 = vector(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), m1)

        # apply to matrix
        m1 = np.random.uniform(size=(N, N)) * (1.0 + 0.0j)
        m2 = matrix(m1)
        m3 = matrix(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), m1)

    apply_test(siteop)

    siteopT = siteop.transpose()
    assert siteopT.is_identity()
    assert siteopT.size() == N
    assert str(siteopT) == "I_" + str(N)
    assert siteopT.mode == mode

    apply_test(siteopT)


@pytest.mark.parametrize(
    "mat, mode",
    [
        (np.array([[1, 2], [3, 4]], dtype=np.float64), 0),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64), 12),
        (np.random.uniform(0, 1, size=(25, 25)), 5),
    ],
)
def test_matrix(mat, mode):
    siteop = site_operator(ops.matrix(mat))
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == 0

    # check that the matrix we have constructed is all close to the input dense matrix
    assert np.allclose(np.array(siteop.todense()), mat)

    siteop = site_operator(ops.matrix(mat), mode)
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == mode

    def apply_test(M, op):
        N = M.shape[0]
        # apply to vector
        m1 = np.random.uniform(size=(N)) * (1.0 + 0.0j)
        m2 = vector(m1)
        m3 = vector(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

        # apply to matrix
        m1 = np.random.uniform(size=(N, N)) * (1.0 + 0.0j)
        m2 = matrix(m1)
        m3 = matrix(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

    apply_test(mat, siteop)

    siteopT = siteop.transpose()
    assert not siteopT.is_identity()
    assert siteopT.size() == mat.shape[1]
    assert siteopT.mode == mode
    assert np.allclose(np.array(siteopT.todense()), mat.T)

    apply_test(mat.T, siteopT)


def test_csr():
    mode = 3
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    spmat = spcsr((data, indices, indptr), shape=(3, 3))
    mat = spmat.toarray()

    siteop = site_operator(ops.sparse_matrix(spmat, dtype=np.complex128))
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == 0

    # check that the matrix we have constructed is all close to the input dense matrix
    assert np.allclose(np.array(siteop.todense()), mat)

    siteop = site_operator(ops.sparse_matrix(spmat), mode)
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == mode

    def apply_test(M, op):
        N = M.shape[0]
        # apply to vector
        m1 = np.random.uniform(size=(N))
        m2 = vector(m1, dtype=np.complex128)
        m3 = vector(m1, dtype=np.complex128)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

        # apply to matrix
        m1 = np.random.uniform(size=(N, N)) * (1.0 + 0.0j)
        m2 = matrix(m1)
        m3 = matrix(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

    apply_test(mat, siteop)

    siteopT = siteop.transpose()
    assert not siteopT.is_identity()
    assert siteopT.size() == mat.shape[0]
    assert siteopT.mode == mode
    assert np.allclose(np.array(siteopT.todense()), mat.T)

    apply_test(mat.T, siteopT)


@pytest.mark.parametrize(
    "mat, mode",
    [
        (np.array([1, 2], dtype=np.float64), 0),
        (np.array([1, 2, 3], dtype=np.float64), 12),
        (np.random.uniform(0, 1, size=(25)), 5),
    ],
)
def test_diagonal(mat, mode):
    siteop = site_operator(ops.diagonal_matrix(mat))
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == 0

    # check that the matrix we have constructed is all close to the input dense matrix
    assert np.allclose(np.array(siteop.todense()), np.diag(mat))

    siteop = site_operator(ops.diagonal_matrix(mat), mode)
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == mode

    def apply_test(M, op):
        N = M.shape[0]
        # apply to vector
        m1 = np.random.uniform(size=(N))
        m2 = vector(m1, dtype=np.complex128)
        m3 = vector(m1, dtype=np.complex128)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

        # apply to matrix
        m1 = np.random.uniform(size=(N, N)) * (1.0 + 0.0j)
        m2 = matrix(m1)
        m3 = matrix(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

    apply_test(np.diag(mat), siteop)

    siteopT = siteop.transpose()
    assert not siteopT.is_identity()
    assert siteopT.size() == mat.shape[0]
    assert siteopT.mode == mode
    assert np.allclose(np.array(siteopT.todense()), np.diag(mat).T)

    apply_test(np.diag(mat).T, siteopT)
