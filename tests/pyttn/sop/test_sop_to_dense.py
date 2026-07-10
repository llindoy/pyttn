import numpy as np
import pytest

from pyttn import (
    sOP,
    sSOP,
    SOP,
    coeff,
    system_modes,
    boson_mode,
    generic_mode,
    operator_dictionary,
    site_operator,
)



# Helpers


def to_numpy(M):
    """Convert pyttn matrix to NumPy via buffer protocol (zero-copy)."""
    return np.asarray(M)


def assert_dense_equal(A, B, tol=1e-12):
    A = np.asarray(A)
    B = np.asarray(B)

    assert A.shape == B.shape
    assert np.allclose(A, B, atol=tol)


def make_simple_sys():
    sys = system_modes(2)
    sys[0] = boson_mode(3)
    sys[1] = boson_mode(3)
    return sys



# sOP


@pytest.mark.parametrize("label", ["a", "adag", "n"])
def test_sOP_todense_default(label):
    sys = make_simple_sys()

    op = sOP(label, 0)

    M1 = op.todense(sys)
    M2 = op.todense(sys, None)

    assert_dense_equal(M1, M2)


def test_sOP_todense_custom_dict():
    sys = system_modes(1)
    sys[0] = generic_mode(2)

    opdict = operator_dictionary(1)

    mat = np.zeros((2, 2), dtype=np.complex128)
    mat[0, 0] = 1.0

    opdict.insert(0, "p0", site_operator(mat, optype="matrix", mode=0))

    op = sOP("p0", 0)

    M = op.todense(sys, opdict)

    assert_dense_equal(M, mat)



# sPOP


def test_sPOP_todense_default():
    sys = make_simple_sys()

    op = sOP("a", 0) * sOP("adag", 1)

    M = np.asarray(op.todense(sys))

    assert M.shape[0] == M.shape[1]


def test_sPOP_todense_custom_dict():
    sys = system_modes(2)
    sys[0] = generic_mode(2)
    sys[1] = generic_mode(2)

    opdict = operator_dictionary(2)

    mat0 = np.diag([1.0, 0.0])
    mat1 = np.diag([0.0, 1.0])

    opdict.insert(0, "p0", site_operator(mat0, optype="matrix", mode=0))
    opdict.insert(1, "p1", site_operator(mat1, optype="matrix", mode=1))

    op = sOP("p0", 0) * sOP("p1", 1)

    M = op.todense(sys, opdict)
    expected = np.kron(mat0, mat1)

    assert_dense_equal(M, expected)



# sNBO


def test_sNBO_todense_default():
    sys = make_simple_sys()

    op = 2.0 * sOP("a", 0)

    M = np.asarray(op.todense(sys))
    M0 = np.asarray(sOP("a", 0).todense(sys))

    assert_dense_equal(M, 2.0 * M0)


def test_sNBO_todense_custom_dict():
    sys = system_modes(1)
    sys[0] = generic_mode(2)

    opdict = operator_dictionary(1)

    mat = np.eye(2, dtype=np.complex128)
    opdict.insert(0, "I", site_operator(mat, optype="matrix", mode=0))

    op = coeff(3.0) * sOP("I", 0)

    M = op.todense(sys, opdict)

    assert_dense_equal(M, 3.0 * mat)



# sSOP


def test_sSOP_todense_default():
    sys = make_simple_sys()

    op = sSOP()
    op += sOP("a", 0)
    op += sOP("adag", 1)

    M = op.todense(sys)

    M_expected = (
        np.asarray(sOP("a", 0).todense(sys))
        + np.asarray(sOP("adag", 1).todense(sys))
    )

    assert_dense_equal(M, M_expected)


def test_sSOP_todense_custom_dict():
    sys = system_modes(1)
    sys[0] = generic_mode(2)

    opdict = operator_dictionary(1)

    mat0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    mat1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)

    opdict.insert(0, "p0", site_operator(mat0, optype="matrix", mode=0))
    opdict.insert(0, "p1", site_operator(mat1, optype="matrix", mode=0))

    op = sSOP()
    op += sOP("p0", 0)
    op += sOP("p1", 0)

    M = op.todense(sys, opdict)

    assert_dense_equal(M, mat0 + mat1)



# SOP


def test_SOP_todense_default():
    sys = make_simple_sys()

    H = SOP(2)
    H += sOP("a", 0)
    H += sOP("adag", 1)

    M = H.todense(sys)

    expected = (
        np.asarray(sOP("a", 0).todense(sys))
        + np.asarray(sOP("adag", 1).todense(sys))
    )

    assert_dense_equal(M, expected)


def test_SOP_todense_custom_dict():
    sys = system_modes(1)
    sys[0] = generic_mode(2)

    opdict = operator_dictionary(1)

    mat = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    opdict.insert(0, "x", site_operator(mat, optype="matrix", mode=0))

    H = SOP(1)
    H += sOP("x", 0)

    M = H.todense(sys, opdict)

    assert_dense_equal(M, mat)