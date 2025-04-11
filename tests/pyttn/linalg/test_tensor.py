
from pyttn.linalg.tensorExt import vector, matrix, tensor_3, tensor_4, tensor
import pytest
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1"


@pytest.mark.parametrize(
    "N",
    [2, 8, 13, 12, 5])
def test_vector(N):
    def run_test(dtype):
        m1 = np.random.uniform(size=(N))
        m2 = tensor(m1, dtype=dtype)

        m3 = np.array(m2)
        assert np.allclose(m1, m3)

        if dtype == np.complex128:
            assert m2.complex_dtype()
        else:
            assert not m2.complex_dtype()

        assert m2.ndim() == 1
        assert m2.shape(0) == N

        m3 = vector(m1, dtype=np.complex128)
        assert m3.complex_dtype()
        assert m3.ndim() == 1
        assert m3.shape(0) == N

        m3 = vector(m2, dtype=np.complex128)
        assert m3.complex_dtype()
        assert m3.ndim() == 1
        assert m3.shape(0) == N

        m2.clear()
        assert m2.shape(0) == 0

    run_test(np.float64)
    run_test(np.complex128)


"""
   .def("set_subblock", [](ttype &o, py::buffer &b)
"""


@pytest.mark.parametrize(
    "M, N",
    [(2, 2), (8, 8), (13, 11), (12, 15), (5, 3)])
def test_matrix(M, N):
    def run_test(dtype):
        m1 = np.random.uniform(size=(M, N))
        m2 = tensor(m1, dtype=dtype)
        m3 = np.array(m2)
        assert np.allclose(m1, m3)

        if dtype == np.complex128:
            assert m2.complex_dtype()
        else:
            assert not m2.complex_dtype()

        m5 = m2.transpose([1, 0])@m2
        assert np.allclose(np.array(m5), m1.T@m1)

        assert m2.ndim() == 2
        assert m2.shape(0) == M
        assert m2.shape(1) == N

        m4 = m2.transpose([1, 0])
        assert m4.shape(0) == N
        assert m4.shape(1) == M
        assert np.allclose(np.array(m4), m1.T)

        m3 = matrix(m1, dtype=np.complex128)
        assert m3.complex_dtype()
        assert m3.ndim() == 2
        assert m3.shape(0) == M
        assert m3.shape(1) == N

        m3 = matrix(m2, dtype=np.complex128)
        assert m3.complex_dtype()
        assert m3.ndim() == 2
        assert m3.shape(0) == M
        assert m3.shape(1) == N

        m2.clear()
        assert m2.shape(0) == 0
        assert m2.shape(1) == 0

    run_test(np.float64)
    run_test(np.complex128)


@pytest.mark.parametrize(
    "M, N, L",
    [(2, 2, 4), (8, 8, 8), (13, 11, 6), (12, 15, 4), (5, 3, 2)])
def test_tensor_3(M, N, L):
    def run_test(dtype):
        m1 = np.random.uniform(size=(M, N, L))
        m2 = tensor(m1, dtype=dtype)
        m3 = np.array(m2)
        assert np.allclose(m1, m3)

        if dtype == np.complex128:
            assert m2.complex_dtype()
        else:
            assert not m2.complex_dtype()

        assert m2.ndim() == 3
        assert m2.shape(0) == M
        assert m2.shape(1) == N
        assert m2.shape(2) == L

        m4 = m2.transpose([1, 0, 2])
        assert m4.shape(0) == N
        assert m4.shape(1) == M
        assert m4.shape(2) == L
        assert np.allclose(np.array(m4), np.transpose(m1, [1, 0, 2]))

        m4 = m2.transpose([2, 0, 1])
        assert m4.shape(0) == L
        assert m4.shape(1) == M
        assert m4.shape(2) == N
        assert np.allclose(np.array(m4), np.transpose(m1, [2, 0, 1]))

        m3 = tensor_3(m1, dtype=np.complex128)
        assert m3.complex_dtype()
        assert m3.ndim() == 3
        assert m3.shape(0) == M
        assert m3.shape(1) == N
        assert m3.shape(2) == L

        m2.clear()
        assert m2.shape(0) == 0
        assert m2.shape(1) == 0
        assert m2.shape(2) == 0

    run_test(np.float64)
    run_test(np.complex128)


@pytest.mark.parametrize(
    "M, N, L, P",
    [(2, 2, 4, 2), (8, 8, 8, 8), (13, 11, 6, 3), (12, 15, 4, 5), (5, 3, 2, 4)])
def test_tensor_4(M, N, L, P):
    def run_test(dtype):
        m1 = np.random.uniform(size=(M, N, L, P))
        m2 = tensor(m1, dtype=dtype)
        m3 = np.array(m2)
        assert np.allclose(m1, m3)

        if dtype == np.complex128:
            assert m2.complex_dtype()
        else:
            assert not m2.complex_dtype()

        assert m2.ndim() == 4
        assert m2.shape(0) == M
        assert m2.shape(1) == N
        assert m2.shape(2) == L
        assert m2.shape(3) == P

        m4 = m2.transpose([1, 0, 2, 3])
        assert m4.shape(0) == N
        assert m4.shape(1) == M
        assert m4.shape(2) == L
        assert m4.shape(3) == P

        assert np.allclose(np.array(m4), np.transpose(m1, [1, 0, 2, 3]))

        m4 = m2.transpose([2, 0, 3, 1])
        assert m4.shape(0) == L
        assert m4.shape(1) == M
        assert m4.shape(2) == P
        assert m4.shape(3) == N
        assert np.allclose(np.array(m4), np.transpose(m1, [2, 0, 3, 1]))

        m3 = tensor_4(m1, dtype=np.complex128)
        assert m3.complex_dtype()
        assert m3.ndim() == 4
        assert m3.shape(0) == M
        assert m3.shape(1) == N
        assert m3.shape(2) == L
        assert m3.shape(3) == P

        m2.clear()
        assert m2.shape(0) == 0
        assert m2.shape(1) == 0
        assert m2.shape(2) == 0
        assert m2.shape(3) == 0

    run_test(np.float64)
    run_test(np.complex128)
