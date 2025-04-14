
import numpy as np
from scipy.sparse import csr_matrix as spcsr
import os

os.environ["OMP_NUM_THREADS"] = "1"

from pyttn.linalg import matrix
from pyttn.linalg import csr_matrix
from pyttn.linalg import available_backends

import pytest
    
@pytest.mark.parametrize("backend", available_backends())
def test_csr_init_1(backend):
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    mdense = spcsr((data, indices, indptr), shape=(3,3)).toarray()

    def run_tests(dtype):
        mat = csr_matrix(data, indices, indptr, dtype=dtype, ncols=3, backend=backend)
        m2 = matrix(np.identity(3), dtype=dtype, backend=backend)

        if dtype == np.complex128:
            assert mat.complex_dtype()
        else:
            assert not mat.complex_dtype()

        m3 = mat@m2
        assert(np.allclose(np.array(m3), mdense))

    run_tests(np.float64)
    run_tests(np.complex128)

@pytest.mark.parametrize("backend", available_backends())
def test_csr_init_2(backend):
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    spmat = spcsr((data, indices, indptr), shape=(3,3))
    mdense = spmat.toarray()

    def run_tests(dtype):
        mat = csr_matrix(spmat, dtype=dtype, backend=backend)
        m2 = matrix(np.identity(3), dtype=dtype, backend=backend)

        if dtype == np.complex128:
            assert mat.complex_dtype()
        else:
            assert not mat.complex_dtype()

        m3 = mat@m2
        assert(np.allclose(np.array(m3), mdense))

    run_tests(np.float64)
    run_tests(np.complex128)

@pytest.mark.parametrize("backend", available_backends())
def test_csr_init_3(backend):
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    coo = [(0, 0, 1.0), (0, 2, 2.0), (1, 2, 3.0), (2, 0, 4.0), (2, 1, 5.0), (2, 2, 6.0)]
    spmat = spcsr((data, indices, indptr), shape=(3,3))
    mdense = spmat.toarray()

    def run_tests(dtype):
        mat = csr_matrix(coo, dtype=dtype, nrows=3, ncols=3, backend=backend)
        m2 = matrix(np.identity(3), dtype=dtype, backend=backend)


        if dtype == np.complex128:
            assert mat.complex_dtype()
        else:
            assert not mat.complex_dtype()

        matb = csr_matrix(mat, dtype=np.complex128, backend=backend)
        assert matb.complex_dtype()

        m3 = mat@m2
        assert(np.allclose(np.array(m3), mdense))

    run_tests(np.float64)
    run_tests(np.complex128)