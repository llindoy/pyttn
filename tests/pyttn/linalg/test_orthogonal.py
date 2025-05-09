import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1"

from pyttn.linalg import matrix
from pyttn.linalg import available_backends
from pyttn.linalg import random_engine
from pyttn.linalg import orthogonal_vector
import pytest

@pytest.mark.parametrize("M, N", [(2, 2), (8, 8), (12, 15), (3, 5)])
def test_fill_random(M, N) -> None:
    for backend in available_backends():

        def run_test(dtype) -> None:

            m2 = matrix(np.zeros((M, N)), dtype=dtype, backend=backend)
            rng = random_engine(backend)
            orthogonal_vector.fill_random(m2, rng)

            m1 = np.array(m2)
            for i in range(min(M, N)):
                for j in range(min(M, N)):
                    if i == j:
                        assert pytest.approx(np.abs(np.dot(np.conj(m1[i, :]), m1[j, :])), 1e-8) == 1
                    elif i != j:
                        assert pytest.approx(np.abs(np.dot(np.conj(m1[i, :]), m1[j, :])), 1e-8) == 0

        run_test(np.float64)
        run_test(np.complex128)


@pytest.mark.parametrize("M, N", [(2, 2), (8, 8), (11, 13), (12, 15), (3, 5)])
def test_pad_random(M, N) -> None:
    for backend in available_backends():

        def run_test(dtype) -> None:
            m2 = matrix(np.zeros((M, N)), dtype=dtype, backend=backend)
            rng = random_engine(backend)
            orthogonal_vector.fill_random(m2, rng)

            orthogonal_vector.pad_random(m2, 1, rng)

            m1 = np.array(m2)
            for i in range(min(M, N)):
                for j in range(min(M, N)):
                    if i == j:
                        assert pytest.approx(np.abs(np.dot(np.conj(m1[i, :]), m1[j, :])), 1e-8) == 1
                    elif i != j:
                        assert pytest.approx(np.abs(np.dot(np.conj(m1[i, :]), m1[j, :])), 1e-8) == 0

        run_test(np.float64)
        run_test(np.complex128)


@pytest.mark.parametrize("M, N", [(11, 13), (12, 15), (3, 5)])
def test_generate(M, N) -> None:
    for backend in available_backends():

        def run_test(dtype) -> None:
            m2 = matrix(np.zeros((M, N)), dtype=dtype, backend=backend)
            rng = random_engine(backend)
            orthogonal_vector.fill_random(m2, rng)

            v = orthogonal_vector.generate(m2, rng)

            m1 = np.array(m2)
            v1 = np.array(v)
            for i in range(min(M, N)):
                assert pytest.approx(np.abs(np.dot(np.conj(v1), m1[i, :])), 1e-8) == 0

        run_test(np.float64)
        run_test(np.complex128)
