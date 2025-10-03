import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from pyttn import matrix_element, ntreeBuilder, Op, ttn

import pytest


@pytest.mark.parametrize(
    "N, order",
    [
        (3,1),
        (16,1),
        (20,1),
        (32,1),
        (3,2),       
        (16,2),
        (20,2),
        (32,2),
        (3,3),
        (16,3),
        (20,3),
        (32,3),
    ],
)
def test_GHz_prep(N, order):
    chi0 = 1
    chi = 16

    dims = [2 for i in range(N)]
    topo = None
    capacity = None
    if order == 1:
        topo = ntreeBuilder.mps_tree(dims, chi0)
        capacity = ntreeBuilder.mps_tree(dims, chi)
    else:
        topo = ntreeBuilder.mlmctdh_tree(dims, order, chi0)
        capacity = ntreeBuilder.mlmctdh_tree(dims, order, chi)
    A = ttn(topo, capacity, dtype=np.complex128)

    A.set_state([0 for i in range(N)])

    cnot_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
    Hmat = 1.0/np.sqrt(2.0)*np.array([[1, 1], [1, -1]], dtype=np.complex128)

    zeros = ttn(topo, dtype=np.complex128)
    zeros.set_state([0 for i in range(N)])
    ones = ttn(topo, dtype=np.complex128)
    ones.set_state([1 for i in range(N)])

    mel = matrix_element(A)

    A@=Op(Hmat, [0], [2])
    for i in range(N-1):
        A@=Op(cnot_mat, [i, i+1], [2, 2])

    assert pytest.approx(mel(zeros, A), 1e-8) == 1.0/np.sqrt(2)
    assert pytest.approx(mel(ones, A), 1e-8) == 1.0/np.sqrt(2)
