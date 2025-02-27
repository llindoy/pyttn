import os
os.environ['OMP_NUM_THREADS']='1'

import numpy as np
import sys
import copy
import h5py

from pyttn import system_modes, tls_mode, SOP, sOP, ntreeBuilder
from pyttn import ttn, matrix_element, dmrg
from pyttn import site_operator, product_operator, sop_operator

import pytest

@pytest.mark.parametrize("state, expected_result", [("ms_mps_1", 1),("ms_mps_2", 2),("ms_mps_3", 1),("ms_mps_4", 2)])
def test_matrix_element_norm(request, state, expected_result):
    A = request.getfixturevalue(state)
    mel = matrix_element(A, nbuffers=1)
    res = mel(A)

    assert res == pytest.approx(expected_result, abs=1e-8)

@pytest.mark.parametrize("a, b, expected_result", [ 
    ("ms_mps_1", "ms_mps_1", 1),("ms_mps_2", "ms_mps_1", 1/np.sqrt(2)),
    ("ms_mps_1", "ms_mps_2", 1/np.sqrt(2)),("ms_mps_2", "ms_mps_2", 2)
    ])
def test_matrix_element_overlap(request, a, b, expected_result):
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(A, B))

    assert res == pytest.approx(expected_result, abs=1e-8)


@pytest.mark.parametrize("op, state, expected_result", [ 
    ("Sz0", "ms_mps_1", 1),("Sz0", "ms_mps_2", 0), ("Sz0", "ms_mps_3", 0),
    ("Sz6", "ms_mps_1", 1),("Sz6", "ms_mps_2", 2), ("Sz6", "ms_mps_3", 0),

    ("Sx0", "ms_mps_1", 0),("Sx0", "ms_mps_2", 0),
    ("Sx6", "ms_mps_1", 0),("Sx6", "ms_mps_2", 0)
    ])
def test_matrix_element_expect(request, op, state, expected_result):
    A = request.getfixturevalue(state)
    op = request.getfixturevalue(op)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(op, A))

    assert res == pytest.approx(expected_result, abs=1e-8)


def test_matrix_element_ms_slice(request):
    A = request.getfixturevalue("ms_mps_4")
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(A.slice(0), A.slice(1)))

    assert res == pytest.approx(1, abs=1e-8)


def test_matrix_element_expect_ms_dmrg(request):
    A, op, E = request.getfixturevalue("tfim_ms_mps_H")
    mel = matrix_element(A, nbuffers=1)
    import copy
    B = copy.deepcopy(A)
    res = np.real(mel(op, A, B))/16
    res2 = np.real(mel(op, A))/16

    assert res == pytest.approx( -1.2510242438, abs=1e-8)
    assert res2 == pytest.approx( -1.2510242438, abs=1e-8)
