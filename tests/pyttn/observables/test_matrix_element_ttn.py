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

@pytest.mark.parametrize("state, expected_result", [("ttn_1", 1),("ttn_2", 1),("ttn_3", 4), ("ttn_4", 1)])
def test_matrix_element_norm(request, state, expected_result):
    A = request.getfixturevalue(state)
    mel = matrix_element(A, nbuffers=1)
    res = mel(A)

    assert pytest.approx(res, 1e-6) == expected_result

@pytest.mark.parametrize("a, b, expected_result", [ 
    ("ttn_1", "ttn_1", 1),("ttn_2", "ttn_1", 0),("ttn_3", "ttn_1", 0),
    ("ttn_1", "ttn_2", 0),("ttn_2", "ttn_2", 1),("ttn_3", "ttn_2", 2),
    ("ttn_1", "ttn_3", 0),("ttn_2", "ttn_3", 2),("ttn_3", "ttn_3", 4)

    ])
def test_matrix_element_overlap(request, a, b, expected_result):
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(A, B))

    assert pytest.approx(res, 1e-6) == expected_result

@pytest.mark.parametrize("op, state, expected_result", [ 
    ("Sz0", "ttn_1", 1),("Sz0", "ttn_2", 1),("Sz0", "ttn_3", 4), ("Sz0", "ttn_4", 0),
    ("Sz6", "ttn_1", 1),("Sz6", "ttn_2", -1),("Sz6", "ttn_3", -4), ("Sz6", "ttn_4", 0),

    ("Sx0", "ttn_1", 0),("Sx0", "ttn_2", 0),("Sx0", "ttn_3", 0),
    ("Sx6", "ttn_1", 0),("Sx6", "ttn_2", 0),("Sx6", "ttn_3", 0),

    ("Sz_prod", "ttn_1", 1),("Sz_prod", "ttn_2", -1),("Sz_prod", "ttn_3", -4),
    ("Sx_prod", "ttn_1", 0),("Sx_prod", "ttn_2", 0),("Sx_prod", "ttn_3", 0)
    ])
def test_matrix_element_expect(request, op, state, expected_result):
    A = request.getfixturevalue(state)
    op = request.getfixturevalue(op)
    mel = matrix_element(A, nbuffers=1)
    res = mel(op, A)

    assert pytest.approx(res, 1e-6) == expected_result

@pytest.mark.parametrize("op, state, expected_result", [ 
    ("Sztot", "ttn_1", 16),("Sztot", "ttn_2", -2), ("Sztot", "ttn_4", 0),
    ("Stot", "ttn_1", 16*16),("Stot", "ttn_2", 4)
    ])
def test_matrix_element_expect_sop(request, op, state, expected_result):
    A = request.getfixturevalue(state)
    op, sysinf = request.getfixturevalue(op)
    op = sop_operator(op, A, sysinf)
    mel = matrix_element(A, nbuffers=1)
    res = mel(op, A)

    assert pytest.approx(res, 1e-6) == expected_result

@pytest.mark.parametrize("op, a, b, expected_result", [ 
    ("Sz0", "ttn_1", "ttn_1", 1),("Sz0", "ttn_2", "ttn_1", 0),("Sz0", "ttn_3", "ttn_1", 0),
    ("Sz0", "ttn_1", "ttn_2", 0),("Sz0", "ttn_2", "ttn_2", 1),("Sz0", "ttn_3", "ttn_2", 2),
    ("Sz0", "ttn_1", "ttn_3", 0),("Sz0", "ttn_2", "ttn_3", 2),("Sz0", "ttn_3", "ttn_3", 4),

    ("Sz6", "ttn_1", "ttn_1", 1),("Sz6", "ttn_2", "ttn_1", 0),("Sz6", "ttn_3", "ttn_1", 0),
    ("Sz6", "ttn_1", "ttn_2", 0),("Sz6", "ttn_2", "ttn_2", -1),("Sz6", "ttn_3", "ttn_2", -2),
    ("Sz6", "ttn_1", "ttn_3", 0),("Sz6", "ttn_2", "ttn_3", -2),("Sz6", "ttn_3", "ttn_3", -4),

    ("Sx0", "ttn_1", "ttn_1", 0),("Sx0", "ttn_2", "ttn_1", 0),("Sx0", "ttn_3", "ttn_1", 0),
    ("Sx0", "ttn_1", "ttn_2", 0),("Sx0", "ttn_2", "ttn_2", 0),("Sx0", "ttn_3", "ttn_2", 0),
    ("Sx0", "ttn_1", "ttn_3", 0),("Sx0", "ttn_2", "ttn_3", 0),("Sx0", "ttn_3", "ttn_3", 0),

    ("Sx_prod", "ttn_1", "ttn_1", 0),("Sx_prod", "ttn_2", "ttn_1", 0),("Sx_prod", "ttn_3", "ttn_1", 0),
    ("Sx_prod", "ttn_1", "ttn_2", 0),("Sx_prod", "ttn_2", "ttn_2", 0),("Sx_prod", "ttn_3", "ttn_2", 0),
    ("Sx_prod", "ttn_1", "ttn_3", 0),("Sx_prod", "ttn_2", "ttn_3", 0),("Sx_prod", "ttn_3", "ttn_3", 0)
    ])
def test_matrix_element_matel(request, op, a, b, expected_result):
    op = request.getfixturevalue(op)
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(op, A, B))

    assert pytest.approx(res, 1e-6) == expected_result

@pytest.mark.parametrize("op, a, b, expected_result", [ 
    ("Sztot", "ttn_1", "ttn_1", 16),("Sztot", "ttn_2", "ttn_1", 0),("Sztot", "ttn_3", "ttn_1", 0),
    ("Sztot", "ttn_1", "ttn_2", 0),("Sztot", "ttn_2", "ttn_2", -2),("Sztot", "ttn_3", "ttn_2", -4),
    ("Sztot", "ttn_1", "ttn_3", 0),("Sztot", "ttn_2", "ttn_3", -4),("Sztot", "ttn_3", "ttn_3", -8),

    ("Stot", "ttn_1", "ttn_1", 16*16),("Stot", "ttn_2", "ttn_1", 0),("Stot", "ttn_3", "ttn_1", 0),
    ("Stot", "ttn_1", "ttn_2", 0),("Stot", "ttn_2", "ttn_2", 4),("Stot", "ttn_3", "ttn_2", 8),
    ("Stot", "ttn_1", "ttn_3", 0),("Stot", "ttn_2", "ttn_3", 8),("Stot", "ttn_3", "ttn_3", 16)
    ])
def test_matrix_element_matel_sop(request, op, a, b, expected_result):
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    op, sysinf = request.getfixturevalue(op)
    op = sop_operator(op, A, sysinf)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(op, A, B))

    assert pytest.approx(res, 1e-6) == expected_result
