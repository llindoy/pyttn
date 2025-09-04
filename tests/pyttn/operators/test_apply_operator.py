import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from pyttn import matrix_element
from pyttn import sop_operator



import pytest

@pytest.mark.parametrize(
    "op, a, b, expected_result",
    [
        ("Sz0", "ttn_1", "ttn_1", 1),
        ("Sz0", "ttn_2", "ttn_1", 0),
        ("Sz0", "ttn_3", "ttn_1", 0),
        ("Sz0", "ttn_1", "ttn_2", 0),
        ("Sz0", "ttn_2", "ttn_2", 1),
        ("Sz0", "ttn_3", "ttn_2", 2),
        ("Sz0", "ttn_1", "ttn_3", 0),
        ("Sz0", "ttn_2", "ttn_3", 2),
        ("Sz0", "ttn_3", "ttn_3", 4),
        ("Sz6", "ttn_1", "ttn_1", 1),
        ("Sz6", "ttn_2", "ttn_1", 0),
        ("Sz6", "ttn_3", "ttn_1", 0),
        ("Sz6", "ttn_1", "ttn_2", 0),
        ("Sz6", "ttn_2", "ttn_2", -1),
        ("Sz6", "ttn_3", "ttn_2", -2),
        ("Sz6", "ttn_1", "ttn_3", 0),
        ("Sz6", "ttn_2", "ttn_3", -2),
        ("Sz6", "ttn_3", "ttn_3", -4),
        ("Sx0", "ttn_1", "ttn_1", 0),
        ("Sx0", "ttn_2", "ttn_1", 0),
        ("Sx0", "ttn_3", "ttn_1", 0),
        ("Sx0", "ttn_1", "ttn_2", 0),
        ("Sx0", "ttn_2", "ttn_2", 0),
        ("Sx0", "ttn_3", "ttn_2", 0),
        ("Sx0", "ttn_1", "ttn_3", 0),
        ("Sx0", "ttn_2", "ttn_3", 0),
        ("Sx0", "ttn_3", "ttn_3", 0),
        ("Sx_prod", "ttn_1", "ttn_1", 0),
        ("Sx_prod", "ttn_2", "ttn_1", 0),
        ("Sx_prod", "ttn_3", "ttn_1", 0),
        ("Sx_prod", "ttn_1", "ttn_2", 0),
        ("Sx_prod", "ttn_2", "ttn_2", 0),
        ("Sx_prod", "ttn_3", "ttn_2", 0),
        ("Sx_prod", "ttn_1", "ttn_3", 0),
        ("Sx_prod", "ttn_2", "ttn_3", 0),
        ("Sx_prod", "ttn_3", "ttn_3", 0),
    ],
)
def test_apply_pops(request, op, a, b, expected_result):
    op = request.getfixturevalue(op)
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    mel = matrix_element(A, nbuffers=1)
    C = op@A
    res = np.real(mel(C, B))

    assert res == pytest.approx(expected_result, abs=1e-8)


@pytest.mark.parametrize(
    "op, a, b, expected_result",
    [
        ("Sztot", "ttn_1", "ttn_1", 16),
        ("Sztot", "ttn_2", "ttn_1", 0),
        ("Sztot", "ttn_3", "ttn_1", 0),
        ("Sztot", "ttn_1", "ttn_2", 0),
        ("Sztot", "ttn_2", "ttn_2", -2),
        ("Sztot", "ttn_3", "ttn_2", -4),
        ("Sztot", "ttn_1", "ttn_3", 0),
        ("Sztot", "ttn_2", "ttn_3", -4),
        ("Sztot", "ttn_3", "ttn_3", -8),
        ("Stot", "ttn_1", "ttn_1", 16 * 16),
        ("Stot", "ttn_2", "ttn_1", 0),
        ("Stot", "ttn_3", "ttn_1", 0),
        ("Stot", "ttn_1", "ttn_2", 0),
        ("Stot", "ttn_2", "ttn_2", 4),
        ("Stot", "ttn_3", "ttn_2", 8),
        ("Stot", "ttn_1", "ttn_3", 0),
        ("Stot", "ttn_2", "ttn_3", 8),
        ("Stot", "ttn_3", "ttn_3", 16),
        ("H", "ttn_4", "ttn_4", -1.2510242438*16)

    ],
)
def test_apply_sop(request, op, a, b, expected_result):
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    op, sysinf = request.getfixturevalue(op)
    op = sop_operator(op, A, sysinf)
    mel = matrix_element(A, nbuffers=1)
    C = op@A
    res = np.real(mel(C, B))

    assert res == pytest.approx(expected_result, abs=1e-8)



#now test the application of Op objects onto the mode
@pytest.mark.parametrize(
    "op, a, b, expected_result",
    [
        ("Sz0_op", "ttn_1", "ttn_1", 1),
        ("Sz0_op", "ttn_2", "ttn_1", 0),
        ("Sz0_op", "ttn_3", "ttn_1", 0),
        ("Sz0_op", "ttn_1", "ttn_2", 0),
        ("Sz0_op", "ttn_2", "ttn_2", 1),
        ("Sz0_op", "ttn_3", "ttn_2", 2),
        ("Sz0_op", "ttn_1", "ttn_3", 0),
        ("Sz0_op", "ttn_2", "ttn_3", 2),
        ("Sz0_op", "ttn_3", "ttn_3", 4),
        ("Sz6_op", "ttn_1", "ttn_1", 1),
        ("Sz6_op", "ttn_2", "ttn_1", 0),
        ("Sz6_op", "ttn_3", "ttn_1", 0),
        ("Sz6_op", "ttn_1", "ttn_2", 0),
        ("Sz6_op", "ttn_2", "ttn_2", -1),
        ("Sz6_op", "ttn_3", "ttn_2", -2),
        ("Sz6_op", "ttn_1", "ttn_3", 0),
        ("Sz6_op", "ttn_2", "ttn_3", -2),
        ("Sz6_op", "ttn_3", "ttn_3", -4),
        ("Sx0_op", "ttn_1", "ttn_1", 0),
        ("Sx0_op", "ttn_2", "ttn_1", 0),
        ("Sx0_op", "ttn_3", "ttn_1", 0),
        ("Sx0_op", "ttn_1", "ttn_2", 0),
        ("Sx0_op", "ttn_2", "ttn_2", 0),
        ("Sx0_op", "ttn_3", "ttn_2", 0),
        ("Sx0_op", "ttn_1", "ttn_3", 0),
        ("Sx0_op", "ttn_2", "ttn_3", 0),
        ("Sx0_op", "ttn_3", "ttn_3", 0),
    ],
)
def test_apply_op_1d(request, op, a, b, expected_result):
    op = request.getfixturevalue(op)
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    mel = matrix_element(A, nbuffers=1)
    C = op@A
    res = np.real(mel(C, B))

    assert res == pytest.approx(expected_result, abs=1e-8)


#now test the application of Op objects onto the mode
@pytest.mark.parametrize(
    "op, a, b, expected_result",
    [
        ("Sz01_op", "ttn_1", "ttn_1", 1),
    ],
)
def test_apply_op_2d(request, op, a, b, expected_result):
    op = request.getfixturevalue(op)
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    mel = matrix_element(A, nbuffers=1)
    C = op@A
    res = np.real(mel(C, B))

    assert res == pytest.approx(expected_result, abs=1e-8)