import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

from pyttn import matrix_element, sop_operator

import pytest


@pytest.mark.parametrize(
    "state, expected_result", [("mps_1", 1), ("mps_2", 1), ("mps_3", 4), ("mps_4", 1)]
)
def test_matrix_element_norm(request, state, expected_result):
    A = request.getfixturevalue(state)
    mel = matrix_element(A, nbuffers=1)
    res = mel(A)

    assert res == pytest.approx(expected_result, abs=1e-8)


@pytest.mark.parametrize(
    "a, b, expected_result",
    [
        ("mps_1", "mps_1", 1),
        ("mps_2", "mps_1", 0),
        ("mps_3", "mps_1", 0),
        ("mps_1", "mps_2", 0),
        ("mps_2", "mps_2", 1),
        ("mps_3", "mps_2", 2),
        ("mps_1", "mps_3", 0),
        ("mps_2", "mps_3", 2),
        ("mps_3", "mps_3", 4),
    ],
)
def test_matrix_element_overlap(request, a, b, expected_result):
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(A, B))

    assert res == pytest.approx(expected_result, abs=1e-8)


@pytest.mark.parametrize(
    "op, state, expected_result",
    [
        ("Sz0", "mps_1", 1),
        ("Sz0", "mps_2", 1),
        ("Sz0", "mps_3", 4),
        ("Sz0", "mps_4", 0),
        ("Sz6", "mps_1", 1),
        ("Sz6", "mps_2", -1),
        ("Sz6", "mps_3", -4),
        ("Sz6", "mps_4", 0),
        ("Sx0", "mps_1", 0),
        ("Sx0", "mps_2", 0),
        ("Sx0", "mps_3", 0),
        ("Sx6", "mps_1", 0),
        ("Sx6", "mps_2", 0),
        ("Sx6", "mps_3", 0),
        ("Sz_prod", "mps_1", 1),
        ("Sz_prod", "mps_2", -1),
        ("Sz_prod", "mps_3", -4),
        ("Sx_prod", "mps_1", 0),
        ("Sx_prod", "mps_2", 0),
        ("Sx_prod", "mps_3", 0),
    ],
)
def test_matrix_element_expect(request, op, state, expected_result):
    A = request.getfixturevalue(state)
    op = request.getfixturevalue(op)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(op, A))

    assert res == pytest.approx(expected_result, abs=1e-8)


@pytest.mark.parametrize(
    "op, state, expected_result",
    [
        ("Sztot", "mps_1", 16),
        ("Sztot", "mps_2", -2),
        ("Sztot", "mps_4", 0),
        ("Stot", "mps_1", 16 * 16),
        ("Stot", "mps_2", 4),
    ],
)
def test_matrix_element_expect_sop(request, op, state, expected_result):
    A = request.getfixturevalue(state)
    op, sysinf = request.getfixturevalue(op)
    op = sop_operator(op, A, sysinf)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(op, A))

    print(op, state, res, expected_result)

    assert res == pytest.approx(expected_result, abs=1e-8)


@pytest.mark.parametrize(
    "op, a, b, expected_result",
    [
        ("Sz0", "mps_1", "mps_1", 1),
        ("Sz0", "mps_2", "mps_1", 0),
        ("Sz0", "mps_3", "mps_1", 0),
        ("Sz0", "mps_1", "mps_2", 0),
        ("Sz0", "mps_2", "mps_2", 1),
        ("Sz0", "mps_3", "mps_2", 2),
        ("Sz0", "mps_1", "mps_3", 0),
        ("Sz0", "mps_2", "mps_3", 2),
        ("Sz0", "mps_3", "mps_3", 4),
        ("Sz6", "mps_1", "mps_1", 1),
        ("Sz6", "mps_2", "mps_1", 0),
        ("Sz6", "mps_3", "mps_1", 0),
        ("Sz6", "mps_1", "mps_2", 0),
        ("Sz6", "mps_2", "mps_2", -1),
        ("Sz6", "mps_3", "mps_2", -2),
        ("Sz6", "mps_1", "mps_3", 0),
        ("Sz6", "mps_2", "mps_3", -2),
        ("Sz6", "mps_3", "mps_3", -4),
        ("Sx0", "mps_1", "mps_1", 0),
        ("Sx0", "mps_2", "mps_1", 0),
        ("Sx0", "mps_3", "mps_1", 0),
        ("Sx0", "mps_1", "mps_2", 0),
        ("Sx0", "mps_2", "mps_2", 0),
        ("Sx0", "mps_3", "mps_2", 0),
        ("Sx0", "mps_1", "mps_3", 0),
        ("Sx0", "mps_2", "mps_3", 0),
        ("Sx0", "mps_3", "mps_3", 0),
        ("Sx_prod", "mps_1", "mps_1", 0),
        ("Sx_prod", "mps_2", "mps_1", 0),
        ("Sx_prod", "mps_3", "mps_1", 0),
        ("Sx_prod", "mps_1", "mps_2", 0),
        ("Sx_prod", "mps_2", "mps_2", 0),
        ("Sx_prod", "mps_3", "mps_2", 0),
        ("Sx_prod", "mps_1", "mps_3", 0),
        ("Sx_prod", "mps_2", "mps_3", 0),
        ("Sx_prod", "mps_3", "mps_3", 0),
    ],
)
def test_matrix_element_matel(request, op, a, b, expected_result):
    op = request.getfixturevalue(op)
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(op, A, B))

    assert res == pytest.approx(expected_result, abs=1e-8)


@pytest.mark.parametrize(
    "op, a, b, expected_result",
    [
        ("Sztot", "mps_1", "mps_1", 16),
        ("Sztot", "mps_2", "mps_1", 0),
        ("Sztot", "mps_3", "mps_1", 0),
        ("Sztot", "mps_1", "mps_2", 0),
        ("Sztot", "mps_2", "mps_2", -2),
        ("Sztot", "mps_3", "mps_2", -4),
        ("Sztot", "mps_1", "mps_3", 0),
        ("Sztot", "mps_2", "mps_3", -4),
        ("Sztot", "mps_3", "mps_3", -8),
        ("Stot", "mps_1", "mps_1", 16 * 16),
        ("Stot", "mps_2", "mps_1", 0),
        ("Stot", "mps_3", "mps_1", 0),
        ("Stot", "mps_1", "mps_2", 0),
        ("Stot", "mps_2", "mps_2", 4),
        ("Stot", "mps_3", "mps_2", 8),
        ("Stot", "mps_1", "mps_3", 0),
        ("Stot", "mps_2", "mps_3", 8),
        ("Stot", "mps_3", "mps_3", 16),
    ],
)
def test_matrix_element_matel_sop(request, op, a, b, expected_result):
    A = request.getfixturevalue(a)
    B = request.getfixturevalue(b)
    op, sysinf = request.getfixturevalue(op)
    op = sop_operator(op, A, sysinf)
    mel = matrix_element(A, nbuffers=1)
    res = np.real(mel(op, A, B))

    assert res == pytest.approx(expected_result, abs=1e-8)
