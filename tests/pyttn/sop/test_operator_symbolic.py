import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest
from pyttn import system_modes, tls_mode, convert_to_dense
from pyttn import sOP, SOP



sxmat = np.array([[0, 1], [1, 0]])
symat = np.array([[0, -1.0j], [1.0j, 0]])
szmat = np.array([[1, 0], [0, -1]])

spmat = (sxmat + 1.0j * symat) / 2.0
smmat = (sxmat - 1.0j * symat) / 2.0

s00 = np.zeros((2, 2))
s01 = np.zeros((2, 2))
s10 = np.zeros((2, 2))
s11 = np.zeros((2, 2))
s00[0, 0] = 1
s01[0, 1] = 1
s10[1, 0] = 1
s11[1, 1] = 1

ops = [
    ["s+", spmat],
    ["s-", smmat],
    ["sx", sxmat],
    ["sy", symat],
    ["sz", szmat],
    ["|0><0|", s00],
    ["|0><1|", s01],
    ["|1><0|", s10],
    ["|1><1|", s11],
]

two_body_ops = []
for op1 in ops:
    for op2 in ops:
        two_body_ops.append([op1, op2])


@pytest.mark.parametrize("op", ops)
def test_sOP(op):
    # test whether it works when we are dealing with a single mode
    def test_1(label, mat):
        sysinf = system_modes(1)
        sysinf[0] = tls_mode()

        H = sOP(label, 0)
        mat2 = convert_to_dense(H, sysinf)
        assert np.allclose(np.array(mat2), mat)

    # test whether it works when we are dealing with a single mode
    def test_2(label, mat, mi):
        mi = mi % 3
        sysinf = system_modes(1)
        sysinf[0] = [tls_mode(), tls_mode(), tls_mode()]

        Id = np.identity(2)

        if mi == 0:
            mat = np.kron(mat, np.kron(Id, Id))
        elif mi == 1:
            mat = np.kron(Id, np.kron(mat, Id))
        elif mi == 2:
            mat = np.kron(Id, np.kron(Id, mat))

        H = sOP(label, mi)
        mat2 = convert_to_dense(H, sysinf)
        assert np.allclose(np.array(mat2), mat)

    test_1(op[0], op[1])

    # test whether the boson operators works when created as a composite mode at various positions
    for mi in range(3):
        test_2(op[0], op[1], mi)


@pytest.mark.parametrize("op1, op2", two_body_ops)
def test_sPOP(op1, op2):
    # test whether it works when we are dealing with a single mode
    def test_1(label, label2, mat, mat2):
        sysinf = system_modes(1)
        sysinf[0] = [tls_mode(), tls_mode()]

        H = sOP(label, 0) * sOP(label2, 1)
        conv = convert_to_dense(H, sysinf)
        assert np.allclose(np.array(conv), np.kron(mat, mat2))

    for op1 in ops:
        for op2 in ops:
            test_1(op1[0], op2[0], op1[1], op2[1])


@pytest.mark.parametrize("op1, op2", two_body_ops)
def test_sNBO(op1, op2):
    # test whether it works when we are dealing with a single mode
    def test_1(label, label2, mat, mat2):
        sysinf = system_modes(1)
        sysinf[0] = [tls_mode(), tls_mode()]

        H = 2.0 * sOP(label, 0) * sOP(label2, 1)
        conv = convert_to_dense(H, sysinf)

        assert np.allclose(np.array(conv), 2.0 * np.kron(mat, mat2))

    for op1 in ops:
        for op2 in ops:
            test_1(op1[0], op2[0], op1[1], op2[1])  #


@pytest.mark.parametrize("op1, op2", two_body_ops)
def test_sSOP(op1, op2):
    # test whether it works when we are dealing with a single mode
    def test_1(label, label2, mat, mat2):
        sysinf = system_modes(1)
        sysinf[0] = [tls_mode(), tls_mode(), tls_mode()]

        H = (
            2.0 * sOP(label, 0) * sOP(label2, 1)
            + 1.0j * sOP(label, 0) * sOP(label2, 2)
            + 3 * sOP(label, 2)
        )
        conv = convert_to_dense(H, sysinf)

        id = np.identity(2)
        res = (
            2.0 * np.kron(mat, np.kron(mat2, id))
            + 1.0j * np.kron(mat, np.kron(id, mat2))
            + 3.0 * np.kron(id, np.kron(id, mat))
        )
        assert np.allclose(np.array(conv), res)

    for op1 in ops:
        for op2 in ops:
            test_1(op1[0], op2[0], op1[1], op2[1])  #


@pytest.mark.parametrize("op1, op2", two_body_ops)
def test_SOP(op1, op2):
    # test whether it works when we are dealing with a single mode
    def test_1(label, label2, mat, mat2):
        sysinf = system_modes(1)
        sysinf[0] = [tls_mode(), tls_mode(), tls_mode()]

        H = SOP(3)
        H = (
            2.0 * sOP(label, 0) * sOP(label2, 1)
            + 1.0j * sOP(label, 0) * sOP(label2, 2)
            + 3 * sOP(label, 2)
        )
        conv = convert_to_dense(H, sysinf)

        id = np.identity(2)
        res = (
            2.0 * np.kron(mat, np.kron(mat2, id))
            + 1.0j * np.kron(mat, np.kron(id, mat2))
            + 3.0 * np.kron(id, np.kron(id, mat))
        )
        assert np.allclose(np.array(conv), res)

    for op1 in ops:
        for op2 in ops:
            test_1(op1[0], op2[0], op1[1], op2[1])  #
