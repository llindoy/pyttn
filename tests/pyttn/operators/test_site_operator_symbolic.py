from pyttn import site_operator, sOP
from pyttn import system_modes, boson_mode, fermion_mode, spin_mode, tls_mode
import pytest
import numpy as np
import os


os.environ["OMP_NUM_THREADS"] = "1"


def test_fermionic():
 # test whether it works when we are dealing with a single mode
    def test_1(label, mat):
        sysinf = system_modes(1)
        sysinf[0] = fermion_mode()

        H = sOP(label, 0)
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == 2
        assert siteop.mode == 0

        assert np.allclose(np.array(siteop.todense()), mat)

    # test whether it works when we are dealing with a single mode
    def test_2(label, mat, mi):
        mi = mi % 3
        sysinf = system_modes(1)
        sysinf[0] = [fermion_mode(), fermion_mode(), fermion_mode()]

        Id = np.identity(2)

        if mi == 0:
            mat = np.kron(mat, np.kron(Id, Id))
        elif mi == 1:
            mat = np.kron(Id, np.kron(mat, Id))
        elif mi == 2:
            mat = np.kron(Id, np.kron(Id, mat))

        H = sOP(label, mi)
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == 8
        assert siteop.mode == 0

    def a():
        return np.array([[0, 1], [0, 0]], dtype=np.complex128)

    def n():
        return np.diag(np.arange(2))

    amat = a()
    nmat = n()

    # test whether the boson operators works when as the only mode
    labels = ["a", "f", "c"]
    for label in labels:
        test_1(label, amat)

    labels = ["adag", "cdag", "fdag", "ad", "cd", "fd"]
    for label in labels:
        test_1(label, amat.T)

    labels = ["n", "adaga", "cdagc", "fdagf", "ada", "cdc", "fdf"]
    for label in labels:
        test_1(label, nmat)

    test_1("v", np.identity(2)-nmat)
    test_1("jw", np.array([[1, 0], [0, -1]]))

    for mi in range(3):
        test_2("a", amat, mi)
        test_2("adag", amat.T, mi)
        test_2("n", nmat, mi)
        test_2("v", np.identity(2)-nmat, mi)
        test_2("jw", np.array([[1, 0], [0, -1]]), mi)


