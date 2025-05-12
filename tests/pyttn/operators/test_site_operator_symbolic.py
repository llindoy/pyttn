import os
os.environ["OMP_NUM_THREADS"] = "1"

from pyttn import site_operator, sOP
from pyttn import system_modes, boson_mode, fermion_mode, spin_mode, tls_mode, nlevel_mode
import pytest
import numpy as np
import random



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

        mat2 = siteop.todense()
        assert np.allclose(np.array(mat2), mat)

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

        mat2 = siteop.todense()
        assert np.allclose(np.array(mat2), mat)

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


@pytest.mark.parametrize("N", [2, 4, 5, 7, 8])
def test_bosonic(N):
    # test whether it works when we are dealing with a single mode
    def test_1(label, mat, N):
        sysinf = system_modes(1)
        sysinf[0] = boson_mode(N)

        H = sOP(label, 0)
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == N
        assert siteop.mode == 0

        mat2 = siteop.todense()
        assert np.allclose(np.array(mat2), mat)

    # test whether it works when we are dealing with a single mode
    def test_2(label, mat, N, mi):
        mi = mi % 3
        sysinf = system_modes(1)
        sysinf[0] = [boson_mode(N), boson_mode(N), boson_mode(N)]

        Id = np.identity(N)

        if mi == 0:
            mat = np.kron(mat, np.kron(Id, Id))
        elif mi == 1:
            mat = np.kron(Id, np.kron(mat, Id))
        elif mi == 2:
            mat = np.kron(Id, np.kron(Id, mat))

        H = sOP(label, mi)
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == N**3
        assert siteop.mode == 0

        mat2 = siteop.todense()
        assert np.allclose(np.array(mat2), mat)

    def a(N):
        res = np.zeros((N, N))
        for i in range(N-1):
            res[i, i+1] = np.sqrt(i+1)
        return res

    def n(N):
        return np.diag(np.arange(N))

    amat = a(N)
    nmat = n(N)

    # test whether the boson operators works when as the only mode
    labels = ["a", "b", "c"]
    for label in labels:
        test_1(label, amat, N)

    labels = ["adag", "cdag", "bdag", "ad", "cd", "bd"]
    for label in labels:
        test_1(label, amat.T, N)

    labels = ["n", "adaga", "cdagc", "bdagb", "ada", "cdc", "bdb"]
    for label in labels:
        test_1(label, nmat, N)

    test_1("x", 1/(np.sqrt(2))*(amat+amat.T), N)
    test_1("q", 1/(np.sqrt(2))*(amat+amat.T), N)
    test_1("p", 1.0j/(np.sqrt(2))*(amat.T-amat), N)

    # test whether the boson operators works when created as a composite mode at various positions
    for mi in range(3):
        test_2("a", amat, N, mi)
        test_2("adag", amat.T, N, mi)
        test_2("n", nmat, N, mi)
        test_2("x", 1/(np.sqrt(2))*(amat+amat.T), N, mi)
        test_2("p", 1.0j/(np.sqrt(2))*(amat.T-amat), N, mi)


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
def test_spin(N):
    # test whether it works when we are dealing with a single mode
    def test_1(label, mat, N):
        sysinf = system_modes(1)
        sysinf[0] = spin_mode(N)

        H = sOP(label, 0)
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == N
        assert siteop.mode == 0

        mat2 = siteop.todense()
        assert np.allclose(np.array(mat2), mat)

    # test whether it works when we are dealing with a single mode
    def test_2(label, mat, N, mi):
        mi = mi % 3
        sysinf = system_modes(1)
        sysinf[0] = [spin_mode(N), spin_mode(N), spin_mode(N)]

        Id = np.identity(N)

        if mi == 0:
            mat = np.kron(mat, np.kron(Id, Id))
        elif mi == 1:
            mat = np.kron(Id, np.kron(mat, Id))
        elif mi == 2:
            mat = np.kron(Id, np.kron(Id, mat))

        H = sOP(label, mi)
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == N**3
        assert siteop.mode == 0

        mat2 = siteop.todense()
        assert np.allclose(np.array(mat2), mat)

    def sx(N):
        if N == 2:
            return 1/2*np.array([[0, 1], [1, 0]])
        elif N == 3:
            return np.sqrt(1/2)*(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        elif N == 4:
            return 1/2*np.array([[0, np.sqrt(3), 0, 0], [np.sqrt(3), 0, 2, 0], [0, 2, 0, np.sqrt(3)], [0, 0, np.sqrt(3), 0]])
        elif N == 5:
            return 1/2*np.array([[0, 2, 0, 0, 0], [2, 0, np.sqrt(6), 0, 0], [0, np.sqrt(6), 0, np.sqrt(6), 0], [0, 0, np.sqrt(6), 0, 2], [0, 0, 0, 2, 0]])
        elif N == 6:
            return 1/2*np.array([[0, np.sqrt(5), 0, 0, 0, 0], [np.sqrt(5), 0, np.sqrt(8), 0, 0, 0], [0, np.sqrt(8), 0, 3, 0, 0], [0, 0, 3, 0, np.sqrt(8), 0], [0, 0, 0, np.sqrt(8), 0, np.sqrt(5)], [0, 0, 0, 0, np.sqrt(5), 0]])

    def sy(N):
        if N == 2:
            return 1/(2.0j)*np.array([[0, 1], [-1, 0]])
        elif N == 3:
            return 1/(np.sqrt(2)*1.0j)*(np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]]))
        elif N == 4:
            return 1/(2.0j)*np.array([[0, np.sqrt(3), 0, 0], [-np.sqrt(3), 0, 2, 0], [0, -2, 0, np.sqrt(3)], [0, 0, -np.sqrt(3), 0]])
        elif N == 5:
            return 1/(2.0j)*np.array([[0, 2, 0, 0, 0], [-2, 0, np.sqrt(6), 0, 0], [0, -np.sqrt(6), 0, np.sqrt(6), 0], [0, 0, -np.sqrt(6), 0, 2], [0, 0, 0, -2, 0]])
        elif N == 6:
            return 1/(2.0j)*np.array([[0, np.sqrt(5), 0, 0, 0, 0], [-np.sqrt(5), 0, np.sqrt(8), 0, 0, 0], [0, -np.sqrt(8), 0, 3, 0, 0], [0, 0, -3, 0, np.sqrt(8), 0], [0, 0, 0, -np.sqrt(8), 0, np.sqrt(5)], [0, 0, 0, 0, -np.sqrt(5), 0]])

    def sz(N):
        S = (N-1.0)/2
        res = np.zeros((N, N))
        for i in range(N):
            m = S-i
            res[i, i] = m
        return res

    sxmat = sx(N)
    symat = sy(N)
    szmat = sz(N)

    spmat = sxmat+1.0j*symat

    # test whether the boson operators works when as the only mode
    labels = ["s+", "sp"]
    for label in labels:
        test_1(label, spmat, N)

    labels = ["s-", "sm"]
    for label in labels:
        test_1(label, spmat.T, N)

    labels = ["sx", "x"]
    for label in labels:
        test_1(label, sxmat, N)

    labels = ["sy", "y"]
    for label in labels:
        test_1(label, symat, N)

    labels = ["sz", "z"]
    for label in labels:
        test_1(label, szmat, N)
    # test whether the boson operators works when created as a composite mode at various positions
    for mi in range(3):
        test_2("s+", spmat, N, mi)
        test_2("s-", spmat.T, N, mi)
        test_2("sx", sxmat, N, mi)
        test_2("sy", symat, N, mi)
        test_2("sz", szmat, N, mi)


def test_tls():
    # test whether it works when we are dealing with a single mode
    def test_1(label, mat):
        sysinf = system_modes(1)
        sysinf[0] = tls_mode()

        H = sOP(label, 0)
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == 2
        assert siteop.mode == 0

        mat2 = siteop.todense()
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
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == 8
        assert siteop.mode == 0

        mat2 = siteop.todense()
        print(mat2)
        assert np.allclose(np.array(mat2), mat)

    sxmat = np.array([[0, 1], [1, 0]])
    symat = np.array([[0, -1.0j], [1.0j, 0]])
    szmat = np.array([[1, 0], [0, -1]])

    spmat = (sxmat+1.0j*symat)/2.0
    smmat = (sxmat-1.0j*symat)/2.0

    # test whether the boson operators works when as the only mode
    labels = ["s+", "sp", "sigma+", "sigmap"]
    for label in labels:
        test_1(label, spmat)

    labels = ["s-", "sm", "sigma-", "sigmam"]
    for label in labels:
        test_1(label, smmat)

    labels = ["sx", "x", "sigmax"]
    for label in labels:
        test_1(label, sxmat)

    labels = ["sy", "y", "sigmay"]
    for label in labels:
        test_1(label, symat)

    labels = ["sz", "z", "sigmaz"]
    for label in labels:
        test_1(label, szmat)

    s00 = np.zeros((2,2))
    s01 = np.zeros((2,2))
    s10 = np.zeros((2,2))
    s11 = np.zeros((2,2))
    s00[0,0]=1
    s01[0,1]=1
    s10[1,0]=1
    s11[1,1]=1

    test_1("|0><0|", s00)
    test_1("|0><1|", s01)
    test_1("|1><0|", s10)
    test_1("|1><1|", s11)
        
    # test whether the boson operators works when created as a composite mode at various positions
    for mi in range(3):
        test_2("s+", spmat, mi)
        test_2("s-", smmat, mi)
        test_2("sx", sxmat, mi)
        test_2("sy", symat, mi)
        test_2("sz", szmat, mi)

        test_2("|0><0|", s00, mi)
        test_2("|0><1|", s01, mi)
        test_2("|1><0|", s10, mi)
        test_2("|1><1|", s11, mi)

@pytest.mark.parametrize("N", [2, 3, 4])
def test_nlevel(N) -> None:
    # test whether it works when we are dealing with a single mode
    def test_1(label, mat):
        sysinf = system_modes(1)
        sysinf[0] = nlevel_mode(N)

        H = sOP(label, 0)
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == N
        assert siteop.mode == 0

        mat2 = siteop.todense()
        assert np.allclose(np.array(mat2), mat)

    # test whether it works when we are dealing with a single mode
    def test_2(label, mat, mi):
        mi = mi % 3
        sysinf = system_modes(1)
        sysinf[0] = [nlevel_mode(N), nlevel_mode(2*N), nlevel_mode(N)]

        Id1 = np.identity(N)
        Id2 = np.identity(2*N)
        Id3 = np.identity(N)
        if mi == 0:
            mat = np.kron(mat, np.kron(Id2, Id3))
        elif mi == 1:
            mat = np.kron(Id1, np.kron(mat, Id3))
        elif mi == 2:
            mat = np.kron(Id1, np.kron(Id2, mat))

        H = sOP(label, mi)
        siteop = site_operator(H, sysinf)
        assert not siteop.is_identity()
        assert siteop.size() == 2*N**3
        assert siteop.mode == 0

        mat2 = siteop.todense()
        print(mat2)
        assert np.allclose(np.array(mat2), mat)

    random.seed(0)

    for test_ind in range(10):
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        label = "|"+str(i)+"><"+str(j)+"|"
        op = np.zeros((N, N))
        op[i, j]=1

        test_1(label, op)

        dims = [N, 2*N, N]
        for mi in range(3):
            op2 = np.zeros((dims[mi], dims[mi]))
            op2[i, j]=1
            test_2(label, op2, mi)