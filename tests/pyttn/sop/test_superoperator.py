# test_superop.py

import pytest

from pyttn import  spin_mode

from pyttn.ttns.sop import SystemInfo, operator, op, SuperOp


@pytest.fixture
def spin_system():
    hsys = SystemInfo()
    hsys["s0"] = spin_mode(2)
    lsys = hsys.liouville_space()
    return hsys, lsys


def test_left_preserves_physical_label(spin_system):

    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    L, Lopdict = SuperOp.left(H(), hsys, lsys)

    labels = set(L.index_to_label.values())

    assert "s0" in labels
    assert "s0~" in labels
    assert Lopdict is None


def test_right_preserves_liouville_indexing(spin_system):

    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    R, Lopdict = SuperOp.right(H(), hsys, lsys)

    labels = set(R.index_to_label.values())

    assert "s0" in labels
    assert "s0~" in labels
    assert Lopdict is None


def test_left_keeps_sigma_plus(spin_system):

    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    L, _ = SuperOp.left(H(), hsys, lsys)

    txt = str(L)

    assert "s+" in txt


def test_right_transposes_sigma_plus(spin_system):

    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    R, _ = SuperOp.right(H(), hsys, lsys)

    txt = str(R)

    assert "s-" in txt


def test_right_transposes_sigma_minus(spin_system):

    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s-", "s0")

    R, _ = SuperOp.right(H(), hsys, lsys)

    txt = str(R)

    assert "s+" in txt


def test_left_and_right_have_same_mode_count(spin_system):

    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    L, _ = SuperOp.left(H(), hsys, lsys)
    R, _ = SuperOp.right(H(), hsys, lsys)

    assert L.nmodes() == R.nmodes()


def test_left_and_right_preserve_dtype(spin_system):

    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    O = H()

    L, _ = SuperOp.left(O, hsys, lsys)
    R, _ = SuperOp.right(O, hsys, lsys)

    assert L.dtype == O.dtype
    assert R.dtype == O.dtype


def test_left_invalid_system():

    hsys = SystemInfo()
    hsys["s0"] = spin_mode(2)

    lsys = SystemInfo()

    @operator(N=1)
    def H():
        return op("s+", "s0")

    with pytest.raises(ValueError):
        SuperOp.left(H(), hsys, lsys)


def test_right_invalid_system():

    hsys = SystemInfo()
    hsys["s0"] = spin_mode(2)

    lsys = SystemInfo()

    @operator(N=1)
    def H():
        return op("s+", "s0")

    with pytest.raises(ValueError):
        SuperOp.right(H(), hsys, lsys)



def test_sigma_plus_commutator_explicit(spin_system):
    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    C, _ = SuperOp.commutator(H(), hsys, lsys)

    @operator(N=2)
    def C_expected():
        return op("s+", "s0") - op("s-", "s0~")

    assert str(C) == str(C_expected())


def test_sigma_plus_anticommutator_explicit(spin_system):
    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    A, _ = SuperOp.anticommutator(H(), hsys, lsys)

    @operator(N=2)
    def A_expected():
        return op("s+", "s0") + op("s-", "s0~")

    assert str(A) == str(A_expected())


def test_sigma_minus_commutator_explicit(spin_system):
    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s-", "s0")

    C, _ = SuperOp.commutator(H(), hsys, lsys)

    @operator(N=2)
    def C_expected():
        return op("s-", "s0") - op("s+", "s0~")

    assert str(C) == str(C_expected())


def test_sigma_minus_anticommutator_explicit(spin_system):
    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s-", "s0")

    A, _ = SuperOp.anticommutator(H(), hsys, lsys)

    @operator(N=2)
    def A_expected():
        return op("s-", "s0") + op("s+", "s0~")

    assert str(A) == str(A_expected())


def test_sz_commutator_explicit(spin_system):
    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("sz", "s0")

    C, _ = SuperOp.commutator(H(), hsys, lsys)

    @operator(N=2)
    def C_expected():
        return op("sz", "s0") - op("sz", "s0~")

    assert str(C) == str(C_expected())

def test_sz_anticommutator_explicit(spin_system):
    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("sz", "s0")

    A, _ = SuperOp.anticommutator(H(), hsys, lsys)

    @operator(N=2)
    def A_expected():
        return (op("sz", "s0")+ op("sz", "s0~"))

    assert str(A) == str(A_expected())

    hsys, lsys = spin_system


def test_commutator_equals_left_plus_right(spin_system):
    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    HL, _ = SuperOp.left(H(), hsys, lsys)
    HR, _ = SuperOp.right(H(), hsys, lsys)

    C, _ = SuperOp.commutator(H(), hsys, lsys)

    assert str(C) == str(HL - HR)


def test_anticommutator_equals_left_plus_right(spin_system):
    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0")

    HL, _ = SuperOp.left(H(), hsys, lsys)
    HR, _ = SuperOp.right(H(), hsys, lsys)

    A, _ = SuperOp.anticommutator(H(), hsys, lsys)

    assert str(A) == str(HL + HR)

def test_commutator_multiterm_operator(spin_system):

    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0") + op("s-", "s0")

    C, _ = SuperOp.commutator(H(), hsys, lsys)

    HL, _ = SuperOp.left(H(), hsys, lsys)
    HR, _ = SuperOp.right(H(), hsys, lsys)

    assert str(C) == str(HL - HR)


def test_anticommutator_multiterm_operator(spin_system):
    hsys, lsys = spin_system

    @operator(N=1)
    def H():
        return op("s+", "s0") + op("s-", "s0")

    A, _ = SuperOp.anticommutator(H(), hsys, lsys)

    HL, _ = SuperOp.left(H(), hsys, lsys)
    HR, _ = SuperOp.right(H(), hsys, lsys)

    assert str(A) == str(HL + HR)
