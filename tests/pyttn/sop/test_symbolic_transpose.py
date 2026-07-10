import numpy as np
import pytest

from pyttn import (
    sOP,
    sPOP,
    sSOP,
    system_modes,
    boson_mode,
    spin_mode,
    tls_mode,
    fermion_mode,
    nlevel_mode,
    generic_mode,
    operator_dictionary,
    site_operator,
)

# IMPORTANT: import your function
from pyttn.ttns.sop import symbolic_transpose



# Helpers


def assert_transpose_equal(op, sys, opdict=None, tol=1e-12):
    """Check symbolic transpose matches dense transpose."""

    M = np.asarray(op.todense(sys, opdict))
    opT, dictT = symbolic_transpose(op, sys, opdict)
    MT = np.asarray(opT.todense(sys, dictT))

    assert M.shape == MT.shape
    assert np.allclose(MT, M.T, atol=tol)



# Mode builders


def make_boson_sys():
    sys = system_modes(1)
    sys[0] = boson_mode(4)
    return sys


def make_spin_sys():
    sys = system_modes(1)
    sys[0] = spin_mode(2)
    return sys


def make_tls_sys():
    sys = system_modes(1)
    sys[0] = tls_mode()
    return sys


def make_fermion_sys():
    sys = system_modes(1)
    sys[0] = fermion_mode()
    return sys


def make_nlevel_sys():
    sys = system_modes(1)
    sys[0] = nlevel_mode(3)
    return sys


def make_product_sys(mode_fn, dim=None):
    sys = system_modes(2)
    sys[0] = mode_fn(dim) if dim else mode_fn()
    sys[1] = mode_fn(dim) if dim else mode_fn()
    return sys



# 1. BOSON OPERATORS
@pytest.mark.parametrize("label", ["a", "adag", "n", "x", "p"])
def test_boson_sOP_transpose(label):
    sys = make_boson_sys()
    op = sOP(label, 0)
    assert_transpose_equal(op, sys)


def test_boson_product():
    sys = make_product_sys(boson_mode, 4)
    op = sOP("a", 0) * sOP("adag", 1)
    assert_transpose_equal(op, sys)



# 2. SPIN OPERATORS
@pytest.mark.parametrize("label", ["sx", "sy", "sz", "sp", "sm"])
def test_spin_sOP_transpose(label):
    sys = make_spin_sys()
    op = sOP(label, 0)
    assert_transpose_equal(op, sys)



# 3. TLS OPERATORS
@pytest.mark.parametrize("label", ["x", "y", "z", "sp", "sm"])
def test_tls_sOP_transpose(label):
    sys = make_tls_sys()
    op = sOP(label, 0)
    assert_transpose_equal(op, sys)



# 4. FERMION OPERATORS
@pytest.mark.parametrize("label", ["a", "adag", "n", "v"])
def test_fermion_sOP_transpose(label):
    sys = make_fermion_sys()
    op = sOP(label, 0)
    assert_transpose_equal(op, sys)


def test_fermion_product():
    sys = make_product_sys(fermion_mode)
    op = sOP("adag", 0) * sOP("a", 1)
    assert_transpose_equal(op, sys)



# 5. N-LEVEL OPERATORS
@pytest.mark.parametrize("label", ["|0><0|", "|0><1|", "|1><0|", "|2><0|"])
def test_nlevel_projectors(label):
    sys = make_nlevel_sys()
    op = sOP(label, 0)
    assert_transpose_equal(op, sys)



# 6. sPOP 
def test_sPOP_order_reversal():
    sys = system_modes(2)
    sys[0] = boson_mode(3)
    sys[1] = boson_mode(3)

    # A ⊗ B
    op = sOP("a", 0) * sOP("adag", 1)

    assert_transpose_equal(op, sys)

def test_sPOP_mixed_modes():
    sys = system_modes(2)
    sys[0] = boson_mode(3)
    sys[1] = spin_mode(2)

    op = sOP("adag", 0) * sOP("sx", 1)

    assert_transpose_equal(op, sys)

def test_sPOP_same_site():
    sys = system_modes(1)
    sys[0] = spin_mode(2)

    op = sOP("sx", 0) * sOP("sz", 0)

    assert_transpose_equal(op, sys)

def test_sPOP_list_constructor():
    sys = system_modes(3)
    sys[0] = boson_mode(3)
    sys[1] = boson_mode(3)
    sys[2] = boson_mode(3)

    ops = [sOP("a", 0), sOP("adag", 1), sOP("n", 2)]

    op = sPOP(ops)

    assert_transpose_equal(op, sys)




# 7. sNBO 
def test_sNBO_real_scalar():
    sys = system_modes(1)
    sys[0] = boson_mode(3)

    op = 2.0 * sOP("a", 0)

    assert_transpose_equal(op, sys)

def test_sNBO_complex_scalar():
    sys = system_modes(1)
    sys[0] = spin_mode(2)

    op = (1.0 + 2.0j) * sOP("sx", 0)

    assert_transpose_equal(op, sys)

def test_sNBO_product():
    sys = system_modes(2)
    sys[0] = boson_mode(3)
    sys[1] = boson_mode(3)

    op = 3.0 * (sOP("a", 0) * sOP("adag", 1))

    assert_transpose_equal(op, sys)

def test_sNBO_nested():
    sys = system_modes(2)
    sys[0] = spin_mode(2)
    sys[1] = spin_mode(2)

    op = (2.0 + 1.0j) * sOP("sx", 0) * sOP("sz", 1)

    assert_transpose_equal(op, sys)

def test_sNBO_custom_operator():
    sys = system_modes(1)
    sys[0] = generic_mode(2)

    opdict = operator_dictionary(1)

    mat = np.array([[0, 1], [2, 3]], dtype=np.complex128)
    opdict.insert(0, "A", site_operator(mat, optype="matrix", mode=0))

    op = 2.0 * sOP("A", 0)
    print(type(op))
    print(type)

    assert_transpose_equal(op, sys, opdict)


# 8. sSOP 
def test_sSOP_sum():
    sys = make_spin_sys()

    op = sSOP()
    op += 2.0*sOP("sx", 0)
    op += sOP("sz", 0)

    assert_transpose_equal(op, sys)





# 9. USER-DEFINED OPERATORS
def test_custom_operator_transpose():
    sys = system_modes(1)
    sys[0] = generic_mode(2)

    opdict = operator_dictionary(1)

    # define matrix
    mat = np.array([[0, 1], [2, 3]], dtype=np.complex128)

    opdict.insert(0, "A", site_operator(mat, optype="matrix", mode=0))

    op = sOP("A", 0)

    assert_transpose_equal(op, sys, opdict)



# 10. MIXED MANY-BODY CASE
def test_mixed_operator():
    sys = system_modes(3)
    sys[0] = boson_mode(3)
    sys[1] = spin_mode(2)
    sys[2] = fermion_mode()

    op = (
        sOP("adag", 0)
        * sOP("sx", 1)
        * sOP("a", 2)
    )

    assert_transpose_equal(op, sys)