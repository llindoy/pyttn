import numpy as np
import pytest

from pyttn.ttns.sop import operator, op, sSOP, lSOP, lCSOP, SystemInfo

from pyttn.ttnpp import spin_mode


# =============================================================================
# Test operators
# =============================================================================


@operator
def H1():
    return (op("sx", "a") + 2.0 * op("sz", "b"))


@operator
def H2():
    return (3.0 * op("sy", "b")+ op("sx", "c"))


@operator
def HX():
    return op("sx", "a")


@operator
def HZ():
    return op("sz", "d")


@operator
def Hsum():
    return op("sx", "a") + op("sz", "d")

@operator
def Hprod():
    return op("sx", "a") * op("sz", "d")


@operator
def HXZ_square():
    return (op("sx", "a") + op("sz", "d"))*(op("sx", "a") + op("sz", "d"))

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sysinfo():
    sys = SystemInfo()
    sys["a"] = spin_mode(2)
    sys["b"] = spin_mode(2)
    sys["c"] = spin_mode(2)
    sys["d"] = spin_mode(2)
    return sys


@pytest.fixture
def ordering():
    return ["a", "b", "c", "d"]


# =============================================================================
# Helpers
# =============================================================================


def dense_operator(H, ordering, sysinfo):
    """
    Convert a labelled operator into a dense matrix.
    """
    site_map = {s: i for i, s in enumerate(ordering)}
    nmodes = len(ordering)

    if isinstance(H, lSOP):
        sop = H.compile(site_map, nmodes)
    elif isinstance(H, lCSOP):
        sop = H.compile(site_map, nmodes)
    else:
        raise TypeError(type(H))

    sys = sysinfo.build_flattened_modes(ordering)
    return sop.todense(sys)


def assert_same_operator(A, B, ordering, sysinfo, atol=1e-12):
    """
    Assert two labelled operators are numerically identical.
    """
    Ad = dense_operator(A, ordering, sysinfo)
    Bd = dense_operator(B, ordering, sysinfo)

    np.testing.assert_allclose(Ad,Bd,atol=atol)


# =============================================================================
# lSOP algebra
# =============================================================================


def test_lsop_add(sysinfo, ordering):

    A = H1()
    B = H2()

    result = A + B
    expected = H1() + H2()

    assert_same_operator(result, expected, ordering, sysinfo)


def test_lsop_iadd(sysinfo, ordering):

    result = H1()
    result += H2()

    expected = H1() + H2()

    assert_same_operator(result, expected, ordering, sysinfo)



def test_lsop_sub(sysinfo, ordering):

    A = H1()
    B = H2()

    result = A - B
    expected = H1() + (-1) * H2()

    assert_same_operator(result, expected, ordering, sysinfo)


def test_lsop_isub(sysinfo, ordering):

    result = H1()
    result -= H2()

    expected = H1() + (-1) * H2()

    assert_same_operator(result, expected, ordering, sysinfo)


def test_lsop_neg(sysinfo, ordering):

    A = H1()

    result = -A
    expected = (-1) * H1()
    assert_same_operator(result, expected, ordering, sysinfo)



def test_lsop_mul(sysinfo, ordering):

    A = H1()

    result = 3.5 * A
    expected = A * 3.5
    assert_same_operator(result, expected, ordering, sysinfo)



def test_lsop_div(sysinfo, ordering):

    A = H1()

    result = A / 2.0
    expected = 0.5 * A
    assert_same_operator(result, expected, ordering, sysinfo)



def test_lsop_matmul(sysinfo, ordering):

    A = H1()
    B = H2()

    result = A @ B

    site_map = {s: i for i, s in enumerate(ordering)}

    expr_a = A.to_sSOP(site_map)
    expr_b = B.to_sSOP(site_map)

    expected = lSOP(expr_a * expr_b,{i: s for i, s in enumerate(ordering)},)
    assert_same_operator(result, expected, ordering, sysinfo)

def test_lsop_matmul_2(sysinfo, ordering):

    A = H1()
    B = H2()

    result = A * B

    site_map = {s: i for i, s in enumerate(ordering)}

    expr_a = A.to_sSOP(site_map)
    expr_b = B.to_sSOP(site_map)

    expected = lSOP(expr_a * expr_b,{i: s for i, s in enumerate(ordering)},)
    assert_same_operator(result, expected, ordering, sysinfo)

# =============================================================================
# Label-space extension
# =============================================================================


def test_lsop_site_extension():

    C = HX() + HZ()

    assert C.sites() == {"a", "d"}


def test_lsop_site_extension_matmul():

    C = HX() @ HZ()

    assert C.sites() == {"a", "d"}

def test_lsop_site_extension_matmul_2():

    C = HX() * HZ()

    assert C.sites() == {"a", "d"}

# =============================================================================
# lCSOP algebra
# =============================================================================


@pytest.fixture
def H1c():
    return H1().to_lCSOP(["a", "b", "c", "d"])


@pytest.fixture
def H2c():
    return H2().to_lCSOP(["a", "b", "c", "d"])


def test_lcsop_add(H1c, H2c, ordering, sysinfo):

    result = H1c + H2c
    expected = (H1() + H2()).to_lCSOP(ordering)

    assert_same_operator(result, expected, ordering, sysinfo)



def test_lcsop_iadd(H1c, H2c, ordering, sysinfo):

    result = H1().to_lCSOP(ordering)
    result += H2c

    expected = (H1() + H2()).to_lCSOP(ordering)

    assert_same_operator(result, expected, ordering, sysinfo)



def test_lcsop_sub(H1c, H2c, ordering, sysinfo):

    result = H1c - H2c
    expected = (H1() - H2()).to_lCSOP(ordering)

    assert_same_operator(result, expected, ordering, sysinfo)



def test_lcsop_isub(H1c, H2c, ordering, sysinfo):

    result = H1().to_lCSOP(ordering)
    result -= H2c

    expected = (H1() - H2()).to_lCSOP(ordering)

    assert_same_operator(result, expected, ordering, sysinfo)



def test_lcsop_neg(H1c, ordering, sysinfo):

    result = -H1c
    expected = (-1 * H1()).to_lCSOP(ordering)
    
    assert_same_operator(result, expected, ordering, sysinfo)



def test_lcsop_mul(H1c, ordering, sysinfo):

    result = 5.0 * H1c
    expected = (5.0 * H1()).to_lCSOP(ordering)
    assert_same_operator(result, expected, ordering, sysinfo)


def test_lcsop_div(H1c, ordering, sysinfo):

    result = H1c / 2.0
    expected = (0.5 * H1()).to_lCSOP(ordering)
    assert_same_operator(result, expected, ordering, sysinfo)



# =============================================================================
# Mixed lSOP/lCSOP algebra
# =============================================================================


def test_mixed_add(sysinfo, ordering):

    A = H1()
    B = H2().to_lCSOP(ordering)

    result = A + B
    expected = H1() + H2()
    assert_same_operator(result, expected, ordering, sysinfo)



def test_mixed_iadd(sysinfo, ordering):

    result = H1()
    result += H2().to_lCSOP(ordering)

    expected = H1() + H2()
    assert_same_operator(result, expected, ordering, sysinfo)



def test_mixed_sub(sysinfo, ordering):

    A = H1()
    B = H2().to_lCSOP(ordering)

    result = A - B
    expected = H1() - H2()

    assert_same_operator(result, expected, ordering, sysinfo)



def test_mixed_matmul(sysinfo, ordering):

    A = H1()
    B = H2().to_lCSOP(ordering)

    result = A @ B

    site_map = {s: i for i, s in enumerate(ordering)}

    expr_a = H1().to_sSOP(site_map)
    expr_b = H2().to_sSOP(site_map)

    expected = lSOP(
        expr_a * expr_b,
        {i: s for i, s in enumerate(ordering)},
    )

    assert_same_operator(result, expected, ordering, sysinfo)

def test_mixed_matmul_2(sysinfo, ordering):

    A = H1()
    B = H2().to_lCSOP(ordering)

    result = B @ A

    site_map = {s: i for i, s in enumerate(ordering)}

    expr_a = H1().to_sSOP(site_map)
    expr_b = H2().to_sSOP(site_map)

    expected = lSOP(
        expr_b * expr_a,
        {i: s for i, s in enumerate(ordering)},
    )

    assert_same_operator(result, expected, ordering, sysinfo)


def test_mixed_matmul_3(sysinfo, ordering):

    A = H1()
    B = H2().to_lCSOP(ordering)

    result = A * B

    site_map = {s: i for i, s in enumerate(ordering)}

    expr_a = H1().to_sSOP(site_map)
    expr_b = H2().to_sSOP(site_map)

    expected = lSOP(
        expr_a * expr_b,
        {i: s for i, s in enumerate(ordering)},
    )

    assert_same_operator(result, expected, ordering, sysinfo)

def test_mixed_matmul_4(sysinfo, ordering):

    A = H1()
    B = H2().to_lCSOP(ordering)

    result = B * A 

    site_map = {s: i for i, s in enumerate(ordering)}

    expr_a = H1().to_sSOP(site_map)
    expr_b = H2().to_sSOP(site_map)

    expected = lSOP(
        expr_b * expr_a,
        {i: s for i, s in enumerate(ordering)},
    )

    assert_same_operator(result, expected, ordering, sysinfo)
# =============================================================================
# Type checking
# =============================================================================


def test_lsop_invalid_add():

    H = H1()

    with pytest.raises(TypeError):
        _ = H + 1.23


def test_lcsop_invalid_add(ordering):

    H = H1().to_lCSOP(ordering)

    with pytest.raises(TypeError):
        _ = H + 1.23


def test_lsop_invalid_matmul():

    H = H1()

    with pytest.raises(TypeError):
        _ = H @ 2


def test_operator_addition_composition(sysinfo, ordering):
    Hx = HX()
    Hz = HZ()
    H = Hx + Hz
    expected = Hsum()
    assert_same_operator(H, expected, ordering, sysinfo)

def test_operator_addition_composition(sysinfo, ordering):
    Hx = HX()
    Hz = HZ()
    H = Hx @ Hz
    expected = Hprod()
    assert_same_operator(H, expected, ordering, sysinfo)

def test_operator_addition_composition(sysinfo, ordering):
    Hx = HX()
    Hz = HZ()
    H = (Hx + Hz) @ (Hx + Hz)
    expected = HXZ_square()
    assert_same_operator(H, expected, ordering, sysinfo)