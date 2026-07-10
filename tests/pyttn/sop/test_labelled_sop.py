import numpy as np
import pytest

from pyttn.ttns.sop import (
    OperatorBuilder,
    lSOP,
    sOP,
    operator_context,
    op,
    fop,
    fOP,
    sSOP,
    operator,
    SystemInfo,
)

from pyttn.ttnpp import spin_mode, fermion_mode

def make_fermion_sys(n):
    labels = [f"p{i}" for i in range(n)]
    sysinfo = SystemInfo()
    for l in labels:
        sysinfo[l] = fermion_mode()
    return sysinfo

def make_spin_sys(n):
    labels = [f"p{i}" for i in range(n)]
    sysinfo = SystemInfo()
    for l in labels:
        sysinfo[l] = spin_mode(2)
    return sysinfo

def compile_dense(lsop, site_map, sysinfo, ordering):
    sys = sysinfo.build_flattened_modes(ordering)
    sop = lsop.compile(site_map, sys.nmodes(), backend="sop")
    return np.asarray(sop.expand().todense(sys))

def to_np(x):
    return np.asarray(x)


# OperatorBuilder

def test_builder_index_consistency():
    b = OperatorBuilder()

    i0 = b._OperatorBuilder__get_index("a")
    i1 = b._OperatorBuilder__get_index("b")
    i0_again = b._OperatorBuilder__get_index("a")

    assert i0 == i0_again
    assert i0 != i1
    assert b.index_to_label[i0] == "a"


def test_builder_multiple_ops_same_site():
    b = OperatorBuilder()

    op1 = b.op("sx", "p0")
    op2 = b.op("sz", "p0")

    assert op1.mode == op2.mode


def test_builder_fermionic_flag():
    b = OperatorBuilder()

    op1 = b.op("n", "p0")
    op2 = b.fop("a", "p1")

    assert not op1.fermionic
    assert op2.fermionic



# operator_context / decorator


def test_operator_context_wrap():
    with operator_context() as b:
        expr = op("sx", "p0")

        lsop = b.wrap(expr)

    assert isinstance(lsop, lSOP)
    assert lsop.sites() == {"p0"}


def test_operator_decorator_basic():
    @operator
    def H():
        return op("sx", "p0") * op("sz", "p1")

    h = H()

    assert isinstance(h, lSOP)
    assert h.sites() == {"p0", "p1"}


# lSOP basic behaviour
def test_lsop_sites_and_repr():
    @operator
    def H():
        return op("sx", "a") * op("sz", "b")

    h = H()

    assert h.sites() == {"a", "b"}
    assert "LabelledSOP" in repr(h)


def test_lsop_dtype():
    @operator
    def H():
        return op("sx", "p0")

    h = H()

    assert h.dtype in (np.float64, np.complex128)



# Conversion correctness


def test_lsop_to_sSOP_vs_SOP_equivalence():
    @operator
    def H():
        return op("sx", "p0") * op("sz", "p1")

    h = H()

    site_map = {"p0": 0, "p1": 1}

    ssop = h.to_sSOP(site_map)
    sop = h.to_SOP(site_map, nmodes=2)

    sys = make_spin_sys(2)
    sys = sys.build_flattened_modes(["p0", "p1"])

    M1 = to_np(ssop.todense(sys))
    M2 = to_np(sop.expand().todense(sys))

    assert np.allclose(M1, M2)


def test_lsop_single_site():
    @operator
    def H():
        return op("sx", "p0")

    h = H()

    site_map = {"p0": 0}

    sop = h.to_SOP(site_map, nmodes=1)

    assert sop.nmodes() == 1
    assert sop.nterms() >= 1



# compile()


def test_lsop_compile_ssop():
    @operator
    def H():
        return op("sx", "p0")

    h = H()

    out = h.compile({"p0": 0}, backend="ssop")

    assert out.nterms() >= 1


def test_lsop_compile_sop():
    @operator
    def H():
        return op("sx", "p0")

    h = H()

    sop = h.compile({"p0": 0}, nmodes=1, backend="sop")

    assert sop.nmodes() == 1


def test_lsop_compile_bad_backend():
    @operator
    def H():
        return op("sx", "p0")

    with pytest.raises(ValueError):
        H().compile({"p0": 0}, backend="invalid")


# Error handling

def test_lsop_missing_site():
    @operator
    def H():
        return op("sx", "p0")

    with pytest.raises(ValueError):
        H().to_SOP({"p1": 0}, nmodes=1)


def test_lsop_partial_site_map():
    @operator
    def H():
        return op("sx", "p0") * op("sz", "p1")

    with pytest.raises(ValueError):
        H().to_SOP({"p0": 0}, nmodes=2)



# Fermionic correctness


def test_lsop_fermionic_build():
    @operator
    def H():
        return fop("a", "p0") * fop("adag", "p1")

    h = H()

    assert "p0" in h.sites()
    assert "p1" in h.sites()



# Jordan-Wigner
def test_lsop_jordan_wigner_basic():
    sys = make_fermion_sys(2)

    @operator
    def H():
        return fop("a", "p0") * fop("adag", "p1")

    h = H()

    ordering = ["p0", "p1"]

    out = h.jordan_wigner(ordering, sys)

    assert isinstance(out, lSOP)
    assert out.sites() == {"p0", "p1"}


def test_lsop_jordan_wigner_missing_label():
    sys = make_fermion_sys(2)

    @operator
    def H():
        return fop("a", "p0")

    with pytest.raises(ValueError):
        H().jordan_wigner(["p1"], sys)


def test_jw_long_range_hamiltonian():
    builder = OperatorBuilder()

    t = 1.0
    U = 0.5

    expr = (
        t * builder.fop("cdag", "p0") * builder.fop("c", "p3")
        + t * builder.fop("cdag", "p3") * builder.fop("c", "p0")
        + t * builder.fop("cdag", "p1") * builder.fop("c", "p2")
        + t * builder.fop("cdag", "p2") * builder.fop("c", "p1")
        + U * builder.op("n", "p1") * builder.op("n", "p3")
    )

    labelled = builder.wrap(expr)

    ordering = ["p0", "p1", "p2", "p3"]
    sys = make_fermion_sys(4)

    jw_labelled = labelled.jordan_wigner(ordering, sys)

    site_map = {lab: i for i, lab in enumerate(ordering)}
    M = compile_dense(jw_labelled, site_map, sys, ordering)

    # basic sanity
    assert M.shape == (16, 16)
    assert np.all(np.isfinite(M))

def test_jw_ordering_permutation_consistency():
    builder = OperatorBuilder()

    expr = (
        builder.fop("cdag", "p0") * builder.fop("c", "p1")
        + builder.fop("cdag", "p1") * builder.fop("c", "p0")
    )

    labelled = builder.wrap(expr)
    sys = make_fermion_sys(2)

    ordering1 = ["p0", "p1"]
    ordering2 = ["p1", "p0"]

    jw1 = labelled.jordan_wigner(ordering1, sys)
    jw2 = labelled.jordan_wigner(ordering2, sys)

    M1 = compile_dense(jw1, {l: i for i, l in enumerate(ordering1)}, sys, ordering1)
    M2 = compile_dense(jw2, {l: i for i, l in enumerate(ordering2)}, sys, ordering2)

    # They should be related by permutation matrix similarity
    # but not generally identical
    assert np.allclose(M1, M2)


def test_jw_hermiticity():
    builder = OperatorBuilder()

    expr = (
        builder.fop("cdag", "p0") * builder.fop("c", "p1")
        + builder.fop("cdag", "p1") * builder.fop("c", "p0")
    )

    labelled = builder.wrap(expr)

    ordering = ["p0", "p1"]
    sys = make_fermion_sys(2)

    jw = labelled.jordan_wigner(ordering, sys)
    M = compile_dense(jw, {l: i for i, l in enumerate(ordering)}, sys, ordering)

    assert np.allclose(M, M.conj().T)


def test_jw_missing_label_error():
    builder = OperatorBuilder()

    expr = builder.fop("c", "p0")
    labelled = builder.wrap(expr)

    sys = make_fermion_sys(1)

    with pytest.raises(ValueError):
        labelled.jordan_wigner(["p1"], sys)



# 5. JW invariance under identity ordering
def test_jw_identity_ordering_stability():
    builder = OperatorBuilder()

    expr = builder.fop("n", "p0")
    labelled = builder.wrap(expr)

    ordering = ["p0"]
    sys = make_fermion_sys(1)

    jw = labelled.jordan_wigner(ordering, sys)

    M1 = compile_dense(labelled, {"p0": 0}, sys, ordering)
    M2 = compile_dense(jw, {"p0": 0}, sys, ordering)

    assert np.allclose(M1, M2)


def test_operator_decorator_spin_example():
    @operator
    def H():
        return op("sx", "p0") * op("sx", "p1") + op("sz", "p0")

    h = H()

    assert h.sites() == {"p0", "p1"}


def test_operator_decorator_hopping():
    @operator
    def hopping():
        return (
            fop("cdag", "p0") * fop("c", "p3")
            + fop("cdag", "p3") * fop("c", "p0")
        )

    h = hopping()

    assert h.sites() == {"p0", "p3"}

def test_operator_parameterised():
    @operator
    def XY_chain(J, h):
        return (
            J * op("sx", "p0") * op("sx", "p1")
            + J * op("sy", "p0") * op("sy", "p1")
            + h * op("sz", "p0")
        )

    H = XY_chain(1.0, 0.5)

    assert isinstance(H, type(XY_chain(1, 1)))
    assert {"p0", "p1"} == H.sites()


def test_jw_long_range_changes_with_order():
    builder = OperatorBuilder()

    expr = builder.fop("cdag", "p0") * builder.fop("c", "p3")
    labelled = builder.wrap(expr)

    sys = make_fermion_sys(4)

    ordering1 = ["p0", "p1", "p2", "p3"]
    ordering2 = ["p3", "p2", "p1", "p0"]

    jw1 = labelled.jordan_wigner(ordering1, sys)
    jw2 = labelled.jordan_wigner(ordering2, sys)

    M1 = compile_dense(jw1, {l: i for i, l in enumerate(ordering1)}, sys, ordering1)
    M2 = compile_dense(jw2, {l: i for i, l in enumerate(ordering2)}, sys, ordering2)

    # long-range JW strings  different representations
    assert not np.allclose(M1, M2)

def test_lsop_matches_manual_ssop_under_permutations():
    @operator
    def H():
        return (
            op("sx", "p0") * op("sz", "p2")
            + op("sy", "p1")
        )

    h = H()

    # Two different orderings
    ordering1 = ["p0", "p1", "p2"]
    ordering2 = ["p2", "p0", "p1"]

    sys1 = make_spin_sys(3)
    sys2 = make_spin_sys(3)

    site_map1 = {lab: i for i, lab in enumerate(ordering1)}
    site_map2 = {lab: i for i, lab in enumerate(ordering2)}

    ssop1 = h.to_sSOP(site_map1)
    ssop2 = h.to_sSOP(site_map2)

    manual1 = sOP("sx", site_map1["p0"])*sOP("sz", site_map1["p2"])+sOP("sy", site_map1["p1"])
    manual2 = sOP("sx", site_map2["p0"])*sOP("sz", site_map2["p2"])+sOP("sy", site_map2["p1"])
    sys1 = sys1.build_flattened_modes(ordering1)
    sys2 = sys2.build_flattened_modes(ordering2)

    M_gen_1 = np.asarray(ssop1.todense(sys1))
    M_man_1 = np.asarray(manual1.todense(sys1))

    M_gen_2 = np.asarray(ssop2.todense(sys2))
    M_man_2 = np.asarray(manual2.todense(sys2))

    assert np.allclose(M_gen_1, M_man_1)
    assert np.allclose(M_gen_2, M_man_2)


def test_jw_matches_manual_long_range_simple():
    builder = OperatorBuilder()

    expr = builder.fop("cdag", "p0") * builder.fop("c", "p3")
    labelled = builder.wrap(expr)

    ordering = ["p0", "p1", "p2", "p3"]
    site_map = {lab: i for i, lab in enumerate(ordering)}

    sys = make_fermion_sys(4)

    # Automatic JW
    jw = labelled.jordan_wigner(ordering, sys)
    M_auto = compile_dense(jw, site_map, sys, ordering)
    # Manual JW: cdag_0 * jw_1 * jw_2 * c_3
    manual = (
        sOP("cdag", site_map["p0"])
        * sOP("jw",   site_map["p1"])
        * sOP("jw",   site_map["p2"])
        * sOP("c",    site_map["p3"])
    )
    sys = sys.build_flattened_modes(ordering)
    M_manual = np.asarray(manual.todense(sys))

    assert np.allclose(M_auto, M_manual)


def test_jw_matches_manual_reversed_simple():
    builder = OperatorBuilder()

    expr = builder.fop("cdag", "p0") * builder.fop("c", "p3")
    labelled = builder.wrap(expr)

    ordering = ["p3", "p2", "p1", "p0"]
    site_map = {lab: i for i, lab in enumerate(ordering)}

    sys = make_fermion_sys(4)

    jw = labelled.jordan_wigner(ordering, sys)
    M_auto = compile_dense(jw, site_map, sys, ordering)

    # indices flipped
    manual = (
        sOP("cdag", site_map["p0"], True)
        * sOP("jw", site_map["p2"], True)
        * sOP("jw", site_map["p1"], True)
        * sOP("c",  site_map["p3"], True)
    )
    sys = sys.build_flattened_modes(ordering)

    M_manual = np.asarray(manual.todense(sys))

    assert np.allclose(M_auto, M_manual)
