import numpy as np
import pytest

from pyttn.ttns.sop import (
    operator,
    op,
    fop,
    sOP,
    lCSOP,
    SystemInfo
)

from pyttn.ttnpp import spin_mode, fermion_mode


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_fermion_sysinfo(n):
    labels = [f"p{i}" for i in range(n)]
    sysinfo = SystemInfo()
    for l in labels:
        sysinfo[l] = fermion_mode()
    return sysinfo


def make_spin_sysinfo(n):
    labels = [f"p{i}" for i in range(n)]
    sysinfo = SystemInfo()
    for l in labels:
        sysinfo[l] = spin_mode(2)
    return sysinfo


def build_sys(sysinfo, ordering):
    return sysinfo.build_flattened_modes(ordering)


def dense_from_sop(sop, sys):
    return np.asarray(sop.todense(sys))


# -----------------------------------------------------------------------------
# Basic behaviour
# -----------------------------------------------------------------------------

def test_lcsop_to_SOP_identity_mapping():
    ordering = ["p0", "p1"]
    sysinfo = make_spin_sysinfo(2)
    sys = build_sys(sysinfo, ordering)

    @operator(N=2)
    def H():
        return op("sx","p0") * op("sz","p1")

    h = H()

    site_map = {"p0": 0, "p1": 1}

    sop = h.to_SOP(site_map, nmodes=2)

    M1 = dense_from_sop(h.expr, sys)
    M2 = dense_from_sop(sop, sys)

    assert np.allclose(M1, M2)


def test_lcsop_to_SOP_permutation():
    ordering = ["p0", "p1", "p2"]
    sysinfo = make_spin_sysinfo(3)
    sys = build_sys(sysinfo, ordering)

    @operator(N=3)
    def H():
        return op("sx","p0") * op("sz","p2") + op("sy","p1")

    h = H()

    site_map = {"p0": 1, "p1": 2, "p2": 0}

    sop = h.to_SOP(site_map, nmodes=3)

    M = dense_from_sop(sop, sys)

    assert M.shape == (8, 8)
    assert np.all(np.isfinite(M))


# -----------------------------------------------------------------------------
# Manual SOP equivalence
# -----------------------------------------------------------------------------

def test_lcsop_matches_manual_sop():
    ordering = ["p0","p1","p2"]
    sysinfo = make_spin_sysinfo(3)
    sys = build_sys(sysinfo, ordering)

    @operator(N=3)
    def H():
        return op("sx","p0") * op("sz","p2") + op("sy","p1")

    h = H()

    site_map = {"p0":0,"p1":1,"p2":2}

    manual = (
        sOP("sx",0)*sOP("sz",2) + sOP("sy",1)
    )

    M1 = dense_from_sop(h.to_SOP(site_map), sys)
    M2 = np.asarray(manual.todense(sys))

    assert np.allclose(M1, M2)


# -----------------------------------------------------------------------------
# Jordan–Wigner
# -----------------------------------------------------------------------------

def test_lcsop_jw_basic():
    ordering = ["p0","p1"]
    sysinfo = make_fermion_sysinfo(2)
    sys = build_sys(sysinfo, ordering)

    @operator(N=2)
    def H():
        return fop("a", "p0") * fop("adag", "p1")

    h = H()

    out = h.jordan_wigner(ordering, sysinfo)

    assert isinstance(out, lCSOP)

    site_map = {"p0":0,"p1":1}
    M = dense_from_sop(out.to_SOP(site_map,2), sys)

    assert M.shape == (4,4)


def test_lcsop_jw_hermiticity():
    ordering = ["p0","p1"]
    sysinfo = make_fermion_sysinfo(2)
    sys = build_sys(sysinfo, ordering)

    @operator(N=2)
    def H():
        return (
            fop("cdag","p0")*fop("c","p1") +
            fop("cdag","p1")*fop("c","p0")
        )

    h = H()
    print("ham:", h)
    jw = h.jordan_wigner(ordering, sysinfo)
    print("jw:", jw)

    site_map = {"p0":0,"p1":1}
    print("SOP", jw.to_SOP(site_map,2), sys)
    M = dense_from_sop(jw.to_SOP(site_map,2), sys)

    print(M)
    print(M.conj().T)
    assert np.allclose(M, M.conj().T)


def test_lcsop_jw_long_range():
    ordering = ["p0","p1","p2","p3"]
    sysinfo = make_fermion_sysinfo(4)
    sys = build_sys(sysinfo, ordering)

    @operator(N=4)
    def H():
        return fop("cdag","p0") * fop("c","p3")

    h = H()
    jw = h.jordan_wigner(ordering, sysinfo)

    site_map = {l:i for i,l in enumerate(ordering)}
    M = dense_from_sop(jw.to_SOP(site_map,4), sys)

    assert np.all(np.isfinite(M))


def test_lcsop_jw_ordering_dependence():
    ordering1 = ["p0","p1","p2","p3"]
    ordering2 = ["p3","p2","p1","p0"]

    sysinfo1 = make_fermion_sysinfo(4)
    sys1 = build_sys(sysinfo1, ordering1)

    sysinfo2 = make_fermion_sysinfo(4)
    sys2 = build_sys(sysinfo2, ordering2)

    @operator(N=4)
    def H():
        return fop("cdag","p0") * fop("c","p3")

    h = H()

    jw1 = h.jordan_wigner(ordering1, sysinfo1)
    jw2 = h.jordan_wigner(ordering2, sysinfo2)

    M1 = dense_from_sop(jw1.to_SOP({l:i for i,l in enumerate(ordering1)},4), sys1)
    M2 = dense_from_sop(jw2.to_SOP({l:i for i,l in enumerate(ordering2)},4), sys2)

    assert not np.allclose(M1, M2)