import pytest

from pyttn.ttnpp import spin_mode, boson_mode, fermion_mode

from pyttn.ttns.sop.system_information import SystemInfo


def make_primitive_sysinfo():
    sysinfo = SystemInfo()
    sysinfo["a"] = fermion_mode()
    sysinfo["b"] = fermion_mode()
    sysinfo["c"] = fermion_mode()
    sysinfo["d"] = fermion_mode()
    return sysinfo


def make_composite_sysinfo():
    sysinfo = SystemInfo({
        "p0": {"a": fermion_mode(), "b": fermion_mode()},
        "p1": {"c": fermion_mode(), "d": fermion_mode()},
    })
    return sysinfo

def test_systeminfo_basic():
    sysinfo = SystemInfo()

    m = spin_mode(2)

    sysinfo["a"] = m

    assert "a" in sysinfo
    assert sysinfo.nprimitive() == 1
    assert sysinfo.composite_labels() == ["a"]
    assert sysinfo.primitive_labels("a") == ["a"]


def test_systeminfo_dict_input():
    sysinfo = SystemInfo()

    m = spin_mode(2)

    sysinfo["a"] = {"p0": m}

    assert sysinfo.nprimitive() == 1
    assert sysinfo.primitive_labels("a") == ["p0"]


def test_systeminfo_build_system_modes():
    sysinfo = SystemInfo()

    m = spin_mode(2)

    sysinfo["a"] = {"p0": m}
    sysinfo["b"] = {"p1": m}

    out = sysinfo.build_system_modes(["a", "b"])

    assert len(out["primitive_labels"]) == 2
    assert out["system_modes"].nmodes() == 2


def test_systeminfo_errors():
    sysinfo = SystemInfo()

    m = spin_mode(2)
    sysinfo["a"] = m

    with pytest.raises(ValueError):
        sysinfo.build_system_modes(["a", "b"])



def test_build_flattened_modes_distinct_dimensions():
    """
    Each primitive mode has a unique local dimension.

    This allows us to verify ordering by checking that the dimensions
    appear in the correct order in the resulting system_modes object.
    """

    sysinfo = SystemInfo({
        "p0": {
            "a": boson_mode(2),
            "b": boson_mode(3),
        },
        "p1": {
            "c": boson_mode(4),
        },
        "p2": {
            "d": boson_mode(5),
        },
    })

    ordering = ["b", "d", "a", "c"]

    sys = sysinfo.build_flattened_modes(ordering)

    # Expected dimensions corresponding to ordering
    expected_dims = [3, 5, 2, 4]

    assert sys.nmodes() == len(ordering)

    # Check each mode has the expected local Hilbert space dimension
    for i, expected_dim in enumerate(expected_dims):
        actual_dim = sys[i].lhd()  
        assert actual_dim == expected_dim, (
            f"Mode {i} has dim {actual_dim}, expected {expected_dim}"
        )



def test_group_modes_basic_primitives():
    sysinfo = make_primitive_sysinfo()

    grouped = sysinfo.group_modes({
        "site0": ["a", "c"],
        "site1": ["b", "d"],
    })

    assert set(grouped.composite_labels()) == {"site0", "site1"}

    assert set(grouped["site0"].keys()) == {"a", "c"}
    assert set(grouped["site1"].keys()) == {"b", "d"}

def test_group_modes_from_composites():
    sysinfo = make_composite_sysinfo()

    grouped = sysinfo.group_modes({
        "left": ["a", "c"],
        "right": ["b", "d"],
    })

    assert set(grouped.composite_labels()) == {"left", "right"}

    assert set(grouped["left"].keys()) == {"a", "c"}
    assert set(grouped["right"].keys()) == {"b", "d"}

def test_group_modes_partial_grouping():
    sysinfo = make_primitive_sysinfo()

    grouped = sysinfo.group_modes({
        "pair": ["a", "b"],
    })

    labels = set(grouped.composite_labels())

    # grouped + ungrouped primitives
    assert labels == {"pair", "c", "d"}

    assert set(grouped["pair"].keys()) == {"a", "b"}

    # ungrouped appear as singles
    assert set(grouped["c"].keys()) == {"c"}
    assert set(grouped["d"].keys()) == {"d"}


def test_group_modes_identity():
    sysinfo = make_primitive_sysinfo()

    grouped = sysinfo.group_modes({
        "a": ["a"],
        "b": ["b"],
        "c": ["c"],
        "d": ["d"],
    })

    assert set(grouped.composite_labels()) == {"a", "b", "c", "d"}

    for label in ["a", "b", "c", "d"]:
        assert set(grouped[label].keys()) == {label}

def test_group_modes_unknown_primitive():
    sysinfo = make_primitive_sysinfo()

    with pytest.raises(ValueError):
        sysinfo.group_modes({
            "bad": ["a", "z"],  # z not present
        })

def test_group_modes_duplicate_assignment():
    sysinfo = make_primitive_sysinfo()

    with pytest.raises(ValueError):
        sysinfo.group_modes({
            "g1": ["a", "b"],
            "g2": ["b", "c"],  # 'b' used twice
        })

def test_group_modes_invalid_group_type():
    sysinfo = make_primitive_sysinfo()

    with pytest.raises(ValueError):
        sysinfo.group_modes({
            "g1": "a",  # not a list
        })

def test_group_modes_cross_composite():
    sysinfo = make_composite_sysinfo()

    grouped = sysinfo.group_modes({
        "mixed": ["a", "d"],
    })

    labels = set(grouped.composite_labels())

    # a,d grouped; b,c left separate
    assert labels == {"mixed", "b", "c"}

    assert set(grouped["mixed"].keys()) == {"a", "d"}
    assert set(grouped["b"].keys()) == {"b"}
    assert set(grouped["c"].keys()) == {"c"}

def test_group_modes_preserves_primitive_count():
    sysinfo = make_composite_sysinfo()
    grouped = sysinfo.group_modes({"g": ["a", "b", "c", "d"]})

    total = sum(len(v) for v in grouped.as_dict().values())

    assert total == 4  # same as original number of primitives


def test_liouville_space_paired_single_modes():

    sys = SystemInfo()

    sys["s0"] = spin_mode(2)
    sys["s1"] = spin_mode(2)

    L = sys.liouville_space()

    assert set(L.composite_labels()) == {"s0", "s1"}

    assert list(L["s0"].keys()) == ["s0", "s0~"]
    assert list(L["s1"].keys()) == ["s1", "s1~"]


def test_liouville_space_none_single_modes():

    sys = SystemInfo()

    sys["s0"] = spin_mode(2)
    sys["s1"] = spin_mode(2)

    L = sys.liouville_space(grouping="none")

    assert set(L.composite_labels()) == {"s0","s1","s0~","s1~",}
    assert list(L["s0"].keys()) == ["s0"]
    assert list(L["s0~"].keys()) == ["s0~"]


def test_liouville_space_preserves_composite_grouping_paired():

    sys = SystemInfo()

    sys["A"] = {"p0": spin_mode(2),"p1": spin_mode(2),}

    sys["B"] = {"p2": spin_mode(2),}

    L = sys.liouville_space(grouping="paired")

    assert set(L.composite_labels()) == {"A", "B"}
    assert list(L["A"].keys()) == ["p0","p1","p0~","p1~",]
    assert list(L["B"].keys()) == ["p2","p2~",]


def test_liouville_space_preserves_composite_grouping_none():
    sys = SystemInfo()
    sys["A"] = { "p0": spin_mode(2), "p1": spin_mode(2), }
    sys["B"] = {"p2": spin_mode(2),}
    L = sys.liouville_space(grouping="none")
    assert set(L.composite_labels()) == { "A", "B", "A~", "B~", }
    assert list(L["A"].keys()) == ["p0","p1",]
    assert list(L["A~"].keys()) == ["p0~","p1~",]
    assert list(L["B"].keys()) == ["p2",]
    assert list(L["B~"].keys()) == ["p2~",]


def test_liouville_space_custom_suffix():
    sys = SystemInfo()
    sys["s0"] = spin_mode(2)
    L = sys.liouville_space( grouping="paired", suffix="_R",)
    assert list(L["s0"].keys()) == ["s0","s0_R",]


def test_liouville_space_preserves_local_dimension():
    sys = SystemInfo()
    sys["A"] = {"p0": spin_mode(2),"p1": spin_mode(2),}
    L = sys.liouville_space(grouping="paired")
    assert sys.local_dim("A") == 4
    assert L.local_dim("A") == 16


def test_liouville_space_empty():
    sys = SystemInfo()
    L = sys.liouville_space()
    assert len(L) == 0


def test_liouville_space_invalid_grouping():
    sys = SystemInfo()
    sys["s0"] = spin_mode(2)
    with pytest.raises(ValueError):
        sys.liouville_space("banana")
