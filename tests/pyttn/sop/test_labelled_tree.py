
import pytest


from pyttn import TopoTree
from pyttn import ntree


def make_simple_tree():
    t = ntree()
    t.insert(0)
    t().insert(1)
    t().insert(2)
    return t


def test_topotree_basic():
    t = make_simple_tree()

    topo = TopoTree(t, ["a", "b"])

    assert topo.leaf_order() == ["a", "b"]

    smap = topo.site_map()

    assert smap["a"] == 0
    assert smap["b"] == 1


def test_topotree_label_mismatch():
    t = make_simple_tree()

    with pytest.raises(ValueError):
        TopoTree(t, ["a"])  # wrong number


def test_topotree_insert_subtree_simple():
    t = make_simple_tree()
    topo = TopoTree(t, ["a", "b"])

    sub = ntree()
    sub.insert(3)
    sub().insert(4)
    sub().insert(5)

    topo.insert_subtree([0], subtree=sub, subtree_labels=["c", "d"])

    assert "c" in topo.leaf_labels
    assert "d" in topo.leaf_labels

