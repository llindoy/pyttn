import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest

from pyttn import tls_mode
from pyttn.oqs import BosonicBath
from pyttn.oqs.baths.bath import Bath
from pyttn.oqs.model import OQSModel
from pyttn.oqs.topology import (
    attach_bath_placeholders,
    build_joint_interaction_graph,
    default_bath_weight,
    propose_joint_topology,
    propose_system_topology,
    propose_topology,
)
from pyttn.ttns.sop import OperatorBuilder, SystemInfo


class _DummyBath(Bath):
    """A bath object with no real physics, used to exercise topology attachment only."""

    def __init__(self):
        pass

    def expfit(self, *args, **kwargs):
        raise NotImplementedError

    def discretise(self, *args, **kwargs):
        raise NotImplementedError


def _chain_model(n=4):
    sysinfo = SystemInfo()
    for i in range(n):
        sysinfo[f"s{i}"] = tls_mode()

    b = OperatorBuilder()
    expr = b.op("sx", "s0")
    for i in range(n - 1):
        term = b.op("sz", f"s{i}") * b.op("sz", f"s{i + 1}")
        expr = expr + term
    H = b.wrap(expr).to_lCSOP()

    return OQSModel(system_info=sysinfo, system_generator=H)


def _coupling(*sites):
    b = OperatorBuilder()
    expr = None
    for site in sites:
        term = b.op("sz", site)
        expr = term if expr is None else expr + term
    return b.wrap(expr).to_lCSOP()


def test_propose_system_topology_covers_all_composites():
    model = _chain_model(4)
    topo = propose_system_topology(model)
    assert set(topo.leaf_order()) == {"s0", "s1", "s2", "s3"}
    assert topo.tree.nleaves() == 4


def test_propose_system_topology_single_composite():
    model = _chain_model(1)
    topo = propose_system_topology(model)
    assert topo.leaf_order() == ["s0"]
    assert topo.tree.nleaves() == 1


def test_attach_bath_single_composite_promotes_leaf():
    model = _chain_model(4)
    model.add_bath(_DummyBath(), _coupling("s0"), tag="bathA")

    topo = propose_system_topology(model)
    full = attach_bath_placeholders(model, topo)

    assert set(full.leaf_order()) == {"s0", "s1", "s2", "s3", "bathA"}
    assert full.tree.nleaves() == 5


def test_attach_bath_single_composite_at_root():
    # a single-composite system's only node is both root and leaf; attaching there
    # exercises the ntree.at([]) root-access edge case.
    model = _chain_model(1)
    model.add_bath(_DummyBath(), _coupling("s0"), tag="bathA")

    full = propose_topology(model)
    assert set(full.leaf_order()) == {"s0", "bathA"}


def test_attach_bath_multiple_composites_common_ancestor():
    model = _chain_model(4)
    model.add_bath(_DummyBath(), _coupling("s1", "s3"), tag="bathB")

    full = propose_topology(model)
    assert set(full.leaf_order()) == {"s0", "s1", "s2", "s3", "bathB"}
    assert full.tree.nleaves() == 5


def test_attach_multiple_baths():
    model = _chain_model(4)
    model.add_bath(_DummyBath(), _coupling("s0"), tag="bathA")
    model.add_bath(_DummyBath(), _coupling("s1", "s3"), tag="bathB")

    full = propose_topology(model)
    assert set(full.leaf_order()) == {"s0", "s1", "s2", "s3", "bathA", "bathB"}
    assert full.tree.nleaves() == 6


def test_attach_bath_unknown_composite_raises():
    model = _chain_model(4)
    b = OperatorBuilder()
    op = b.wrap(b.op("sz", "not_in_tree")).to_lCSOP()
    model.add_bath(_DummyBath(), op, tag="bad")

    # a topology that doesn't cover the coupling operator's site should raise
    topo = propose_system_topology(model)
    with pytest.raises(ValueError):
        attach_bath_placeholders(model, topo)


def test_propose_topology_reuses_existing_placeholder():
    model = _chain_model(4)
    model.add_bath(_DummyBath(), _coupling("s0"), tag="bathA")

    full = propose_topology(model)
    # calling attach_bath_placeholders again on an already-augmented topology is a no-op
    again = attach_bath_placeholders(model, full)
    assert again.leaf_order() == full.leaf_order()


def _ohmic_bath(alpha=1.0, wc=5.0):
    def J(w):
        return np.abs(np.pi / 2 * alpha * wc * (w / wc) * np.exp(-np.abs(w / wc))) * np.where(w > 0, 1.0, -1.0)

    return BosonicBath(J, beta=None)


def test_default_bath_weight_requires_reorganisation_energy():
    model = _chain_model(2)
    model.add_bath(_DummyBath(), _coupling("s0"), tag="bathA")
    spec = model.baths[0]

    with pytest.raises(AttributeError):
        default_bath_weight(spec, model.system_info)


def test_default_bath_weight_formula():
    model = _chain_model(2)
    bath = _ohmic_bath(alpha=1.0, wc=5.0)
    coupling = OperatorBuilder()
    op = coupling.wrap(0.5 * coupling.op("sz", "s0")).to_lCSOP()
    model.add_bath(bath, op, tag="bathA")
    spec = model.baths[0]

    weights = default_bath_weight(spec, model.system_info)
    lam = bath.reorganisation_energy()
    expected = np.sqrt(lam * (0.5 ** 2))
    assert set(weights.keys()) == {"s0"}
    assert weights["s0"] == pytest.approx(expected)


def test_build_joint_interaction_graph_includes_bath_nodes():
    model = _chain_model(4)
    model.add_bath(_ohmic_bath(), _coupling("s0"), tag="bathA")

    graph = build_joint_interaction_graph(model)
    labels, W = graph.weight_matrix()
    assert set(labels) == {"s0", "s1", "s2", "s3", "bathA"}
    i = labels.index("s0")
    j = labels.index("bathA")
    assert W[i, j] > 0


def test_build_joint_interaction_graph_unknown_composite_raises():
    model = _chain_model(4)
    b = OperatorBuilder()
    op = b.wrap(b.op("sz", "not_in_tree")).to_lCSOP()
    model.add_bath(_ohmic_bath(), op, tag="bad")

    with pytest.raises(ValueError):
        build_joint_interaction_graph(model)


def test_propose_joint_topology_single_pass():
    model = _chain_model(4)
    model.add_bath(_ohmic_bath(), _coupling("s0"), tag="bathA")
    model.add_bath(_ohmic_bath(alpha=0.01), _coupling("s3"), tag="bathB")

    full = propose_joint_topology(model)
    assert set(full.leaf_order()) == {"s0", "s1", "s2", "s3", "bathA", "bathB"}
    assert full.tree.nleaves() == 6


def test_propose_topology_joint_mode_dispatch():
    model = _chain_model(4)
    model.add_bath(_ohmic_bath(), _coupling("s0"), tag="bathA")

    full = propose_topology(model, bath_placement="joint")
    assert set(full.leaf_order()) == {"s0", "s1", "s2", "s3", "bathA"}


def test_propose_topology_unknown_bath_placement_raises():
    model = _chain_model(4)
    model.add_bath(_ohmic_bath(), _coupling("s0"), tag="bathA")

    with pytest.raises(ValueError):
        propose_topology(model, bath_placement="bogus")


def test_propose_topology_custom_bath_weight():
    model = _chain_model(4)
    model.add_bath(_DummyBath(), _coupling("s0"), tag="bathA")

    def custom_weight(spec, sysinfo):
        return {"s0": 42.0}

    full = propose_topology(model, bath_placement="joint", bath_weight=custom_weight)
    assert set(full.leaf_order()) == {"s0", "s1", "s2", "s3", "bathA"}


def _disconnected_model():
    # two independent, non-interacting pairs {a0,a1} and {b0,b1} - the interaction
    # graph has two connected components.
    sysinfo = SystemInfo()
    for lbl in ["a0", "a1", "b0", "b1"]:
        sysinfo[lbl] = tls_mode()

    b = OperatorBuilder()
    expr = b.op("sz", "a0") * b.op("sz", "a1") + b.op("sz", "b0") * b.op("sz", "b1")
    H = b.wrap(expr).to_lCSOP()

    return OQSModel(system_info=sysinfo, system_generator=H)


def test_disconnected_graph_previously_failed():
    # documents the failure mode this fixes: a disconnected weight matrix produces a
    # spanning forest, not a tree, and convert_nx_to_tree rejects it outright.
    import networkx as nx

    from pyttn.ttns.topology import generate_spanning_tree
    from pyttn.ttns.topology.networkx_converter import convert_nx_to_tree

    W = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
    nx_tree, root = generate_spanning_tree(W, root_index=0)
    assert not nx.is_tree(nx_tree)
    with pytest.raises(RuntimeError):
        convert_nx_to_tree(nx_tree, root)


@pytest.mark.parametrize("disconnected_strategy,degree", [("weak_link", None), ("join", 1), ("join", 2), ("join", 4)])
def test_propose_system_topology_disconnected(disconnected_strategy, degree):
    model = _disconnected_model()
    kwargs = {"disconnected_strategy": disconnected_strategy}
    if degree is not None:
        kwargs["degree"] = degree

    topo = propose_system_topology(model, **kwargs)
    assert set(topo.leaf_order()) == {"a0", "a1", "b0", "b1"}
    assert topo.tree.nleaves() == 4


def test_propose_system_topology_disconnected_three_components_with_singletons():
    # mixes a 2-node component with two isolated single-node components, exercising
    # the leaf-vs-internal-node promotion subtlety in the "join" backbone builders.
    sysinfo = SystemInfo()
    for lbl in ["a0", "a1", "b0", "c0"]:
        sysinfo[lbl] = tls_mode()
    b = OperatorBuilder()
    expr = b.op("sz", "a0") * b.op("sz", "a1") + b.op("sx", "b0") + b.op("sx", "c0")
    H = b.wrap(expr).to_lCSOP()
    model = OQSModel(system_info=sysinfo, system_generator=H)

    for disconnected_strategy, degree in [("weak_link", None), ("join", 1), ("join", 2), ("join", 4)]:
        kwargs = {"disconnected_strategy": disconnected_strategy}
        if degree is not None:
            kwargs["degree"] = degree
        topo = propose_system_topology(model, **kwargs)
        assert set(topo.leaf_order()) == {"a0", "a1", "b0", "c0"}, (disconnected_strategy, degree)
        assert topo.tree.nleaves() == 4, (disconnected_strategy, degree)


def test_propose_system_topology_disconnected_unknown_strategy_raises():
    model = _disconnected_model()
    with pytest.raises(ValueError):
        propose_system_topology(model, disconnected_strategy="bogus")


def test_propose_system_topology_disconnected_invalid_degree_raises():
    model = _disconnected_model()
    with pytest.raises(ValueError):
        propose_system_topology(model, disconnected_strategy="join", degree=0)


def test_propose_joint_topology_disconnected_with_bath():
    model = _disconnected_model()
    model.add_bath(_ohmic_bath(), _coupling("a1"), tag="phonon")

    full = propose_joint_topology(model)
    assert set(full.leaf_order()) == {"a0", "a1", "b0", "b1", "phonon"}

    full_join = propose_joint_topology(model, disconnected_strategy="join", degree=4)
    assert set(full_join.leaf_order()) == {"a0", "a1", "b0", "b1", "phonon"}


def test_propose_topology_disconnected_via_dispatcher():
    model = _disconnected_model()
    full = propose_topology(model, disconnected_strategy="join", degree=1)
    assert set(full.leaf_order()) == {"a0", "a1", "b0", "b1"}