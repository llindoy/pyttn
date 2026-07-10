import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest

import pyttn
from pyttn import tls_mode
from pyttn.oqs import BosonicBath, DiscreteBosonicBath, ESPRITDecomposition, FermionicBath, OrthopolDiscretisation
from pyttn.oqs.method_builder import Method, MethodBuilder
from pyttn.oqs.model import OQSModel
from pyttn.ttns.sop import OperatorBuilder, SystemInfo

ALPHA, WC, S = 1.0, 5.0, 1
DELTA, EPS = 0.5, 0.0


def _spectral_density(w):
    return np.abs(np.pi / 2 * ALPHA * WC * np.power(w / WC, S) * np.exp(-np.abs(w / WC))) * np.where(w > 0, 1.0, -1.0)


def _spin_boson_model():
    sysinfo = SystemInfo()
    sysinfo["spin"] = tls_mode()

    b = OperatorBuilder()
    H = b.wrap(DELTA * b.op("sx", "spin") + EPS * b.op("sz", "spin")).to_lCSOP()

    model = OQSModel(system_info=sysinfo, system_generator=H)

    bath = BosonicBath(_spectral_density, beta=None)
    coupling = OperatorBuilder()
    A = coupling.wrap(coupling.op("sz", "spin")).to_lCSOP()

    model.add_bath(
        bath,
        A,
        tag="phonon",
        params={
            "unitary": {"decomposition": OrthopolDiscretisation(16, bath.find_wmin(8 * WC), 8 * WC), "degree": 2, "chi0": 4, "chi": 16},
            "tedopa": {"decomposition": OrthopolDiscretisation(16, bath.find_wmin(8 * WC), 8 * WC), "degree": 1, "chi0": 4, "chi": 16},
            "heom": {"decomposition": ESPRITDecomposition(K=6, tmax=2.4, Nt=49), "degree": 2, "chi0": 8, "chi": 32},
            "pseudomode": {"decomposition": ESPRITDecomposition(K=4, tmax=1.6, Nt=33), "degree": 2, "chi0": 8, "chi": 32},
        },
    )
    return model


@pytest.mark.parametrize("method", ["unitary", "tedopa", "heom", "pseudomode"])
def test_build_produces_consistent_result(method):
    model = _spin_boson_model()
    result = MethodBuilder(model).build(method)

    assert result.topology.tree.nleaves() == result.capacity.tree.nleaves()
    assert set(result.topology.leaf_order()) == set(result.capacity.leaf_order())
    assert result.system_modes.nprimitive_modes() == len(result.site_map)
    assert result.generator.nmodes() == len(result.site_map)
    assert "phonon" not in result.baths or result.baths["phonon"] is not None


def test_build_accepts_method_enum():
    model = _spin_boson_model()
    result = MethodBuilder(model).build(Method.UNITARY)
    assert "spin" in result.site_map


@pytest.mark.parametrize("method", ["unitary", "tedopa"])
def test_hilbert_space_methods_have_no_trace_state(method):
    model = _spin_boson_model()
    result = MethodBuilder(model).build(method)
    assert result.trace_state is None


@pytest.mark.parametrize("method", ["heom", "pseudomode"])
def test_liouville_methods_build_trace_state(method):
    model = _spin_boson_model()
    result = MethodBuilder(model).build(method)

    assert result.trace_state is not None

    A = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    A.set_state([0 for _ in range(result.system_modes.nmodes())])
    assert result.trace_state.nmodes() == A.nmodes()
    mel = pyttn.matrix_element(A, result.trace_state)

    # a freshly initialised (vacuum bath, ground system state) density matrix has unit trace
    assert np.real(mel(A, result.trace_state)) == pytest.approx(1.0, abs=1e-8)

    sz = pyttn.site_operator(pyttn.sOP("sz", result.site_map["spin"]), result.system_modes)
    assert np.real(mel(sz, A, result.trace_state)) == pytest.approx(1.0, abs=1e-8)


def _multi_spin_bath_model():
    sysinfo = SystemInfo()
    for i in range(3):
        sysinfo[f"s{i}"] = tls_mode()

    b = OperatorBuilder()
    H = b.wrap(DELTA * b.op("sx", "s0") + b.op("sz", "s0") * b.op("sz", "s1") + b.op("sz", "s1") * b.op("sz", "s2")).to_lCSOP()
    model = OQSModel(system_info=sysinfo, system_generator=H)

    bath = BosonicBath(_spectral_density, beta=None)
    coupling = OperatorBuilder()
    A = coupling.wrap(coupling.op("sz", "s2")).to_lCSOP()
    model.add_bath(
        bath,
        A,
        tag="phonon",
        params={"unitary": {"decomposition": OrthopolDiscretisation(8, bath.find_wmin(8 * WC), 8 * WC), "degree": 2, "chi0": 4, "chi": 16}},
    )
    return model


@pytest.mark.parametrize("bath_placement", ["attach", "joint"])
def test_build_bath_placement_options(bath_placement):
    model = _multi_spin_bath_model()
    result = MethodBuilder(model).build("unitary", bath_placement=bath_placement)

    assert set(result.topology.leaf_order()) == set(result.capacity.leaf_order())
    assert result.system_modes.nprimitive_modes() == len(result.site_map)
    assert {"s0", "s1", "s2"}.issubset(result.site_map.keys())


def test_build_joint_bath_placement_runs_dynamics():
    model = _multi_spin_bath_model()
    result = MethodBuilder(model).build("unitary", bath_placement="joint")

    A_ttn = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    A_ttn.set_state([0] * result.system_modes.nmodes())
    h = pyttn.sop_operator(result.generator, A_ttn, result.system_modes)
    op = pyttn.site_operator(pyttn.sOP("sz", result.site_map["s0"]), result.system_modes)
    mel = pyttn.matrix_element(A_ttn)
    sweep = pyttn.tdvp(A_ttn, h, krylov_dim=16)
    sweep.dt = 0.05
    sweep.coefficient = -1.0j

    Sz0 = np.real(mel(op, A_ttn, A_ttn))
    sweep.step(A_ttn, h)
    Sz1 = np.real(mel(op, A_ttn, A_ttn))
    assert Sz0 == pytest.approx(1.0)
    assert Sz1 < Sz0


def test_build_unknown_bath_placement_raises():
    model = _spin_boson_model()
    with pytest.raises(ValueError):
        MethodBuilder(model).build("unitary", bath_placement="bogus")


def test_build_joint_bath_placement_custom_weight():
    model = _multi_spin_bath_model()

    def custom_weight(spec, sysinfo):
        return {"s2": 5.0}

    result = MethodBuilder(model).build("unitary", bath_placement="joint", bath_weight=custom_weight)
    assert "phonon_0" in result.site_map


def test_unitary_build_matches_hand_built_dynamics():
    model = _spin_boson_model()
    result = MethodBuilder(model).build("unitary")

    A_ttn = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    A_ttn.set_state([0] * result.system_modes.nmodes())
    h = pyttn.sop_operator(result.generator, A_ttn, result.system_modes)
    op = pyttn.site_operator(pyttn.sOP("sz", result.site_map["spin"]), result.system_modes)
    mel = pyttn.matrix_element(A_ttn)
    sweep = pyttn.tdvp(A_ttn, h, krylov_dim=16)
    sweep.dt = 0.05
    sweep.coefficient = -1.0j

    nstep = 10
    Sz_builder = np.zeros(nstep + 1)
    Sz_builder[0] = np.real(mel(op, A_ttn, A_ttn))
    for i in range(nstep):
        sweep.step(A_ttn, h)
        Sz_builder[i + 1] = np.real(mel(op, A_ttn, A_ttn))

    # hand-built reference, mirroring tutorials/open_quantum_systems/oqs_general.ipynb
    bath = BosonicBath(_spectral_density, beta=None)
    g, w = bath.discretise(OrthopolDiscretisation(16, bath.find_wmin(8 * WC), 8 * WC))
    discbath = DiscreteBosonicBath(g, w)
    discbath.truncate_modes()

    sysinf2 = pyttn.system_modes(1)
    sysinf2[0] = pyttn.tls_mode()
    sysinf2 = pyttn.combine_systems(sysinf2, discbath.system_information())

    H2 = pyttn.SOP(17)
    H2 += DELTA * pyttn.sOP("sx", 0) + EPS * pyttn.sOP("sz", 0)
    H2 = discbath.add_system_bath_hamiltonian(H2, pyttn.sOP("sz", 0), geom="star")

    topo = pyttn.ntree("(1(2(2)))")
    capacity = pyttn.ntree("(1(2(2)))")
    discbath.add_bath_tree(topo(), 2, 4, 4)
    discbath.add_bath_tree(capacity(), 2, 16, 16)

    A2 = pyttn.ttn(topo, capacity, dtype=np.complex128)
    A2.set_state([0] * sysinf2.nmodes())
    h2 = pyttn.sop_operator(H2, A2, sysinf2)
    op2 = pyttn.site_operator(pyttn.sOP("sz", 0), sysinf2)
    mel2 = pyttn.matrix_element(A2)
    sweep2 = pyttn.tdvp(A2, h2, krylov_dim=16)
    sweep2.dt = 0.05
    sweep2.coefficient = -1.0j

    Sz_manual = np.zeros(nstep + 1)
    Sz_manual[0] = np.real(mel2(op2, A2, A2))
    for i in range(nstep):
        sweep2.step(A2, h2)
        Sz_manual[i + 1] = np.real(mel2(op2, A2, A2))

    assert np.max(np.abs(Sz_builder - Sz_manual)) < 1e-3


def test_heom_and_pseudomode_agree_at_short_time():
    # HEOM and pseudomode are equivalent up to a similarity transform; at short times
    # (before bath-specific truncation differences accumulate) their magnetisation
    # traces should closely agree, and both should track the unitary/coherent trace.
    model = _spin_boson_model()

    traces = {}
    for method in ("unitary", "heom", "pseudomode"):
        result = MethodBuilder(model).build(method)
        A_ttn = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
        A_ttn.set_state([0] * result.system_modes.nmodes())
        h = pyttn.sop_operator(result.generator, A_ttn, result.system_modes)
        op = pyttn.site_operator(pyttn.sOP("sz", result.site_map["spin"]), result.system_modes)
        mel = pyttn.matrix_element(A_ttn)

        if method == "unitary":
            ref_ttn = A_ttn
        else:
            raw_bath = result.baths["phonon"]
            id_state = [np.identity(2, dtype=np.complex128).flatten()] + raw_bath.identity_product_state(method=method)
            ref_ttn = pyttn.ttn(result.topology.tree, dtype=np.complex128)
            ref_ttn.set_product(id_state)

        sweep = pyttn.tdvp(A_ttn, h, krylov_dim=12, expansion="subspace", subspace_neigs=4, subspace_krylov_dim=8)
        sweep.dt = 0.05
        sweep.coefficient = -1.0j

        Sz = np.zeros(4)
        Sz[0] = np.real(mel(op, A_ttn, ref_ttn))
        for i in range(3):
            sweep.step(A_ttn, h)
            Sz[i + 1] = np.real(mel(op, A_ttn, ref_ttn))
        traces[method] = Sz

    assert np.max(np.abs(traces["heom"] - traces["pseudomode"])) < 1e-2
    assert np.max(np.abs(traces["heom"] - traces["unitary"])) < 1e-2


def test_unitary_requires_hilbert_representation():
    sysinfo = SystemInfo()
    sysinfo["spin"] = {"spin": tls_mode(), "spin~": tls_mode()}
    b = OperatorBuilder()
    H = b.wrap(b.op("sz", "spin") - b.op("sz", "spin~")).to_lCSOP()

    from pyttn.oqs.model import Representation

    model = OQSModel(system_info=sysinfo, system_generator=H, representation=Representation.LIOUVILLE)
    with pytest.raises(ValueError):
        MethodBuilder(model).build("unitary")


def test_too_many_coupling_channels_rejected():
    model = _spin_boson_model()
    spec = model.baths[0]
    b = OperatorBuilder()
    # nchannels()==2 is a valid (asymmetric raising/lowering) convention; a third
    # channel is what should be rejected as an unsupported correlated-channel bath.
    spec.coupling_ops.append(b.wrap(b.op("sx", "spin")).to_lCSOP())
    spec.coupling_ops.append(b.wrap(b.op("sy", "spin")).to_lCSOP())

    with pytest.raises(NotImplementedError):
        MethodBuilder(model).build("unitary")


def _disconnected_bath_model():
    # two independent, non-interacting spin pairs; the system interaction graph has
    # two connected components, and a bath couples to just one of them.
    sysinfo = SystemInfo()
    for lbl in ["a0", "a1", "b0", "b1"]:
        sysinfo[lbl] = tls_mode()

    b = OperatorBuilder()
    H = b.wrap(DELTA * b.op("sx", "a0") + b.op("sz", "a0") * b.op("sz", "a1") + DELTA * b.op("sx", "b0") + b.op("sz", "b0") * b.op("sz", "b1")).to_lCSOP()
    model = OQSModel(system_info=sysinfo, system_generator=H)

    bath = BosonicBath(_spectral_density, beta=None)
    coupling = OperatorBuilder()
    A = coupling.wrap(coupling.op("sz", "a1")).to_lCSOP()
    model.add_bath(
        bath,
        A,
        tag="phonon",
        params={"unitary": {"decomposition": OrthopolDiscretisation(8, bath.find_wmin(8 * WC), 8 * WC), "degree": 2, "chi0": 4, "chi": 16}},
    )
    return model


@pytest.mark.parametrize(
    "bath_placement,disconnected_strategy,degree",
    [
        ("attach", "weak_link", None),
        ("attach", "join", 1),
        ("attach", "join", 4),
        ("joint", "weak_link", None),
        ("joint", "join", 1),
        ("joint", "join", 4),
    ],
)
def test_build_disconnected_system(bath_placement, disconnected_strategy, degree):
    model = _disconnected_bath_model()
    kwargs = {"bath_placement": bath_placement, "disconnected_strategy": disconnected_strategy}
    if degree is not None:
        kwargs["degree"] = degree

    result = MethodBuilder(model).build("unitary", **kwargs)
    expected_system = {"a0", "a1", "b0", "b1"}
    assert expected_system.issubset(result.site_map.keys())
    assert result.topology.tree.nleaves() == result.capacity.tree.nleaves()

    A_ttn = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    A_ttn.set_state([0] * result.system_modes.nmodes())
    h = pyttn.sop_operator(result.generator, A_ttn, result.system_modes)
    op = pyttn.site_operator(pyttn.sOP("sz", result.site_map["a0"]), result.system_modes)
    mel = pyttn.matrix_element(A_ttn)
    sweep = pyttn.tdvp(A_ttn, h, krylov_dim=16)
    sweep.dt = 0.05
    sweep.coefficient = -1.0j

    Sz0 = np.real(mel(op, A_ttn, A_ttn))
    sweep.step(A_ttn, h)
    Sz1 = np.real(mel(op, A_ttn, A_ttn))
    assert Sz0 == pytest.approx(1.0)
    assert Sz1 < Sz0


def _flat_band(w, W=10.0, gamma=1.0):
    return np.pi * np.where(np.abs(w / W) < 1, gamma * np.sqrt(1 - (w / W) ** 2) / (np.pi * W), 0) * 10


def _split_fermionic_model(Nb=3, attachment="branch", ordering="filled_first", W=10.0, eps=-0.1):
    sysinfo = SystemInfo()
    sysinfo["c"] = pyttn.fermion_mode()
    b = OperatorBuilder()
    model = OQSModel(system_info=sysinfo, system_generator=b.wrap(eps * b.fop("n", "c")))

    bath = FermionicBath(_flat_band, beta=None)
    coupling = OperatorBuilder()
    decomp_f = OrthopolDiscretisation(Nb, *bath.estimate_bounds(wmax=W, sigma="+"))
    decomp_e = OrthopolDiscretisation(Nb, *bath.estimate_bounds(wmax=W, sigma="-"))
    params = {"decomposition": (decomp_f, decomp_e), "channels": "filled_empty", "Ef": 0.0, "attachment": attachment, "ordering": ordering, "chi0": 4, "chi": 8, "geom": "star"}
    model.add_bath(bath, [coupling.wrap(coupling.fop("cdag", "c")), coupling.wrap(coupling.fop("c", "c"))], tag="lead", params=params)
    return model, coupling


def _run_one_step(result, coupling, dt=0.01):
    A = pyttn.ttn(result.topology.tree, result.capacity.tree, dtype=np.complex128)
    A.set_state([1] + [0] * (result.system_modes.nmodes() - 1))
    h = pyttn.sop_operator(result.generator, A, result.system_modes)
    mel = pyttn.matrix_element(A)
    n_op = result.jordan_wigner(coupling.wrap(coupling.fop("n", "c")))
    n_compiled = pyttn.sop_operator(n_op.compile(result.site_map, len(result.site_map)), A, result.system_modes)

    n0 = np.real(mel(n_compiled, A))
    sweep = pyttn.tdvp(A, h, krylov_dim=8)
    sweep.dt, sweep.coefficient = dt, -1.0j
    sweep.step(A, h)
    n1 = np.real(mel(n_compiled, A))
    return n0, n1


def test_split_bath_requires_fermionic_bath():
    sysinfo = SystemInfo()
    sysinfo["spin"] = tls_mode()
    b = OperatorBuilder()
    model = OQSModel(system_info=sysinfo, system_generator=b.wrap(DELTA * b.op("sx", "spin")))
    bath = BosonicBath(_spectral_density, beta=None)
    coupling = OperatorBuilder()
    params = {"decomposition": OrthopolDiscretisation(8, bath.find_wmin(8 * WC), 8 * WC), "channels": "filled_empty", "chi0": 4, "chi": 16}
    model.add_bath(bath, coupling.wrap(coupling.op("sz", "spin")), tag="phonon", params=params)
    with pytest.raises(ValueError):
        MethodBuilder(model).build("unitary", jordan_wigner_ordering="tree")


def test_fermionic_model_without_jw_ordering_raises():
    model, _ = _split_fermionic_model()
    with pytest.raises(ValueError):
        MethodBuilder(model).build("unitary")


def test_fermionic_model_auto_jw_ordering_not_implemented():
    model, _ = _split_fermionic_model()
    with pytest.raises(NotImplementedError):
        MethodBuilder(model).build("unitary", jordan_wigner_ordering="auto")


def test_jordan_wigner_on_non_fermionic_build_raises():
    model = _spin_boson_model()
    result = MethodBuilder(model).build("unitary")
    b = OperatorBuilder()
    with pytest.raises(ValueError):
        result.jordan_wigner(b.wrap(b.op("sz", "spin")))


def test_split_bath_branch_attachment_runs_dynamics():
    model, coupling = _split_fermionic_model(attachment="branch")
    result = MethodBuilder(model).build("unitary", min_chi=4, max_chi=8, jordan_wigner_ordering="tree")

    assert result.trace_state is None
    assert result.jordan_wigner_ordering is not None
    assert result.system_info is not None

    n0, n1 = _run_one_step(result, coupling)
    assert n0 == pytest.approx(1.0)
    assert n1 < n0
    assert n1 == pytest.approx(1.0, abs=1e-2)  # small dt, should barely have decayed


@pytest.mark.parametrize("ordering", ["filled_first", "interleaved"])
def test_split_bath_merge_attachment_matches_branch(ordering):
    model_branch, coupling_branch = _split_fermionic_model(attachment="branch")
    result_branch = MethodBuilder(model_branch).build("unitary", min_chi=4, max_chi=8, jordan_wigner_ordering="tree")
    n0_branch, n1_branch = _run_one_step(result_branch, coupling_branch)

    model_merge, coupling_merge = _split_fermionic_model(attachment="merge", ordering=ordering)
    result_merge = MethodBuilder(model_merge).build("unitary", min_chi=4, max_chi=8, jordan_wigner_ordering="tree")
    n0_merge, n1_merge = _run_one_step(result_merge, coupling_merge)

    assert n0_merge == pytest.approx(n0_branch, abs=1e-8)
    assert n1_merge == pytest.approx(n1_branch, abs=1e-8)


def test_split_bath_merge_custom_ordering():
    ordering = [("empty", 0), ("filled", 0), ("empty", 1), ("filled", 1), ("empty", 2), ("filled", 2)]
    model, coupling = _split_fermionic_model(attachment="merge", ordering=ordering)
    result = MethodBuilder(model).build("unitary", min_chi=4, max_chi=8, jordan_wigner_ordering="tree")

    # leaf order should be "c" (system) followed by lead_0..lead_5 in the custom order
    assert result.topology.leaf_order() == ["c"] + [f"lead_{i}" for i in range(6)]

    n0, n1 = _run_one_step(result, coupling)
    assert n0 == pytest.approx(1.0)
    assert n1 < n0


def test_split_bath_merge_invalid_ordering_raises():
    model, _ = _split_fermionic_model(attachment="merge", ordering="bogus")
    with pytest.raises(ValueError):
        MethodBuilder(model).build("unitary", jordan_wigner_ordering="tree")


def test_split_bath_channels_requires_fermionic_bath_object():
    sysinfo = SystemInfo()
    sysinfo["spin"] = tls_mode()
    b = OperatorBuilder()
    model = OQSModel(system_info=sysinfo, system_generator=b.wrap(DELTA * b.op("sx", "spin")))
    bath = BosonicBath(_spectral_density, beta=None)
    coupling = OperatorBuilder()
    params = {"decomposition": OrthopolDiscretisation(8, bath.find_wmin(8 * WC), 8 * WC), "channels": "filled_empty"}
    model.add_bath(bath, coupling.wrap(coupling.op("sz", "spin")), tag="phonon", params=params)
    with pytest.raises(ValueError):
        MethodBuilder(model).build("unitary", jordan_wigner_ordering="tree")


def test_jordan_wigner_ordering_explicit_matches_tree_ordering():
    model, coupling = _split_fermionic_model(attachment="branch")
    result_tree = MethodBuilder(model).build("unitary", min_chi=4, max_chi=8, jordan_wigner_ordering="tree")
    explicit_ordering = result_tree.topology.leaf_order()

    model2, coupling2 = _split_fermionic_model(attachment="branch")
    result_explicit = MethodBuilder(model2).build("unitary", min_chi=4, max_chi=8, jordan_wigner_ordering=explicit_ordering)

    n0_tree, n1_tree = _run_one_step(result_tree, coupling)
    n0_explicit, n1_explicit = _run_one_step(result_explicit, coupling2)
    assert n0_explicit == pytest.approx(n0_tree)
    assert n1_explicit == pytest.approx(n1_tree)
