import os

os.environ["OMP_NUM_THREADS"] = "1"

import copy

import numpy as np
import pytest

from pyttn import SOP, dmrg, matrix_element, ntreeBuilder, sOP, site_operator, sop_operator, system_modes, tdvp, tls_mode, ttn
from pyttn.simulation import DMRGSimulation, Observable, TDVPSimulation
from pyttn.ttns.sop import OperatorBuilder, SystemInfo


def _labelled_tfim(N, J=1.0, h=1.0):
    sysinfo = SystemInfo()
    labels = [f"s{i}" for i in range(N)]
    for label in labels:
        sysinfo[label] = tls_mode()

    b = OperatorBuilder()
    H = -1.0 * h * b.op("sx", labels[0])
    for i in range(1, N):
        H = H + -1.0 * h * b.op("sx", labels[i])
    for i in range(N - 1):
        H = H + -1.0 * J * b.op("sz", labels[i]) * b.op("sz", labels[i + 1])

    raw = sysinfo.build_system_modes(labels)
    return b.wrap(H), raw["system_modes"], raw["primitive_label_to_index"], labels


def _tfim(N, J=1.0, h=1.0):
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    H = SOP(N)
    for i in range(N):
        H += -1.0 * h * sOP("sx", i)
    for i in range(N - 1):
        H += -1.0 * J * sOP("sz", i) * sOP("sz", i + 1)

    return sysinf, H


def _build_state(sysinf, dims, chi, dtype=np.complex128):
    topo = ntreeBuilder.mps_tree(dims, chi)
    A = ttn(topo, dtype=dtype)
    A.random()
    A.normalise()
    return A


def test_dmrg_simulation_converges_to_known_ground_state():
    N = 16
    expected = -1.2510242438
    sysinf, H = _tfim(N)
    A = _build_state(sysinf, [2 for _ in range(N)], 16)
    h = sop_operator(H, A, sysinf)

    sim = DMRGSimulation(A, h, nsweep=10, integrator_kwargs={"krylov_dim": 12})
    results = sim.run()

    energy_per_site = np.real(results.data["E"][results.count - 1]) / N
    assert pytest.approx(energy_per_site, abs=1e-6) == expected


def test_dmrg_simulation_matches_manual_loop():
    N = 8
    sysinf, H = _tfim(N)

    A1 = _build_state(sysinf, [2 for _ in range(N)], 8)
    A2 = copy.deepcopy(A1)
    h = sop_operator(H, A1, sysinf)

    sim = DMRGSimulation(A1, h, nsweep=5, integrator_kwargs={"krylov_dim": 12})
    sim.run()

    mel2 = matrix_element(A2)
    h2 = sop_operator(H, A2, sysinf)
    sweep2 = dmrg(A2, h2, krylov_dim=12)
    for _ in range(5):
        sweep2.step(A2, h2)
    expected_energy = sweep2.E()

    assert pytest.approx(np.real(sim.integrator.E()), rel=1e-10) == pytest.approx(np.real(expected_energy), rel=1e-10)


def test_dmrg_simulation_energy_tol_stops_early():
    N = 8
    sysinf, H = _tfim(N)
    A = _build_state(sysinf, [2 for _ in range(N)], 8)
    h = sop_operator(H, A, sysinf)

    sim = DMRGSimulation(A, h, nsweep=50, energy_tol=1.0, integrator_kwargs={"krylov_dim": 12})
    sim.run()

    assert sim.results.count < 50


def test_tdvp_simulation_matches_manual_loop():
    N = 4
    sysinf, H = _tfim(N)

    A1 = _build_state(sysinf, [2 for _ in range(N)], 8)
    A2 = copy.deepcopy(A1)

    h1 = sop_operator(H, A1, sysinf)
    op1 = site_operator(sOP("sz", 0), sysinf)
    obs = Observable("sz0", op=op1)

    sim = TDVPSimulation(A1, h1, dt=0.05, nstep=6, stride=1, initial_ramp_steps=0, observables=[obs], integrator_kwargs={"krylov_dim": 12})
    results = sim.run()

    # manual reference loop replicating the same steps without the ramp
    mel2 = matrix_element(A2)
    h2 = sop_operator(H, A2, sysinf)
    op2 = site_operator(sOP("sz", 0), sysinf)
    sweep2 = tdvp(A2, h2, krylov_dim=12)
    sweep2.dt = 0.05
    sweep2.coefficient = -1.0j

    expected = [np.real(mel2(op2, A2))]
    for _ in range(6):
        sweep2.step(A2, h2)
        expected.append(np.real(mel2(op2, A2)))

    got = np.real(results.data["sz0"][: results.count])
    assert got == pytest.approx(expected, abs=1e-8)


def test_tdvp_simulation_stride_reduces_record_count():
    N = 4
    sysinf, H = _tfim(N)
    A = _build_state(sysinf, [2 for _ in range(N)], 8)
    h = sop_operator(H, A, sysinf)
    op = site_operator(sOP("sz", 0), sysinf)
    obs = Observable("sz0", op=op)

    sim = TDVPSimulation(A, h, dt=0.05, nstep=10, stride=2, initial_ramp_steps=0, observables=[obs], integrator_kwargs={"krylov_dim": 12})
    results = sim.run()

    assert results.count == 6
    assert results.t[results.count - 1] == pytest.approx(0.5)


def test_tdvp_simulation_output_file_written(tmp_path):
    pytest.importorskip("h5py")
    N = 4
    sysinf, H = _tfim(N)
    A = _build_state(sysinf, [2 for _ in range(N)], 8)
    h = sop_operator(H, A, sysinf)
    op = site_operator(sOP("sz", 0), sysinf)
    obs = Observable("sz0", op=op)

    fname = tmp_path / "out.h5"
    sim = TDVPSimulation(
        A, h, dt=0.05, nstep=4, stride=1, initial_ramp_steps=0, observables=[obs],
        integrator_kwargs={"krylov_dim": 12}, output_file=str(fname), output_stride=2,
    )
    sim.run()

    import h5py
    with h5py.File(str(fname), "r") as h5:
        assert "sz0" in h5
        assert h5["sz0"].shape[0] == 5


def test_tdvp_simulation_checkpoint_written(tmp_path):
    N = 4
    sysinf, H = _tfim(N)
    A = _build_state(sysinf, [2 for _ in range(N)], 8)
    h = sop_operator(H, A, sysinf)

    fname = tmp_path / "checkpoint.bin"
    sim = TDVPSimulation(A, h, dt=0.05, nstep=3, initial_ramp_steps=0, integrator_kwargs={"krylov_dim": 12}, checkpoint_file=str(fname))
    sim.run()

    assert fname.exists()


def test_simulation_measure_extra_values_and_states():
    N = 4
    sysinf, H = _tfim(N)
    A = _build_state(sysinf, [2 for _ in range(N)], 8)
    B = copy.deepcopy(A)
    h = sop_operator(H, A, sysinf)
    op = site_operator(sOP("sz", 0), sysinf)
    obs = Observable("sz0_AB", op=op)

    sim = TDVPSimulation(A, h, dt=0.05, nstep=1, initial_ramp_steps=0, observables=[obs], integrator_kwargs={"krylov_dim": 12}, dtype=np.complex128)
    # override mel with a two-state variant so extra_states is meaningful
    sim.mel = matrix_element(A, B)
    values = sim.measure(0, 0.0, extra_states=(B,))
    assert "sz0_AB" in values


def test_tdvp_simulation_reference_states_used_in_every_measurement():
    N = 4
    sysinf, H = _tfim(N)
    A = _build_state(sysinf, [2 for _ in range(N)], 8)
    B = copy.deepcopy(A)
    h = sop_operator(H, A, sysinf)
    op = site_operator(sOP("sz", 0), sysinf)
    obs = Observable("sz0_AB", op=op)

    sim = TDVPSimulation(A, h, dt=0.05, nstep=2, initial_ramp_steps=0, observables=[obs], reference_states=(B,), integrator_kwargs={"krylov_dim": 12})
    sim.mel = matrix_element(A, B)
    results = sim.run()

    assert results.count == 3
    assert not np.any(np.isnan(results.data["sz0_AB"][: results.count]))


def test_simulation_accepts_prebuilt_integrator():
    N = 4
    sysinf, H = _tfim(N)
    A = _build_state(sysinf, [2 for _ in range(N)], 8)
    h = sop_operator(H, A, sysinf)
    sweep = tdvp(A, h, krylov_dim=12)

    sim = TDVPSimulation(A, h, dt=0.05, nstep=2, initial_ramp_steps=0, integrator=sweep)
    assert sim.integrator is sweep
    sim.run()


def test_observable_accepts_labelled_operator_matches_compiled_site_operator():
    N = 4
    Hl, sysinf, site_map, labels = _labelled_tfim(N)
    Hsop = Hl.compile(site_map, len(site_map))

    topo = ntreeBuilder.mps_tree([2 for _ in range(N)], 8)
    A1 = ttn(topo, dtype=np.complex128)
    A1.random()
    A1.normalise()
    A2 = copy.deepcopy(A1)

    coupling = OperatorBuilder()
    sz_labelled = coupling.wrap(coupling.op("sz", labels[0]))
    sim1 = TDVPSimulation(A1, sop_operator(Hsop, A1, sysinf), dt=0.05, nstep=2, initial_ramp_steps=0, observables=[Observable("sz0", op=sz_labelled)], system_modes=sysinf, site_map=site_map, integrator_kwargs={"krylov_dim": 12})
    results1 = sim1.run()

    op2 = site_operator(sOP("sz", site_map[labels[0]]), sysinf)
    sim2 = TDVPSimulation(A2, sop_operator(Hsop, A2, sysinf), dt=0.05, nstep=2, initial_ramp_steps=0, observables=[Observable("sz0", op=op2)], integrator_kwargs={"krylov_dim": 12})
    results2 = sim2.run()

    assert np.allclose(results1.data["sz0"][: results1.count], results2.data["sz0"][: results2.count])


def test_observable_labelled_operator_without_site_map_raises():
    N = 4
    Hl, sysinf, site_map, labels = _labelled_tfim(N)
    Hsop = Hl.compile(site_map, len(site_map))

    topo = ntreeBuilder.mps_tree([2 for _ in range(N)], 8)
    A = ttn(topo, dtype=np.complex128)
    A.random()
    A.normalise()

    coupling = OperatorBuilder()
    sz_labelled = coupling.wrap(coupling.op("sz", labels[0]))
    with pytest.raises(ValueError):
        TDVPSimulation(A, sop_operator(Hsop, A, sysinf), dt=0.05, nstep=1, initial_ramp_steps=0, observables=[Observable("sz0", op=sz_labelled)])
