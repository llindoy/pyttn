import os

os.environ["OMP_NUM_THREADS"] = "1"

import copy

import numpy as np
import pytest

from pyttn import SOP, matrix_element, ntreeBuilder, sOP, site_operator, sop_operator, system_modes, tls_mode, ttn
from pyttn.simulation import DMRGSimulation, Pipeline, TDVPSimulation

N = 4
CHI = 8


def _tfim():
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    H = SOP(N)
    for i in range(N):
        H += -1.0 * sOP("sx", i)
    for i in range(N - 1):
        H += -1.0 * sOP("sz", i) * sOP("sz", i + 1)

    return sysinf, H


def _initial_state(seed=0):
    topo = ntreeBuilder.mps_tree([2 for _ in range(N)], CHI)
    A = ttn(topo, dtype=np.complex128)
    A.random()
    A.normalise()
    return A


def _flip_operator(sysinf):
    return site_operator(sOP("sx", 0), sysinf)


def _apply_flip(A, sysinf):
    A.apply_one_body_operator(_flip_operator(sysinf))
    return A


def test_pipeline_ground_state_then_dynamics_matches_manual_chain():
    sysinf, H = _tfim()
    A0 = _initial_state()
    A_pipeline = copy.deepcopy(A0)
    A_manual = copy.deepcopy(A0)

    def ground_state_stage(state):
        assert state is None
        h = sop_operator(H, A_pipeline, sysinf)
        return DMRGSimulation(A_pipeline, h, nsweep=5, integrator_kwargs={"krylov_dim": 12})

    def dynamics_stage(state):
        h = sop_operator(H, state, sysinf)
        return TDVPSimulation(state, h, dt=0.05, nstep=3, initial_ramp_steps=0, integrator_kwargs={"krylov_dim": 12})

    pipeline = Pipeline()
    pipeline.add_stage("ground_state", ground_state_stage)
    pipeline.add_stage("dynamics", dynamics_stage, transform=lambda A: _apply_flip(A, sysinf))
    final_state = pipeline.run()

    assert set(pipeline.simulations.keys()) == {"ground_state", "dynamics"}
    assert pipeline.results("ground_state").count == 5
    assert pipeline.results("dynamics").count == 4

    # manually chain the same two stages, starting from an independent deep copy
    # of the same initial state, and confirm the pipeline produced an identical result
    h_manual = sop_operator(H, A_manual, sysinf)
    ground_sim = DMRGSimulation(A_manual, h_manual, nsweep=5, integrator_kwargs={"krylov_dim": 12})
    ground_sim.run()

    state = ground_sim.state
    _apply_flip(state, sysinf)
    h2_manual = sop_operator(H, state, sysinf)
    dyn_sim = TDVPSimulation(state, h2_manual, dt=0.05, nstep=3, initial_ramp_steps=0, integrator_kwargs={"krylov_dim": 12})
    dyn_sim.run()

    mel = matrix_element(final_state, dyn_sim.state)
    overlap = mel(final_state, dyn_sim.state)
    assert abs(overlap) == pytest.approx(1.0, abs=1e-8)


def test_pipeline_stage_receives_transformed_state():
    sysinf, H = _tfim()
    seen = {}

    def stage_one(state):
        A = _initial_state()
        h = sop_operator(H, A, sysinf)
        return TDVPSimulation(A, h, dt=0.05, nstep=1, initial_ramp_steps=0, integrator_kwargs={"krylov_dim": 12})

    def transform(state):
        seen["transformed"] = True
        return _apply_flip(state, sysinf)

    def stage_two(state):
        seen["stage_two_state"] = state
        h = sop_operator(H, state, sysinf)
        return TDVPSimulation(state, h, dt=0.05, nstep=1, initial_ramp_steps=0, integrator_kwargs={"krylov_dim": 12})

    pipeline = Pipeline()
    pipeline.add_stage("one", stage_one)
    pipeline.add_stage("two", stage_two, transform=transform)
    pipeline.run()

    assert seen.get("transformed") is True
    assert seen["stage_two_state"] is pipeline.simulations["two"].state


def test_pipeline_first_stage_receives_none():
    sysinf, H = _tfim()
    received = []

    def stage(state):
        received.append(state)
        A = _initial_state()
        h = sop_operator(H, A, sysinf)
        return TDVPSimulation(A, h, dt=0.05, nstep=1, initial_ramp_steps=0, integrator_kwargs={"krylov_dim": 12})

    pipeline = Pipeline()
    pipeline.add_stage("only", stage)
    pipeline.run()

    assert received == [None]
