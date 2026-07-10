import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest

from pyttn import SOP, matrix_element, ntreeBuilder, sOP, site_operator, sop_operator, system_modes, tls_mode, ttn
from pyttn.simulation import Observable


def _two_site_state():
    N = 2
    sysinf = system_modes(N)
    for i in range(N):
        sysinf[i] = tls_mode()

    topo = ntreeBuilder.mps_tree([2 for _ in range(N)], 4)
    A = ttn(topo, dtype=np.complex128)
    A.random()
    A.normalise()

    B = ttn(topo, dtype=np.complex128)
    B.random()
    B.normalise()

    return sysinf, A, B


def test_observable_norm():
    sysinf, A, _ = _two_site_state()
    mel = matrix_element(A)
    obs = Observable("norm")
    assert obs.evaluate(mel, A) == pytest.approx(np.real(mel(A)))


def test_observable_expectation():
    sysinf, A, _ = _two_site_state()
    mel = matrix_element(A)
    op = site_operator(sOP("sz", 0), sysinf)
    obs = Observable("sz0", op=op)
    assert obs.evaluate(mel, A) == pytest.approx(mel(op, A))


def test_observable_expectation_with_mode():
    sysinf, A, _ = _two_site_state()
    mel = matrix_element(A)
    op = site_operator(sOP("sz", 0), sysinf)
    obs = Observable("sz_mode0", op=op, mode=0)
    assert obs.evaluate(mel, A) == pytest.approx(mel(op, 0, A))


def test_observable_overlap():
    sysinf, A, B = _two_site_state()
    mel = matrix_element(A, B)
    obs = Observable("overlap")
    assert obs.evaluate(mel, A, B) == pytest.approx(mel(A, B))


def test_observable_matrix_element_two_states():
    sysinf, A, B = _two_site_state()
    mel = matrix_element(A, B)
    op = site_operator(sOP("sz", 0), sysinf)
    obs = Observable("sz0_AB", op=op)
    assert obs.evaluate(mel, A, B) == pytest.approx(mel(op, A, B))


def test_observable_requires_at_least_one_state():
    sysinf, A, _ = _two_site_state()
    mel = matrix_element(A)
    obs = Observable("norm")
    with pytest.raises(ValueError):
        obs.evaluate(mel)


def test_observable_repr():
    obs = Observable("label", op=None, mode=2)
    assert "label" in repr(obs)
