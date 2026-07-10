import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest

from pyttn.simulation import Ensemble


class _FakeResults:
    def __init__(self, index):
        rng = np.random.default_rng(index)
        self._data = {"t": np.arange(5, dtype=float), "value": rng.random(5)}

    def as_dict(self):
        return self._data


class _FakeSimulation:
    def __init__(self, index):
        self.index = index
        self.results = _FakeResults(index)
        self.ran = False

    def run(self):
        self.ran = True


def _sample_fn(index):
    return _FakeSimulation(index)


def _mean_aggregate(results):
    return {key: np.mean([r[key] for r in results], axis=0) for key in results[0]}


def test_ensemble_serial_runs_every_sample():
    ensemble = Ensemble(_sample_fn, n_samples=4, n_workers=1)
    results = ensemble.run()

    assert len(results) == 4
    for i, r in enumerate(results):
        expected = _FakeResults(i).as_dict()
        assert np.allclose(r["value"], expected["value"])


def test_ensemble_samples_are_independent():
    ensemble = Ensemble(_sample_fn, n_samples=3, n_workers=1)
    results = ensemble.run()

    assert not np.allclose(results[0]["value"], results[1]["value"])
    assert not np.allclose(results[1]["value"], results[2]["value"])


def test_ensemble_serial_and_parallel_agree():
    serial = Ensemble(_sample_fn, n_samples=4, n_workers=1).run()
    parallel = Ensemble(_sample_fn, n_samples=4, n_workers=2).run()

    assert len(serial) == len(parallel)
    for r_serial, r_parallel in zip(serial, parallel):
        assert np.allclose(r_serial["value"], r_parallel["value"])


def test_ensemble_aggregate_applied():
    ensemble = Ensemble(_sample_fn, n_samples=5, n_workers=1, aggregate=_mean_aggregate)
    aggregated = ensemble.run()

    expected = _mean_aggregate([_FakeResults(i).as_dict() for i in range(5)])
    assert np.allclose(aggregated["value"], expected["value"])


def test_ensemble_aggregate_applied_with_parallel_workers():
    ensemble = Ensemble(_sample_fn, n_samples=5, n_workers=2, aggregate=_mean_aggregate)
    aggregated = ensemble.run()

    expected = _mean_aggregate([_FakeResults(i).as_dict() for i in range(5)])
    assert np.allclose(aggregated["value"], expected["value"])
