import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest

from pyttn.simulation import ResultsBuffer


def test_results_buffer_record_and_as_dict():
    buf = ResultsBuffer(["sz", "sx"], nrecords=3, dtype=np.float64)
    buf.record(0, 0.0, {"sz": 1.0, "sx": 0.5}, maxchi=2)
    buf.record(1, 0.1, {"sz": 0.9, "sx": 0.4}, maxchi=4)

    data = buf.as_dict()
    assert list(data["t"]) == pytest.approx([0.0, 0.1])
    assert list(data["maxchi"]) == pytest.approx([2, 4])
    assert list(data["sz"]) == pytest.approx([1.0, 0.9])
    assert list(data["sx"]) == pytest.approx([0.5, 0.4])


def test_results_buffer_count_tracks_highest_index():
    buf = ResultsBuffer(["sz"], nrecords=5)
    buf.record(3, 0.3, {"sz": 1.0})
    assert buf.count == 4
    assert len(buf.as_dict()["sz"]) == 4


def test_results_buffer_record_without_maxchi_leaves_default():
    buf = ResultsBuffer(["sz"], nrecords=1)
    buf.record(0, 0.0, {"sz": 1.0})
    assert buf.maxchi[0] == 0.0


def test_results_buffer_unknown_label_raises():
    buf = ResultsBuffer(["sz"], nrecords=1)
    with pytest.raises(KeyError):
        buf.record(0, 0.0, {"unknown": 1.0})


def test_results_buffer_index_out_of_range_raises():
    buf = ResultsBuffer(["sz"], nrecords=1)
    with pytest.raises(IndexError):
        buf.record(1, 0.0, {"sz": 1.0})


def test_results_buffer_to_hdf5_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")

    buf = ResultsBuffer(["sz"], nrecords=2, dtype=np.float64)
    buf.record(0, 0.0, {"sz": 1.0}, maxchi=2)
    buf.record(1, 0.5, {"sz": 0.5}, maxchi=3)

    fname = tmp_path / "results.h5"
    buf.to_hdf5(str(fname), attrs={"note": "test"})

    with h5py.File(str(fname), "r") as h5:
        assert list(h5["t"][:]) == pytest.approx([0.0, 0.5])
        assert list(h5["sz"][:]) == pytest.approx([1.0, 0.5])
        assert list(h5["maxchi"][:]) == pytest.approx([2, 3])
        assert h5.attrs["note"] == "test"


def test_results_buffer_to_hdf5_only_writes_recorded_portion(tmp_path):
    h5py = pytest.importorskip("h5py")

    buf = ResultsBuffer(["sz"], nrecords=5, dtype=np.float64)
    buf.record(0, 0.0, {"sz": 1.0})
    buf.record(1, 0.1, {"sz": 2.0})

    fname = tmp_path / "partial.h5"
    buf.to_hdf5(str(fname))

    with h5py.File(str(fname), "r") as h5:
        assert h5["sz"].shape == (2,)
