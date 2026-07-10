import os

os.environ["OMP_NUM_THREADS"] = "1"

import pytest

from pyttn import tls_mode
from pyttn.oqs.model import OQSModel
from pyttn.ttns.sop import OperatorBuilder, SystemInfo


def test_validate_single_primitive_system():
    sysinfo = SystemInfo()
    sysinfo["spin"] = tls_mode()

    b = OperatorBuilder()
    H = b.wrap(0.5 * b.op("sx", "spin") + b.op("sz", "spin")).to_lCSOP()

    model = OQSModel(system_info=sysinfo, system_generator=H)
    model.validate()


def test_validate_multi_primitive_composite():
    # a Liouville-doubled composite ('spin' with two primitives, 'spin' and 'spin~')
    # exercises the union-over-composites fix to validate().
    sysinfo = SystemInfo()
    sysinfo["spin"] = {"spin": tls_mode(), "spin~": tls_mode()}

    b = OperatorBuilder()
    H = b.wrap(b.op("sz", "spin") - b.op("sz", "spin~")).to_lCSOP()

    model = OQSModel(system_info=sysinfo, system_generator=H)
    model.validate()


def test_validate_missing_system_info():
    b = OperatorBuilder()
    H = b.wrap(b.op("sz", "spin")).to_lCSOP()
    model = OQSModel(system_generator=H)
    with pytest.raises(ValueError):
        model.validate()


def test_validate_missing_generator():
    sysinfo = SystemInfo()
    sysinfo["spin"] = tls_mode()
    model = OQSModel(system_info=sysinfo)
    with pytest.raises(ValueError):
        model.validate()


def test_validate_unknown_site():
    sysinfo = SystemInfo()
    sysinfo["spin"] = tls_mode()

    b = OperatorBuilder()
    H = b.wrap(b.op("sz", "not_in_system")).to_lCSOP()

    model = OQSModel(system_info=sysinfo, system_generator=H)
    with pytest.raises(ValueError):
        model.validate()
