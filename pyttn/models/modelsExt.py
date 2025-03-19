# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np


def AIM(*args, dtype=np.complex128):
    from pyttn.ttnpp.models import AIM_complex

    try:
        from pyttn.ttnpp.models import AIM_real

        if dtype == np.complex128:
            return AIM_complex(*args)
        elif dtype == np.float64:
            return AIM_real(*args)
        else:
            raise RuntimeError("Invalid dtype for AIM")
    except ImportError:
        if dtype == np.complex128:
            return AIM_complex(*args)
        else:
            raise RuntimeError("Invalid dtype for AIM")


def electronic_structure(*args, dtype=np.complex128, **kwargs):
    from pyttn.ttnpp.models import electronic_structure_complex

    try:
        from pyttn.ttnpp.models import electronic_structure_real

        if dtype == np.complex128:
            return electronic_structure_complex(*args, **kwargs)
        elif dtype == np.float64:
            return electronic_structure_real(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for electronic_structure model")
    except ImportError:
        if dtype == np.complex128:
            return electronic_structure_complex(*args, **kwargs)
        else:
            raise RuntimeError("Invalid dtype for electronic_structure model")


def TFIM(*args, dtype=np.complex128):
    from pyttn.ttnpp.models import TFIM_complex

    try:
        from pyttn.ttnpp.models import TFIM_real

        if dtype == np.complex128:
            return TFIM_complex(*args)
        elif dtype == np.float64:
            return TFIM_real(*args)
        else:
            raise RuntimeError("Invalid dtype for TFIM")
    except ImportError:
        if dtype == np.complex128:
            return TFIM_complex(*args)
        else:
            raise RuntimeError("Invalid dtype for TFIM")


def __init_sb_type(sbt1, sbt2, *args, dtype=None):
    if args:
        if sbt2 is None:
            return sbt1(*args)
        else:
            if args[-1].dtype == np.complex128 or dtype == np.complex128:
                return sbt1(*args)
            else:
                return sbt2(*args)
    else:
        if sbt2 is None:
            return sbt1()
        else:
            if dtype == np.complex128:
                return sbt1()
            elif dtype == np.float64:
                return sbt2()
            else:
                raise RuntimeError("Invalid dtype for spin boson")


def spin_boson(*args, geom="star", dtype=np.complex128):
    from pyttn.ttnpp.models import (
        spin_boson_generic_complex,
        spin_boson_star_complex,
        spin_boson_chain_complex,
        spin_boson_chain_real,
    )

    try:
        from pyttn.ttnpp.models import (
            spin_boson_generic_real,
            spin_boson_star_real,
            spin_boson_chain_real,
        )
    except ImportError:
        spin_boson_generic_real = None
        spin_boson_star_real = None
        spin_boson_chain_real = None

    if geom == "star":
        return __init_sb_type(
            spin_boson_star_complex, spin_boson_star_real, *args, dtype=dtype
        )
    elif geom == "chain":
        return __init_sb_type(
            spin_boson_chain_complex, spin_boson_chain_real, *args, dtype=dtype
        )
    elif geom == "generic":
        return __init_sb_type(
            spin_boson_generic_complex, spin_boson_generic_real, *args, dtype=dtype
        )
    else:
        raise RuntimeError("Invalid geom for spin boson.")
