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

from pyttn.ttnpp import sepState_complex, ket_complex
try:
    from pyttn.ttnpp import sepState_real, ket_real
    allow_real_type=True
except ImportError:
    allow_real_type = False
from pyttn.ttnpp import stateStr as __stateStr
import numpy as np

def stateStr(*args):
    r"""A function for constructing a separable state for state vector preparation

    :param *args: A variable list for specifying the coefficient.  Valid options are

        - Default construct the stateStr
        - state (list[int]) - Construct separable state from a set of excitation indices

        - state (stateStr) - Construct sepState from another sepState
    """
    return __stateStr(*args)


def sepState(*args, dtype=np.complex128):
    r"""A function for constructing a separable state for state vector preparation

    :param *args: A variable list for specifying the coefficient.  Valid options are

        - Default construct the sepState
        - state (list[int]) - Construct separable state from a set of excitation indices
        - arg (dtype), state (list[int]) -Construct separable state from a coefficient and set of excitation indices
        - state (sepState) - Construct sepState from another sepState
    """
    if (dtype == np.complex128):
        return sepState_complex(*args)
    elif (dtype == np.float64 and allow_real_type):
        return sepState_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sepState")


def isSepState(state):
    if allow_real_type:
        return isinstance(state, (sepState_complex, sepState_real))
    else:
        return isinstance(state, sepState_complex)

def ket(*args, dtype=np.complex128):
    r"""A function for constructing a separable state for state vector preparation

    :param *args: A variable list for specifying the coefficient.  Valid options are

        -  Default construct the ket
        - state (ket) - Construct ket from another ket
    """
    if (dtype == np.complex128):
        return ket_complex(*args)
    elif (dtype == np.float64 and allow_real_type):
        return ket_real(*args)
    else:
        raise RuntimeError("Invalid dtype for ket")

def isKet(state):
    if allow_real_type:
        return isinstance(state, (ket_complex, ket_real))
    else:
        return isinstance(state, ket_complex)