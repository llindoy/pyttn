# This files is part of the pyTTN package.
# (C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Union, TypeAlias
from pyttn.ttnpp import (
    sOP,
    sPOP,
    sNBO_real,
    sNBO_complex,
    sSOP_real,
    sSOP_complex,
    coeff_real,
    coeff_complex,
)
import numpy as np

sOP_type: TypeAlias = sOP
sPOP_type: TypeAlias = sPOP
sNBO_type: TypeAlias = sNBO_real | sNBO_complex
sSOP_type: TypeAlias = sSOP_real | sSOP_complex
coeff_type: TypeAlias = coeff_real | coeff_complex


def coeff(
    *args, dtype: Union[float, complex, np.float64, np.complex128] = np.complex128
) -> coeff_type:
    r"""A function for constructing the coeff type for Hamiltonian specification

    :param coeff: A variable list for specifying the coefficient.  Valid options are

        - Default construct the coefficient
        - value (dtype) - Set the coefficient to a constant value
        - func (callable) - set the coefficient to a time-dependent value
    :param dtype: The internal variable type for the product operator.
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The coefficient object
    :rtype: coeff_real or coeff_complex
    """
    if dtype == np.complex128 or dtype is complex:
        return coeff_complex(*args)
    elif dtype == np.float64 or dtype is float:
        return coeff_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sNBO")


def sNBO(
    *args, dtype: Union[float, complex, np.float64, np.complex128] = np.complex128
) -> sNBO_type:
    r"""A function for constructing an n-body operator string

    :param *args: A variable list for specifying the coefficient.  Valid options are

        -  Default construct the sNBO
        - op (sOP) - Construct NBO from single site operator
        - pop (sPOP) - Construct NBO from product operator
        - arg (dtype), op (sOP) - Construct NBO as a product of a constant and a single site operator
        - arg (dtype), pop (sPOP) - Construct NBO as a product of a constant and a product operator
        - arg (coeff), op (sOP) - Construct NBO as a product of a coefficient and a single site operator
        - arg (coeff), pop (sPOP) - Construct NBO as a product of a coefficient and a product operator
        - nbo (sNBO) - Construct NBO from another NBO

    :param dtype: The internal variable type for the n-body operator string.
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The n-body operator object
    :rtype: sNBO_real or sNBO_complex
    """
    if dtype == np.complex128 or dtype is complex:
        return sNBO_complex(*args)
    elif dtype == np.float64 or dtype is float:
        return sNBO_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sNBO")


def sSOP(
    *args, dtype: Union[float, complex, np.float64, np.complex128] = np.complex128
) -> sSOP_type:
    r"""A function for constructing a sum-of-product string operator

    :param *args: A variable list for specifying the coefficient.  Valid options are

        -  Default construct the sSOP
        - op (str) - Construct the sSOP from a string defining a sOP
        - op (sOP) - Construct sSOP from single site operator
        - pop (sPOP) - Construct sSOP from product operator
        - nbo (sNBO) - Construct sSOP from an sNBO
        - sop (sSOP) - Construct sSOP from another sSOP

    :param dtype: The internal variable type for the sum-of-product string operator.
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The sum-of-product string operator
    :rtype: sSOP_real or sSOP_complex
    """
    if dtype == np.complex128 or dtype is complex:
        return sSOP_complex(*args)
    elif dtype == np.float64 or dtype is float:
        return sSOP_real(*args)
    else:
        raise RuntimeError("Invalid dtype for sSOP")


def __is_sOP(a):
    return isinstance(a, sOP)


def __is_sPOP(a):
    return isinstance(a, sPOP)


def __is_sNBO(a):
    return isinstance(a, sNBO_complex) or isinstance(a, sNBO_real)


def __is_sSOP(a):
    return isinstance(a, sSOP_complex) or isinstance(a, sSOP_real)
