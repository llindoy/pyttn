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
from .sSOPExt import __is_sSOP, __is_sNBO, __is_sPOP, __is_sOP
import numpy as np


from pyttn.ttnpp import SOP_complex
from pyttn.ttnpp import multiset_SOP_complex

try:
    from pyttn.ttnpp import SOP_real
    from pyttn.ttnpp import multiset_SOP_real

    __support_real_SOP = True
    SOP_type: TypeAlias = SOP_real | SOP_complex
    ms_SOP_type: TypeAlias = multiset_SOP_real | multiset_SOP_complex
except ImportError:
    __support_real_SOP = False
    SOP_type: TypeAlias = SOP_complex
    ms_SOP_type: TypeAlias = multiset_SOP_complex


def SOP(
    *args, dtype: Union[float, complex, np.float64, np.complex128] = np.complex128
) -> SOP_type:
    r"""Factory function for constructing a sum-of-product compact string operator.

    :param *args: Variable length list of arguments. This function can handle two possible lists of arguments

        - N (int) - The number of modes of the SOP
        - N (int), label (str) - The number of modes of the SOP and a label for the SOP
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The SOP object
    :rtype: SOP_real or SOP_complex
    """

    if __support_real_SOP:
        from pyttn.ttnpp import SOP_real

        if dtype == np.complex128 or dtype is complex:
            return SOP_complex(*args)
        elif dtype == np.float64 or dtype is float:
            return SOP_real(*args)
        else:
            raise RuntimeError("Invalid dtype for SOP")
    else:
        if dtype == np.complex128 or dtype is complex:
            return SOP_complex(*args)
        else:
            raise RuntimeError("Invalid dtype for SOP")


def ms_SOP(
    *args, dtype: Union[float, complex, np.float64, np.complex128] = np.complex128
) -> ms_SOP_type:
    """Factory function for constructing a multiset sum-of-product compact string operator.

    :param *args: Variable length list of arguments. This function can handle two possible lists of arguments

        - nset( int), N (int) - The number of set variables, The number of modes of the SOP
        - nset( int), N (int), label (str) - The number of set varibles. The number of modes of the SOP and a label for the SOP
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The multiset_SOP object
    :rtype: multiset_SOP_real or multiset_SOP_complex
    """
    return multiset_SOP(*args, dtype=dtype)


def multiset_SOP(
    *args, dtype: Union[float, complex, np.float64, np.complex128] = np.complex128
) -> ms_SOP_type:
    """Factory function for constructing a multiset sum-of-product compact string operator.

    :param *args: Variable length list of arguments. This function can handle two possible lists of arguments

        - nset( int), N (int) - The number of set variables, The number of modes of the SOP
        - nset( int), N (int), label (str) - The number of set varibles. The number of modes of the SOP and a label for the SOP
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The multiset_SOP object
    :rtype: multiset_SOP_real or multiset_SOP_complex
    """

    if __support_real_SOP:
        if dtype == np.complex128 or dtype is complex:
            return multiset_SOP_complex(*args)
        elif dtype == np.float64 or dtype is float:
            return multiset_SOP_real(*args)
        else:
            raise RuntimeError("Invalid dtype for multiset_SOP")
    else:
        if dtype == np.complex128 or dtype is complex:
            return multiset_SOP_complex(*args)
        else:
            raise RuntimeError("Invalid dtype for multiset_SOP")


def sum_of_product(
    nset: int,
    *args,
    dtype: Union[float, complex, np.float64, np.complex128] = np.complex128,
) -> SOP_type | ms_SOP_type:
    r"""Factory function for constructing a generic sum-of-product compact string operator.

    :param nset: The number of set variables to use
    :type nset: int
    :param *args: Variable length list of arguments. This function can handle two possible lists of arguments

        - N (int) - The number of modes of the SOP
        - N (int), label (str) - The number of modes of the SOP and a label for the SOP
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The SOP object
    :rtype: SOP_real or SOP_complex or multiset_SOP_real or multiset_SOP_complex

    """
    if nset == 1:
        return SOP(*args, dtype=dtype)
    elif nset > 1:
        return multiset_SOP(nset, *args, dtype=dtype)
    else:
        raise RuntimeError("Failed to construct sum of product operator object.")


def __is_SOP_real(a) -> bool:
    try:
        from pyttn.ttnpp import SOP_real, multiset_SOP_real

        return isinstance(a, SOP_real) or isinstance(a, multiset_SOP_real)
    except ImportError:
        return False


def __is_SOP_complex(a) -> bool:
    from pyttn.ttnpp import SOP_complex, multiset_SOP_complex

    return isinstance(a, SOP_complex) or isinstance(a, multiset_SOP_complex)


def __is_SOP(a) -> bool:
    return __is_SOP_real(a) or __is_SOP_complex(a)


def __validate_SOP(Op, sys):
    if __is_SOP(Op):
        if Op.nmodes() != sys.nprimitive_nmodes():
            raise RuntimeError("Operator size is not compatible with system size.")
        return Op
    elif __is_sSOP(Op):
        op = SOP(sys.nprimitive_modes())
        for term in Op:
            op += term
        return op
    elif __is_sNBO(Op) or __is_sPOP(Op) or __is_sOP(Op):
        op = SOP(sys.nprimitive_modes())
        op += Op
        return op
    else:
        raise RuntimeError("Invalid operator type.")
