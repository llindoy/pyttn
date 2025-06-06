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

from typing import TypeAlias, Optional, Union
import numpy as np

from pyttn.ttnpp import matrix_element_complex, ttn_complex, ms_ttn_complex

try:
    from pyttn.ttnpp import matrix_element_real, ttn_real, ms_ttn_real

    __use_real_matel = True
    matrix_element_type: TypeAlias = matrix_element_real | matrix_element_complex

except ImportError:
    __use_real_matel = False
    matrix_element_type: TypeAlias = matrix_element_complex


def matrix_element(
    *args,
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = np.complex128,
    **kwargs,
) -> matrix_element_type:
    """A factory method for constructing an object used for evaluating matrix elements from a TTN. If this function is passed a TTN object it uses
    this to construct a matrix_element suitable for evaluating matrix elements of this class.  Otherwise this will construct an empty matrix_element
    object with the required dtype.

    :param *args: A variable length set of arguments that is either empty or contains the TTN that we will want to evaluate matrix elements of.
    :type *args: empty or single ttn_type | ms_ttn_type
    :param dtype: The type to be stored in the matrix element object.  This is ignored if the a TTN object is passed in the first argument.
    :type dtype: {np.float64, np.complex128}, optional
    :param **kwargs: Keyword arguments to pass to the matrix element engine constructor.  For details see matrix_element_real or matrix_element_complex

    :returns: The matrix_element object
    :rtype: matrix_element_type
    """

    if __use_real_matel:
        if args:
            if isinstance(args[0], ttn_complex) or isinstance(args[0], ms_ttn_complex):
                return matrix_element_complex(*args, **kwargs)
            elif isinstance(args[0], ttn_real) or isinstance(args[0], ms_ttn_real):
                return matrix_element_real(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for matrix_element")
        else:
            if dtype == np.complex128 or dtype is complex:
                return matrix_element_complex(**kwargs)
            elif dtype == np.float64 or dtype is float:
                return matrix_element_real(**kwargs)
            else:
                raise RuntimeError("Invalid dtype for matrix_element")

    else:
        if args:
            if isinstance(args[0], ttn_complex) or isinstance(args[0], ms_ttn_complex):
                return matrix_element_complex(*args, **kwargs)
            else:
                raise RuntimeError("Invalid dtype for matrix_element")
        else:
            if dtype == np.complex128 or dtype is complex:
                return matrix_element_complex(**kwargs)
            else:
                raise RuntimeError("Invalid dtype for matrix_element")
