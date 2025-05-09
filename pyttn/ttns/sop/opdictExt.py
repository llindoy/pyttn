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

from typing import Union, TypeAlias
import numpy as np

from pyttn.ttnpp import operator_dictionary_complex

try:
    from pyttn.ttnpp import operator_dictionary_real
    __real_opdict = True
    operator_dictionary_type: TypeAlias = operator_dictionary_real | operator_dictionary_complex
except ImportError:
    __real_opdict = False
    operator_dictionary_type: TypeAlias =  operator_dictionary_complex

def operator_dictionary(*args, dtype: Union[float, complex, np.float64, np.complex128]=np.complex128)-> operator_dictionary_type:
    r"""Factory function for constructing a user defined operator dictionary.

    :param *args: Variable length list of arguments.
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional

    :returns: The operator dictionary object
    :rtype: operator_dictionary_real or operator_dictionary_complex
    """

    if __real_opdict:

        if dtype == np.complex128 or dtype is complex:
            return operator_dictionary_complex(*args)
        elif dtype == np.float64 or dtype is float:
            return operator_dictionary_real(*args)
        else:
            raise RuntimeError("Invalid dtype for operator_dictionary")

    else:
        if dtype == np.complex128 or dtype is complex:
            return operator_dictionary_complex(*args)
        else:
            raise RuntimeError("Invalid dtype for operator_dictionary")


def __is_operator_dictionary(a) -> bool:
    if __real_opdict:
        return isinstance(a, operator_dictionary_complex) or isinstance(
            a, operator_dictionary_real
        )

    else:
        return isinstance(a, operator_dictionary_complex)

