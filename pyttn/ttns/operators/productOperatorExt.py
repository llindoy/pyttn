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

import numpy as np
from typing import TypeAlias, Optional, Union

from pyttn.ttns.sop.sSOPExt import sOP, sPOP, sNBO_type
from pyttn.ttnpp import system_modes

from pyttn.ttnpp import product_operator_complex

try:
    from pyttn.ttnpp import product_operator_real

    __real_ttn_import = True
    product_operator_type: TypeAlias = product_operator_complex | product_operator_real

except ImportError:
    __real_ttn_import = False
    product_operator_type: TypeAlias = product_operator_complex


# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import (
        product_operator_complex as product_operator_complex_cuda,
    )

    __cuda_import = True

    # and if we have imported real ttns we import the cuda versions
    if __real_ttn_import:
        from pyttn.ttnpp.cuda import product_operator_real as product_operator_real_cuda

        product_operator_type: TypeAlias = (
            product_operator_type
            | product_operator_complex_cuda
            | product_operator_real_cuda
        )

    else:
        product_operator_type: TypeAlias = (
            product_operator_type | product_operator_complex_cuda
        )

except ImportError:
    __cuda_import = False


def product_operator(
    h: sOP | sPOP | sNBO_type,
    sysinf: system_modes,
    *args,
    dtype: Optional[Union[float, complex, np.float64, np.complex128]] = np.complex128,
    backend: str = "blas",
    **kwargs,
) -> product_operator_type:
    """Function for constructing a product_operator

    :param h: The product operator representation of the Hamiltonian
    :type h: sOP or sPOP or sNBO_type
    :param sysinf: The composition of the system defining the default dictionary to be considered for each node
    :type sysinf: system_modes
    :type *args: Variable length list of arguments. See product_operator_real/product_operator_complex for options
    :param dtype: The internal variable type for the product operator.(Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional
    :param backend: The computational backend to use for the product operator  (Default: "blas")
    :type backend: {"blas", "cuda"}, optional
    :type **kwargs: Additional keyword arguments. See product_operator_real/product_operator_complex for options
    """
    if backend == "blas":
        if dtype == np.complex128 or not __real_ttn_import:
            return product_operator_complex(h, sysinf, *args, **kwargs)
        else:
            return product_operator_real(h, sysinf, *args, **kwargs)
    elif __cuda_import and backend == "cuda":
        if dtype == np.complex128 or not __real_ttn_import:
            return product_operator_complex(h, sysinf, *args, **kwargs)
        else:
            return product_operator_real(h, sysinf, *args, **kwargs)
