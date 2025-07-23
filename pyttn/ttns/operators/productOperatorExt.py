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

from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np

from pyttn.ttnpp import product_operator_complex, system_modes
from pyttn.ttns.sop.sSOPExt import sNBO, sOP, sPOP

try:
    from pyttn.ttnpp import product_operator_real

    _real_ttn_import = True

except ImportError:
    _real_ttn_import = False

# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import (
        product_operator_complex as product_operator_complex_cuda,
    )

    _cuda_import = True
    # and if we have imported real ttns we import the cuda versions
    if _real_ttn_import:
        from pyttn.ttnpp.cuda import product_operator_real as product_operator_real_cuda

except ImportError:
    _cuda_import = False


class product_operator(metaclass=ABCMeta):
    """A class for handling product operators."""

    def __new__(
        cls,
        h: Union[sOP, sPOP, sNBO],
        sysinf: system_modes,
        *args,
        dtype: Optional[
            Union[float, complex, np.float64, np.complex128]
        ] = np.complex128,
        backend: str = "blas",
        use_sparse: bool = True,
    ) -> "product_operator":
        """Function for constructing a product_operator

        :param h: The product operator representation of the Hamiltonian
        :type h: Union[sOP, sPOP, sNBO]
        :param sysinf: The composition of the system defining the default dictionary to be considered for each node
        :type sysinf: system_modes
        :type *args: Variable length list of arguments. Valid options are:

            - Empty: Build the product operator using the default operator dictionaries
            - opdict (:class:`operator_dictionary`): Build the product operator using a user defined operator dictionary

        :param dtype: The internal variable type for the product operator.(Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional
        :param backend: The computational backend to use for the product operator  (Default: "blas")
        :type backend: {"blas", "cuda"}, optional
        :param use_sparse: Whether or not to use sparse matrix representations of operators, defaults to True
        :type use_sparse: bool, optional  
        """
        if backend == "blas":
            if dtype == np.complex128 or not _real_ttn_import:
                return product_operator_complex(h, sysinf, *args, use_sparse=use_sparse)
            else:
                return product_operator_real(h, sysinf, *args, use_sparse=use_sparse)
        elif _cuda_import and backend == "cuda":
            if dtype == np.complex128 or not _real_ttn_import:
                return product_operator_complex(h, sysinf, *args, use_sparse=use_sparse)
            else:
                return product_operator_real(h, sysinf, *args, use_sparse=use_sparse)

    @abstractmethod
    def initialise(self, op: Union[sOP, sPOP, sNBO], sysinf: system_modes, *args, use_sparse: bool = True):
        """Initialise the product_operator object given a sOP and system_modes information

        :param op: The product operator representation of the Hamiltonian
        :type op: Union[sOP, sPOP, sNBO]
        :param sysinf: The information about the system degrees of freedom
        :type sysinf: system_modes
        :type *args: Variable length list of arguments. Valid options are:

            - Empty: Build the product operator using the default operator dictionaries
            - opdict (:class:`operator_dictionary`): Build the product operator using a user defined operator dictionary

        :param use_sparse: Whether or not to use sparse matrix representations of operators, defaults to True
        :type use_sparse: bool, optional
        """
        pass

    @abstractmethod
    def assign(self, o: "product_operator"):
        """Assign the value of the product operator from another 

        :param o: The product operator to copy into this one
        :type o: product_operator
        """
        pass

    @abstractmethod
    def complex_dtype(self) -> bool:
        """Returns whether or not the product_operator is storing a complex valued dtype

        :return: whether or not the product_operator is storing a complex valued dtype
        :rtype: bool
        """
        pass

    @abstractmethod
    def __copy__(self):
        """Function implementing shallow copy of the product_operator object"""
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the product_operator object"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the product_operator object

        :return: The string representation of the product_operator
        :rtype: str
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns the backend type of the product_operator

        :return: The backend type of the object
        :rtype: str
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear and deallocate all internal buffers of the ttn"""
        pass

    @abstractmethod
    def nmodes(self) -> int:
        """
        :returns: The number of modes the product operator acts on
        :rtype: int
        """
        pass

product_operator.register(product_operator_complex)
if _real_ttn_import:
    product_operator.register(product_operator_real)

if _cuda_import:
    product_operator.register(product_operator_complex_cuda)
    if _real_ttn_import:
        product_operator.register(product_operator_real_cuda)

product_operator_type = product_operator
