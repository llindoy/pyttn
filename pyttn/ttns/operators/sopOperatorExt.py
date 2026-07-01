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

from abc import ABCMeta, abstractmethod
from typing import Union

from pyttn.ttnpp import SOP_complex, sop_operator_complex, system_modes, ttn_complex
from pyttn.ttns.sop.SOPExt import SOP
from pyttn.ttns.ttns.ttnExt import ttn

try:
    from pyttn.ttnpp import SOP_real, sop_operator_real, ttn_real

    _real_ttn_import = True

except ImportError:
    _real_ttn_import = False

# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import sop_operator_complex as sop_operator_complex_cuda
    from pyttn.ttnpp.cuda import ttn_complex as ttn_complex_cuda

    _cuda_import = True

    # and if we have imported real ttns we import the cuda versions
    if _real_ttn_import:
        from pyttn.ttnpp.cuda import sop_operator_real as sop_operator_real_cuda
        from pyttn.ttnpp.cuda import ttn_real as ttn_real_cuda

except ImportError:
    _cuda_import = False


def _sop_operator_blas(h, A, sysinf, *args, compress: bool = True, identity_opt: bool = True, use_sparse: bool = True):
    if not _real_ttn_import:
        if isinstance(A, ttn_complex) and isinstance(h, SOP_complex):
            return sop_operator_complex(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a SOPOperator.")
    else:
        if isinstance(A, ttn_real) and isinstance(h, SOP_real):
            return sop_operator_real(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        elif isinstance(A, ttn_complex) and isinstance(h, SOP_complex):
            return sop_operator_complex(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a SOPOperator.")


def _sop_operator_cuda(h, A, sysinf, *args, compress: bool = True, identity_opt: bool = True, use_sparse: bool = True):
    if not _real_ttn_import:
        if isinstance(A, ttn_complex_cuda) and isinstance(h, SOP_complex):
            return sop_operator_complex_cuda(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a SOPOperator.")
    else:
        if isinstance(A, ttn_real_cuda) and isinstance(h, SOP_real):
            return sop_operator_real_cuda(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        elif isinstance(A, ttn_complex_cuda) and isinstance(h, SOP_complex):
            return sop_operator_complex_cuda(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a SOPOperator.")


class SOPOperator(metaclass=ABCMeta):
    """A class for handling the sum-of-product operators."""

    def __new__( 
        cls, h: SOP, A: ttn, sysinf: system_modes, *args, compress: bool = True, identity_opt: bool = True, use_sparse: bool = True,
    ) -> "SOPOperator":
        """Function for constructing the hierarchical sum of product operator of a string operator

        :param h: The sum of product operator representation of the Hamiltonian
        :type h: SOP
        :param A: A TTN object with defining the topology of output hierarchical SOP object
        :type A: ttn
        :param sysinf: The composition of the system defining the default dictionary to be considered for each node
        :type sysinf: system_modes
        :type `*args`: Variable length list of arguments. Valid options are:

            - Empty: Build the sum-of-product operator using the default operator dictionaries
            - opdict (:class:`operator_dictionary`): Build the sum-of-product operator using a user defined operator dictionary

        :param compress: Whether or not to use the compressed hierarchical SOP representation.  If False this uses the standard sum-of-product representation., defaults to True
        :type compress: bool, optional
        :param identity_opt: Whether or not to perform optimisations arising from the presence of identity operators, defaults to True
        :type identity_opt: bool, optional
        :param use_sparse: Whether or not to use sparse matrix representations of operators, defaults to True
        :type use_sparse: bool, optional
        :return: The SOPOperator representation of the input SOP
        :rtype: SOPOperator
        """

        if len(args) > 0:
            if args[0].backend() != A.backend():
                raise RuntimeError(
                    "Attempted to construct SOPOperator with opdict but opdict backend is not compatible with ttn backend."
                )
        if A.backend() == "blas":
            return _sop_operator_blas(
                h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,
            )
        elif _cuda_import and A.backend() == "cuda":
            return _sop_operator_cuda(
                h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,
            )
        else:
            raise RuntimeError("Invalid backend type for SOPOperator")
        
    @abstractmethod
    def initialise( 
        self, op: SOP, A: ttn, sysinf: system_modes, *args, compress: bool = True, identity_opt: bool = True, use_sparse: bool = True,
    ):
        """Initialise the SOPOperator object given a sOP and system_modes information

        :param h: The sum of product operator representation of the Hamiltonian
        :type h: SOP
        :param A: A TTN object with defining the topology of output hierarchical SOP object
        :type A: ttn
        :param sysinf: The composition of the system defining the default dictionary to be considered for each node
        :type sysinf: system_modes
        :type `*args`: Variable length list of arguments. Valid options are

            - Empty: Build the sum-of-product operator using the default operator dictionaries
            - opdict (:class:`operator_dictionary`): Build the sum-of-product operator using a user defined operator dictionar

        :param compress: Whether or not to use the compressed hierarchical SOP representation.  If False this uses the standard sum-of-product representation., defaults to True
        :type compress: bool, optional
        :param identity_opt: Whether or not to perform optimisations arising from the presence of identity operators, defaults to True
        :type identity_opt: bool, optional
        :param use_sparse: Whether or not to use sparse matrix representations of operators, defaults to True
        :type use_sparse: bool, optional
        :return: The SOPOperator representation of the input SOP
        :rtype: SOPOperator
        """
        pass

    @abstractmethod
    def assign(self, o: "SOPOperator"):
        """Assign the value of the sum-of-product operator from another

        :param o: The sum-of-product operator to copy into this one
        :type o: SOPOperator
        """
        pass

    @abstractmethod
    def __copy__(self):
        """Function implementing shallow copy of the sum-of-product operator object"""
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the sum-of-product operator object"""
        pass
    
    @property
    @abstractmethod
    def Eshift(self) -> Union[float, complex]:
        """A constant energy shift acting on the sum-of-product operator"""
        pass

    @abstractmethod
    def set_Eshift(self, v : Union[float, complex]):
        """A constant energy shift acting on the sum-of-product operator
        
        :param v: The new value of the Eshift object
        :type v: Union[float, complex]
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear and deallocate all internal buffers of the SOPOperator"""
        pass

    @abstractmethod
    def update(self, mode: int, t: float, dt: float):
        """Update the operators associated with mode mode so that they store their value at time t.
        Additionally, this takes the time-step allowing for the use of average timestep expressions

        :param mode: The mode to update
        :type mode: int
        :param t: The new time point
        :type t: float
        :param dt: The integration timestep
        :type dt: float
        """
        pass

    @abstractmethod
    def nterms(self) -> int:
        """
        :returns: The number terms in the sum-of-product operator
        :rtype: int
        """
        pass

    @abstractmethod
    def nmodes(self) -> int:
        """
        :returns: The number of modes the sum-of-product operator acts on
        :rtype: int
        """
        pass

    @abstractmethod
    def complex_dtype(self) -> bool:
        """Returns whether or not the SOPOperator is storing a complex valued dtype

        :return: whether or not the SOPOperator is storing a complex valued dtype
        :rtype: bool
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Returns the NumPy dtype of the underlying operator representation.

        This corresponds to the scalar type used internally, e.g.
        ``np.float64`` or ``np.complex128``.

        :return: The dtype of the operator
        :rtype: numpy.dtype
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns the backend type of the SOPOperator

        :return: The backend type of the object
        :rtype: str
        """
        pass

    @abstractmethod
    def bond_dimensions(self) -> dict[tuple[int, int], int]:
        """Return a dictionary containing the bond (the two sites forming the bond) and bond dimension of all bonds in the network

        :return: All bond dimensions in the network
        :rtype: dict[tuple[int, int], int]
        """
        pass
    
SOPOperator.register(sop_operator_complex)
if _real_ttn_import:
    SOPOperator.register(sop_operator_real)

if _cuda_import:
    SOPOperator.register(sop_operator_complex_cuda)
    if _real_ttn_import:
        SOPOperator.register(sop_operator_real_cuda)

sop_operator=SOPOperator