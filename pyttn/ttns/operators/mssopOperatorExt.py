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
from typing import Union

from pyttn.ttnpp import (
    ms_ttn_complex,
    multiset_SOP_complex,
    multiset_sop_operator_complex,
    system_modes,
)
from pyttn.ttns.sop.SOPExt import multiset_SOP
from pyttn.ttns.ttns.msttnExt import multiset_ttn

try:
    from pyttn.ttnpp import ms_ttn_real, multiset_sop_operator_real, multiset_SOP_real

    _real_ttn_import = True

except ImportError:
    _real_ttn_import = False


# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import ms_ttn_complex as ms_ttn_complex_cuda
    from pyttn.ttnpp.cuda import (
        multiset_sop_operator_complex as multiset_sop_operator_complex_cuda,
    )

    _cuda_import = True

    # and if we have imported real ttns we import the cuda versions
    if _real_ttn_import:
        from pyttn.ttnpp.cuda import ms_ttn_real as ms_ttn_real_cuda
        from pyttn.ttnpp.cuda import (
            multiset_sop_operator_real as multiset_sop_operator_real_cuda,
        )


except ImportError:
    _cuda_import = False


def _multiset_sop_operator_blas(h, A, sysinf, *args, compress: bool = True, identity_opt: bool = True, use_sparse: bool = True,):
    if not _real_ttn_import:
        if isinstance(A, ms_ttn_complex) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a multiset_sop_operator."
            )
    else:
        if (
            isinstance(A, ms_ttn_real)
            and isinstance(h, multiset_SOP_real)
            and multiset_SOP_real is not None
        ):
            return multiset_sop_operator_real(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        elif isinstance(A, ms_ttn_complex) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a multiset_sop_operator."
            )


def _multiset_sop_operator_cuda(h, A, sysinf, *args, compress: bool = True, identity_opt: bool = True, use_sparse: bool = True,):
    if not _real_ttn_import:
        if isinstance(A, ms_ttn_complex_cuda) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex_cuda(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a multiset_sop_operator."
            )
    else:
        if (
            isinstance(A, ms_ttn_real_cuda)
            and isinstance(h, multiset_SOP_real)
            and multiset_SOP_real is not None
        ):
            return multiset_sop_operator_real_cuda(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        elif isinstance(A, ms_ttn_complex_cuda) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex_cuda(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a multiset_sop_operator."
            )


class multiset_sop_operator(metaclass=ABCMeta):
    def __new__(cls, 
        h: multiset_SOP, A: multiset_ttn, sysinf: system_modes, *args, compress: bool = True, identity_opt: bool = True, use_sparse: bool = True,
    ) -> "multiset_sop_operator":
        """Function for constructing the multiset hierarchical sum of product operator of a string operator

        :param h: The multiset sum of product operator representation of the Hamiltonian
        :type h: multiset_SOP
        :param A: A TTN object with defining the topology of output hierarchical SOP object
        :type A: multiset_ttn
        :param sysinf: The composition of the system defining the default dictionary to be considered for each node
        :type sysinf: system_modes
        :type `*args`: Variable length list of arguments. Valid options are

            - Empty: Build the multiset sum-of-product operator using the default operator dictionaries
            - opdict (:class:`operator_dictionary`): Build the multiset sum-of-product operator using a user defined operator dictionary

        :param compress: Whether or not to use the compressed hierarchical SOP representation.  If False this uses the standard multiset sum-of-product representation., defaults to True
        :type compress: bool, optional
        :param identity_opt: Whether or not to perform optimisations arising from the presence of identity operators, defaults to True
        :type identity_opt: bool, optional
        :param use_sparse: Whether or not to use sparse matrix representations of operators, defaults to True
        :type use_sparse: bool, optional
        :returns: The multiset sop operator
        :rtype: multiset_sop_operator
        """
        if len(args) > 0:
            if args[0].backend() != A.backend():
                raise RuntimeError(
                    "Attempted to construct multiset_sop_operator with opdict but opdict backend is not compatible with ms_ttn backend."
                )

        if A.backend() == "blas":
            return _multiset_sop_operator_blas(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        elif _cuda_import and A.backend() == "cuda":
            return _multiset_sop_operator_cuda(h, A, sysinf, *args, compress=compress, identity_opt=identity_opt, use_sparse=use_sparse,)
        else:
            raise RuntimeError("Invalid backend type for multiset_sop_operator")
        
    @abstractmethod
    def initialise( 
        self, op: multiset_SOP, sysinf: system_modes, *args, compress: bool = True, identity_opt: bool = True, use_sparse: bool = True,
    ) -> None:
        """Initialise the multiset_sop_operator object given a sOP and system_modes information

        :param h: The multiset  sum of product operator representation of the Hamiltonian
        :type h: multiset_SOP
        :param A: A TTN object with defining the topology of output hierarchical SOP object
        :type A: multiset_ttn
        :param sysinf: The composition of the system defining the default dictionary to be considered for each node
        :type sysinf: system_modes
        :type `*args`: Variable length list of arguments. Valid options are

            - Empty: Build the multiset sum-of-product operator using the default operator dictionaries
            - opdict (:class:`operator_dictionary`): Build the multiset sum-of-product operator using a user defined operator dictionary

        :param compress: Whether or not to use the compressed hierarchical SOP representation.  If False this uses the standard multiset sum-of-product representation., defaults to True
        :type compress: bool, optional
        :param identity_opt: Whether or not to perform optimisations arising from the presence of identity operators, defaults to True
        :type identity_opt: bool, optional
        :param use_sparse: Whether or not to use sparse matrix representations of operators, defaults to True
        :type use_sparse: bool, optional
        :return: The multiset_sop_operator representation of the input SOP
        :rtype: sop_operator
        """
        pass

    @abstractmethod
    def assign(self, o: "multiset_sop_operator"):
        """Assign the value of the multiset sum-of product product operator from another

        :param o: The multiset sum-of-product operator to copy into this one
        :type o: multiset_sop_operator
        """
        pass

    @abstractmethod
    def __copy__(self):
        """Function implementing shallow copy of the multiset sum-of-product operator object"""
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the multiset sum-of-product operator object"""
        pass

    @abstractmethod
    def Eshift(self, i: int, j: int) -> Union[float, complex]:
        """Return the value of the energy shift associated with the [i, j]th term of the multiset_sop_operator

        :param i: Index i
        :type i: int
        :param j: Index j
        :type j: int
        :return: The constant energy shift associated with this term
        :rtype: Union[float, complex]
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear and deallocate all internal buffers of the multiset_sop_operator"""
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
    def nset(self) -> int:
        """
        :returns: The number of set multiset sum-of-product operator
        :rtype: int
        """
        pass

    @abstractmethod
    def nmodes(self) -> int:
        """
        :returns: The number of modes the multiset sum-of-product operator acts on
        :rtype: int
        """
        pass

    @abstractmethod
    def complex_dtype(self) -> bool:
        """Returns whether or not the multiset_sop_operator is storing a complex valued dtype

        :return: whether or not the multiset_sop_operator is storing a complex valued dtype
        :rtype: bool
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns the backend type of the multiset_sop_operator

        :return: The backend type of the object
        :rtype: str
        """
        pass

multiset_sop_operator.register(multiset_sop_operator_complex)
if _real_ttn_import:
    multiset_sop_operator.register(multiset_sop_operator_real)

if _cuda_import:
    multiset_sop_operator.register(multiset_sop_operator_complex_cuda)
    if _real_ttn_import:
        multiset_sop_operator.register(multiset_sop_operator_real_cuda)

ms_sop_operator = multiset_sop_operator
