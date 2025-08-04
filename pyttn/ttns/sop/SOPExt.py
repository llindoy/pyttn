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

import numpy as np

from pyttn.ttnpp import SOP_complex, multiset_SOP_complex, system_modes

from .opdictExt import operator_dictionary
from .sSOPExt import OPBase, sSOP

try:
    from pyttn.ttnpp import SOP_real, multiset_SOP_real

    _support_real_SOP = True
except ImportError:
    _support_real_SOP = False


class SOP(metaclass=ABCMeta):
    """A class for storing a compact sum-of-product operators object.  This class is less flexible than the SOP object but provides considerably higher performance."""

    def __new__(
        cls,
        *args,
        dtype: Union[float, complex, np.float64, np.complex128] = np.complex128,
    ) -> "SOP":
        """Factory function for constructing a sum-of-product compact string operator.

        :param `*args`: Variable length list of arguments. This function can handle two possible lists of arguments

            - N (int) - The number of modes of the SOP
            - N (int), label (str) - The number of modes of the SOP and a label for the SOP
        :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional

        :returns: The SOP object
        :rtype: SOP
        """
        if _support_real_SOP:
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

    @abstractmethod
    def assign(self, o: "SOP"):
        """Assign the value of the SOP from another

        :param o: The SOP to copy into this one
        :type o: SOP
        """
        pass

    @abstractmethod
    def __copy__(self):
        """Function implementing shallow copy of the SOP object"""
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the SOP object"""
        pass

    @abstractmethod
    def __iter__(self):
        """Return an iterator over the terms in the SOP"""

    @abstractmethod
    def clear(self):
        """Clear all internal storage used by the SOP object"""
        pass

    @abstractmethod
    def resize(self, n: int):
        """Resizes the SOP object so that it can act on n modes

        :param n: The number of modes
        :type n: int
        """

    @abstractmethod
    def reserve(self, n: int):
        """Allocate the internal buffer for the SOP object so that it is capable of storing n terms

        :param n: The number of elements that should be reserved.
        :type n: int
        """

    @abstractmethod
    def nmodes(self) -> int:
        """Returns the number of modes that the SOP acts on

        :return: The number of modes that the SOP acts on
        :rtype: int
        """

    @abstractmethod
    def nterms(self) -> int:
        """Returns the number of terms in the SOP

        :return: The number of terms in the SOP
        :rtype: int
        """
        pass

    @property
    @abstractmethod
    def operator_dictionary(self) -> operator_dictionary:
        """The operator dictionary object used by the SOP object"""
        pass

    @abstractmethod
    def set_operator_dictionary(self, opdict: operator_dictionary):
        """Set the value of the operator dictionary object to be used by the SOP

        :param opdict: The new operator dictionary to use
        :type opdict: operator_dictionary
        """
        pass

    @abstractmethod
    def get_operator_dictionary(self) -> operator_dictionary:
        """Access the operator dictionary object used by the SOP

        :return: The operator dictionary
        :rtype: operator_dictionary
        """
        pass

    @abstractmethod
    def insert(self, *args):
        """Insert a new n-body operator into the sum-of-product representation

        :param `*args`: An input defining the n-boddy operator. Valid  options are:

            - coeff (Union[float, complex, np.float64, np.complex128]), pop (sPOP): Add a new sNBO defined by coeff*pop
            - term (sNBO): Add the specified n-body operator
        """
        pass

    @abstractmethod
    def set_is_fermion_mode(self, is_fermion_mode: list[bool]) -> bool:
        """Set whether or not the modes in the SOP object are fermionic

        :param is_fermion_mode: A list specifying whether each mode is fermionic
        :type is_fermion_mode: list[bool]
        :return: Whether or not this set operation was successful
        :rtype: bool
        """
        pass

    @abstractmethod
    def prune_zeros(self, tol: float = 1e-15):
        """Remove any terms in the Hamiltonian definition that are zero within the user specified tolerance

        :param tol: A pruning tolerance for removing zero terms, defaults to 1e-15
        :type tol: float, optional
        """
        pass

    @abstractmethod
    def jordan_wigner(self, sysinf: system_modes, tol: float = 1e-15) -> "SOP":
        """Perform an inplace Jordan-Wigner mapping of the SOP object

        :param sysinf: Definition of the modes forming the system
        :type sysinf: system_modes
        :param tol: A pruning tolerance for removing zero terms, defaults to 1e-15
        :type tol: float, optional
        :return: The jordan-wigner mapped SOP object
        :rtype: SOP
        """
        pass

    @abstractmethod
    def expand(self) -> sSOP:
        """Expand the SOP object to form an sSOP object

        :return: The sSOP representation of this SOP
        :rtype: sSOP
        """

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the SOP object

        :return: The string representation of the SOP
        :rtype: str
        """
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        """A string labelling the operator"""
        pass

    @abstractmethod
    def __add__(
        self, op: Union[float, complex, np.float64, np.complex128, OPBase]
    ) -> "SOP":
        """Add a symbolic operator type to the current SOP to obtain a SOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: SOP
        """
        pass

    @abstractmethod
    def __radd__(
        self, op: Union[float, complex, np.float64, np.complex128, OPBase]
    ) -> "SOP":
        """Add the SOP object from an operator or number

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: SOP
        """
        pass

    @abstractmethod
    def __iadd__(
        self, op: Union[float, complex, np.float64, np.complex128, OPBase]
    ) -> "SOP":
        """Add inplace a symbolic operator type to the current SOP to obtain a SOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: SOP
        """
        pass

    @abstractmethod
    def __sub__(
        self, op: Union[float, complex, np.float64, np.complex128, OPBase]
    ) -> "SOP":
        """Subtract a symbolic operator type to the current SOP to obtain a SOP

        :param op: The operator to be subtracted to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: SOP
        """
        pass

    @abstractmethod
    def __rsub__(
        self, op: Union[float, complex, np.float64, np.complex128, OPBase]
    ) -> "SOP":
        """Subtract the SOP object from an operator or number

        :param op: The operator to be subtracted to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: SOP
        """
        pass

    @abstractmethod
    def __isub__(
        self, op: Union[float, complex, np.float64, np.complex128, OPBase]
    ) -> "SOP":
        """Subtract inplace a symbolic operator type to the current SOP to obtain a SOP

        :param op: The operator to be subtracted to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: SOP
        """
        pass

    @abstractmethod
    def __idiv__(self, v: Union[float, complex, np.float64, np.complex128]) -> "SOP":
        """Functions in place division a SOP by a scalar .

        :param v: The scalar to divide the SOP by
        :type v: float | complex | np.float64 | np.complex128
        :return: A sum-of-product operator storing the result
        :rtype: SOP
        """
        pass

    @abstractmethod
    def __imul__(
        self,
        v: Union[float, complex, np.float64, np.complex128],
    ) -> "SOP":
        """Functions for inplace multiplyication of a SOP by a scalar or operator type.

        :param v: The object to multiply the SOP by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: A sum-of-product operator type storing the result
        :rtype: SOP
        """
        pass


SOP.register(SOP_complex)
if _support_real_SOP:
    SOP.register(SOP_real)

class multiset_SOP(metaclass=ABCMeta):
    """A class for storing a compact multiset sum-of-product operator object."""

    def __new__(
        cls,
        *args,
        dtype: Union[float, complex, np.float64, np.complex128] = np.complex128,
    ) -> "multiset_SOP":
        """Factory function for constructing a multiset sum-of-product compact string operator.

        :param `*args`: Variable length list of arguments. This function can handle two possible lists of arguments

            - nset( int), N (int) - The number of set variables, The number of modes of the SOP
            - nset( int), N (int), label (str) - The number of set varibles. The number of modes of the SOP and a label for the SOP
        :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional

        :returns: The multiset_SOP object
        :rtype: multiset_SOP
        """
        if _support_real_SOP:
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

    @abstractmethod
    def assign(self, o: "multiset_SOP"):
        """Assign the value of the multiset_SOP from another

        :param o: The multiset SOP to copy into this one
        :type o: multiset_SOP
        """
        pass

    @abstractmethod
    def __copy__(self):
        """Function implementing shallow copy of the multiset_SOP object"""
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the multiset_SOP object"""
        pass

    @abstractmethod
    def clear(self):
        """Clear all internal storage used by the multiset_SOP object"""
        pass

    @abstractmethod
    def resize(self, nset: int, n: int):
        """Resizes the multiset_SOP object so that it can act on n modes

        :param nset: The number of set variables defining the multiset SOP
        :type nset: int
        :param n: The number of modes
        :type n: int
        """

    @abstractmethod
    def nset(self) -> int:
        """Returns the number of set variables the multiset SOP acts on

        :return: The number of set variables that the multiset_SOP acts on
        :rtype: int
        """

    @abstractmethod
    def nmodes(self) -> int:
        """Returns the number of modes that the multiset_SOP acts on

        :return: The number of modes that the multiset_SOP acts on
        :rtype: int
        """

    @abstractmethod
    def nterms(self) -> int:
        """Returns the number of set variables that are set in the multiset_SOP

        :return: The number of set variables in the multiset_SOP
        :rtype: int
        """
        pass

    @abstractmethod
    def set(self, i: int, j: int, vals: SOP):
        """Set the [i, j] element of the multiset SOP to the value vals

        :param i: The outgoing set index
        :type i: int
        :param j: The incoming set index
        :type j: int
        :param vals: A SOP defining the operator terms associated with set index [i, j]
        :type vals: SOP
        """
        pass

    @abstractmethod
    def set_is_fermion_mode(self, is_fermion_mode: list[bool]) -> bool:
        """Set whether or not the modes in the multiset SOP object are fermionic

        :param is_fermion_mode: A list specifying whether each mode is fermionic
        :type is_fermion_mode: list[bool]
        :return: Whether or not this set operation was successful
        :rtype: bool
        """
        pass

    @abstractmethod
    def prune_zeros(self, tol: float = 1e-15):
        """Remove any terms in the Hamiltonian definition that are zero within the user specified tolerance

        :param tol: A pruning tolerance for removing zero terms, defaults to 1e-15
        :type tol: float, optional
        """
        pass

    @abstractmethod
    def jordan_wigner(self, sysinf: system_modes, tol: float = 1e-15) -> "SOP":
        """Perform an inplace Jordan-Wigner mapping of the multiset SOP object

        :param sysinf: Definition of the modes forming the system
        :type sysinf: system_modes
        :param tol: A pruning tolerance for removing zero terms, defaults to 1e-15
        :type tol: float, optional
        :return: The jordan-wigner mapped SOP object
        :rtype: SOP
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the SOP object

        :return: The string representation of the SOP
        :rtype: str
        """
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        """A string labelling the operator"""
        pass

    @abstractmethod
    def __getitem__(self, a: tuple[int, int]) -> SOP:
        """Return the SOP associated with a specific set index in the multiset SOP object

        :param a: The set index to access
        :type a: tuple[int, int]
        :return:  The SOP associated with a specific set index in the multiset SOP object
        :rtype: SOP
        """
        pass

    @abstractmethod
    def __setitem__(self, a: tuple[int, int], v: SOP):
        """Set the SOP associated with a specific set index in the multiset SOP object

        :param a: The set index to edit
        :type a: tuple[int, int]
        :param v:  The SOP associated with a specific set index in the multiset SOP object
        :type v: SOP
        """
        pass


multiset_SOP.register(multiset_SOP_complex)
if _support_real_SOP:
    SOP.register(multiset_SOP_real)

ms_SOP = multiset_SOP
ms_SOPBase = multiset_SOP
