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

from pyttn.ttnpp import (
    coeff_complex,
    coeff_real,
    sNBO_complex,
    sNBO_real,
    sSOP_complex,
    sSOP_real,
)
from pyttn.ttnpp import sOP as _sOP
from pyttn.ttnpp import sPOP as _sPOP


class OPBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        """Base class for all symbolic string operator types:
        
            - sOP
            - sPOP 
            - sNBO
            - sSOP
        """
        pass


class sOP(OPBase):
    """The single site operator used for the string operator handling functionality of pyTTN.  This class allows for definition of
    a string label for an operator and the mode that the operator acts upon. In addition to allowing for arbitrary string labels
    with the combination of user defined operator dictionaries.  This code supports several automatic dictionaries depending on
    the type of mode considered.  These are

    Fermion Modes

      - Annihilation operator :math:`\\hat{c}` :  {"c", "a", "f"}
      - Creation operator :math:`\\hat{c}^\\dagger` :  {"cdag", "adag", "fdag", "cd", "ad", "fd"}
      - Number operator :math:`\\hat{c}^\\dagger\\hat{c}` :  {"n", "cdagc", "adaga", "fdagf", "cdc", "ada", "fdf"}
      - Vacancy operator :math:`1-\\hat{c}^\\dagger\\hat{c}` :  "v"

    Bosonic Modes

      - Annihilation operator :math:`\\hat{c}` :  {"c", "a", "b"}
      - Creation operator :math:`\\hat{c}^\\dagger` :  {"cdag", "adag", "bdag", "cd", "ad", "bd"}
      - Number operator :math:`\\hat{c}^\\dagger\\hat{c}` :  {"n", "cdagc", "adaga", "bdagb", "cdc", "ada", "bdb"}
      - Position operator :math:`\\hat{q}` : {"q", "x"}
      - Momentum operator :math:`\\hat{p}` : "p"
      - Kinetic Energy Operator :math:`\\frac{1}{2} \\hat{p}^2` : "ke"
      - Powers of any of the above operators (x) :math:`\\hat{x}^n` Y "x^n"

    Spin Modes for arbitrary spin S

      - :math:`\\hat{S}_x` : {"sx", "x"}
      - :math:`\\hat{S}_y` : {"sy", "y"}
      - :math:`\\hat{S}_z` : {"sz", "z"}
      - :math:`\\hat{S}_+` : {"s+", "sp"}
      - :math:`\\hat{S}_-` : {"s-", "sm"}

    Two Level System Modes

      - :math:`\\hat{\\sigma}_x` : {"sx", "x", "sigmax"}
      - :math:`\\hat{\\sigma}_y` : {"sy", "y", "sigmay"}
      - :math:`\\hat{\\sigma}_z` : {"sz", "z", "sigmaz"}
      - :math:`\\hat{\\sigma}_+` : {"s+", "sp", "sigma+", "sigmap"}
      - :math:`\\hat{\\sigma}_-` : {"s-", "sm", "sigma-", "sigmam"}

    N Level System Modes

      - :math:`\\left|m\\right\\rangle\\left\\langle n \\right|` : {"`|m><n|`"}

    """

    def __new__(cls, label: str, mode: int, is_fermionic: bool = False):
        """A function for constructing a new sOP object

        :param label: The label of the sOP
        :type label: str
        :param mode: The mode the sOP acts on
        :type mode: int
        :param is_fermionic: Whether or not the operator is a fermionic operator
        :type is_fermionic: bool, optional
        """
        return _sOP(label, mode, is_fermionic)

    
    def assign(self, o: "sOP"):
        """Assign the value of the sOP from another

        :param o: The sOP to copy into this one
        :type o: sOP
        """
        pass

    
    def __copy__(self):
        """Function implementing shallow copy of the sOP object"""
        pass

    
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the sOP object"""
        pass

    
    def clear(self):
        """Clear all internal storage used by the sOP object"""
        pass

    @property
    def fermionic(self) -> bool:
        """Returns whether or not this represents a fermionic operator

        :return: Whether or not this representas a fermionic operator
        :rtype: bool
        """

    @property
    def mode(self) -> int:
        """Returns the mode the sOP object acts on

        :return: The mode the sOP object acts on
        :rtype: int
        """

    @property
    def op(self) -> str:
        """Returns the label of the sOP object

        :return: The label of the sOP object
        :rtype: str
        """

    
    def __add__(self, op: OPBase) -> "sSOP":
        """Add a symbolic operator type to the current sOP to obtain a sSOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __sub__(self, op: OPBase) -> "sSOP":
        """Subtract a symbolic operator type to the current sOP to obtain a sSOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __div__(self, v: Union[float, complex, np.float64, np.complex128]) -> "sNBO":
        """Functions for dividing a sOP by a scalar .

        :param v: The scalar to divide the sOP bu
        :type v: float | complex | np.float64 | np.complex128
        :return: An n-body operator type storing the result
        :rtype: sNBO
        """
        pass

    
    def __mul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, "coeff", OPBase],
    ) -> Union["sPOP", "sNBO", "sSOP"]:
        """Functions for multiplying a sOP by a scalar or operator type.
        The return type depends on the the type of the other object and is either:

            * :class:`sPOP` : If v is :class:`sOP` | :class:`sPOP`
            * :class:`sNBO` : If v is float | complex | np.float64 | np.complex128 | :class:`coeff` | :class:`sNBO`
            * :class:`sSOP` : If v is :class:`sSOP`

        :param v: The object to multiply the sOP by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: An n-body operator type storing the result
        :rtype: sPOP | sNBO | sSOP
        """
        pass

    
    def __rmul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, "coeff", OPBase],
    ) -> Union["sPOP", "sNBO", "sSOP"]:
        """Functions for right multiplying a sOP by a scalar or operator type.
        The return type depends on the the type of the other object and is either:

            * :class:`sPOP` : If v is :class:`sOP` | :class:`sPOP`
            * :class:`sNBO` : If v is float | complex | np.float64 | np.complex128 | :class:`coeff` | :class:`sNBO`
            * :class:`sSOP` : If v is :class:`sSOP`

        :param v: The object to multiply the sOP by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: An n-body operator type storing the result
        :rtype: sPOP | sNBO | sSOP
        """
        pass

    
    def __str__(self) -> str:
        """Return the string representation of the sOP object

        :return: The string representation of the sOP
        :rtype: str
        """
        pass


sOP.register(_sOP)
sOPBase = sOP


class sPOP(OPBase):
    """The string Product Operator used for storing a product of sOP objects"""

    def __new__(cls, *args):
        """A function for creating a new sPOP object

        :param `*args`: A variable list for specifying the coefficient.  Valid options are

            -  Default construct the sPOP
            - op (:class:`sOP`) - Construct sPOP from single site operator
            - ops (list[:class:`sOP`]) - Construct sPOP from a list of single site operators
            - pop (:class:`sPOP`) - Construct sPPO from product operator
        :return: The sPOP object
        :rtype: sPOP
        """
        return _sPOP(*args)

    
    def assign(self, o: "sPOP"):
        """Assign the value of the sPOP from another

        :param o: The sPOP to copy into this one
        :type o: sPOP
        """
        pass

    
    def __copy__(self):
        """Function implementing shallow copy of the sPOP object"""
        pass

    
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the sPOP object"""
        pass

    
    def clear(self):
        """Clear all internal storage used by the sPOP object"""
        pass

    
    def insert_front(self, o: sOP):
        """Insert a sOP object at the front of this product.

        :param o: The sOP to insert
        :type o: sOP
        """

    
    def insert_back(self, o: sOP):
        """Insert a sOP object at the back of this product.

        :param o: The sOP to insert
        :type o: sOP
        """

    
    def nmodes(self) -> int:
        """Returns the number of modes that the sPOP acts on

        :return: The number of modes that the sPOP acts on
        :rtype: int
        """

    
    def size(self) -> int:
        """Returns the number of sOP objects in the sPOP

        :return: The number of sOP objects in the sPOP
        :rtype: int
        """

    @property
    
    def ops(self) -> list[sOP]:
        """Returns the list of sOP operators

        :return: The list of sOP operators
        :rtype: list[sOP]
        """

    
    def __add__(self, op: OPBase) -> "sSOP":
        """Add a symbolic operator type to the current sPOP to obtain a sSOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __sub__(self, op: OPBase) -> "sSOP":
        """Subtract a symbolic operator type to the current sPOP to obtain a sSOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __div__(self, v: Union[float, complex, np.float64, np.complex128]) -> "sNBO":
        """Functions for dividing a sPOP by a scalar .

        :param v: The scalar to divide the sPOP by
        :type v: float | complex | np.float64 | np.complex128
        :return: An n-body operator type storing the result
        :rtype: sNBO
        """
        pass

    
    def __mul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, "coeff", OPBase],
    ) -> Union["sPOP", "sNBO", "sSOP"]:
        """Functions for multiplying a sPOP by a scalar or operator type.
        The return type depends on the the type of the other object and is either:

            * :class:`sPOP` : If v is :class:`sOP` | :class:`sPOP`
            * :class:`sNBO` : If v is float | complex | np.float64 | np.complex128 | :class:`coeff` | :class:`sNBO`
            * :class:`sSOP` : If v is :class:`sSOP`

        :param v: The object to multiply the sPOP by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: An n-body operator type storing the result
        :rtype: sPOP | sNBO | sSOP
        """
        pass

    
    def __rmul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, "coeff", OPBase],
    ) -> Union["sPOP", "sNBO", "sSOP"]:
        """Functions for right multiplying a sPOP by a scalar or operator type.
        The return type depends on the the type of the other object and is either:

            * :class:`sPOP` : If v is :class:`sOP` | :class:`sPOP`
            * :class:`sNBO` : If v is float | complex | np.float64 | np.complex128 | :class:`coeff` | :class:`sNBO`
            * :class:`sSOP` : If v is :class:`sSOP`

        :param v: The object to multiply the sPOP by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: An n-body operator type storing the result
        :rtype: sPOP | sNBO | sSOP
        """
        pass

    
    def __str__(self) -> str:
        """Return the string representation of the sPOP object

        :return: The string representation of the sPOP
        :rtype: str
        """
        pass

    
    def __iter__(self):
        """Return an iterator over the sOP objects in the sPOP"""


sPOP.register(_sPOP)
sPOPBase = sPOP


class coeff(metaclass=ABCMeta):
    """A class for handling potentially time-dependent coefficients"""

    def __new__(
        cls,
        *args,
        dtype: Union[float, complex, np.float64, np.complex128] = np.complex128,
    ) -> "coeff":
        """A function for constructing the coeff type for Hamiltonian specification

        :param `*args`: A variable list for specifying the coefficient.  Valid options are

            - Default construct the coefficient
            - value (dtype) - Set the coefficient to a constant value
            - func (callable) - set the coefficient to a time-dependent value
        :param dtype: The internal variable type for the product operator.
        :type dtype: {np.float64, np.complex128}, optional

        :returns: The coefficient object
        :rtype: coeff
        """
        if dtype == np.complex128 or dtype is complex:
            return coeff_complex(*args)
        elif dtype == np.float64 or dtype is float:
            return coeff_real(*args)
        else:
            raise RuntimeError("Invalid dtype for sNBO")

    @abstractmethod
    def assign(self, o: "coeff"):
        """Assign the value of the coeff from another

        :param o: The coeff to copy into this one
        :type o: coeff
        """
        pass

    @abstractmethod
    def __copy__(self):
        """Function implementing shallow copy of the coeff object"""
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the coeff object"""
        pass

    @abstractmethod
    def clear(self):
        """Clear all internal storage used by the coeff object"""
        pass

    @abstractmethod
    def is_zero(self) -> bool:
        """Whether or not the coefficient stores the zero value

        :returns: Whether of not the coefficient stores the zero value
        :rtype: bool
        """

    @abstractmethod
    def is_positive(self) -> bool:
        """Whether or not the coefficient is positive

        :returns: Whether of not the coefficient is positive
        :rtype: bool
        """
    
    @abstractmethod
    def is_time_dependent(self) -> bool:
        """Whether or not the coefficient is time dependent

        :returns: Whether of not the coefficient is time dependent
        :rtype: bool
        """
    
    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the coeff object

        :return: The string representation of the coeff
        :rtype: str
        """
        pass

    @abstractmethod
    def __call__(self, t: Union[float, np.float64]) -> Union[float, complex]:
        """Return the value of the coefficient at a given time point

        :param t: The time to evaluate the coefficient at
        :type t: float | np.float64
        :return: The value of the coeffficient at time t
        :rtype: float | complex
        """
        pass

    @abstractmethod
    def __add__(
        self, op: Union[float, complex, np.float64, np.complex128, "coeff"]
    ) -> "coeff":
        """Add a scalar to a coeff object

        :param op: The scalar to be added to the current operator
        :type op: float | complex | np.float64 | np.complex128 | coeff
        :return: The coeff type storing the result
        :rtype: coeff
        """
        pass

    @abstractmethod
    def __radd__(self, op: Union[float, complex, np.float64, np.complex128]) -> "coeff":
        """Add a scalar to a coeff object

        :param op: The scalar to be added to the current operator
        :type op: float | complex | np.float64 | np.complex128
        :return: The coeff type storing the result
        :rtype: coeff
        """
        pass

    @abstractmethod
    def __iadd__(
        self, op: Union[float, complex, np.float64, np.complex128, "coeff"]
    ) -> "coeff":
        """Inplace addition of a coeff from a scalar

        :param op: The scalar to be added
        :type op: float | complex | np.float64 | np.complex128 | coeff
        :return: The coeff type storing the result
        :rtype: coeff
        """
        pass

    @abstractmethod
    def __sub__(
        self, op: Union[float, complex, np.float64, np.complex128, "coeff"]
    ) -> "coeff":
        """Subtract a scalar to a coeff object

        :param op: The scalar to be subtracted from the current operator
        :type op: float | complex | np.float64 | np.complex128 | coeff
        :return: The coeff type storing the result
        :rtype: coeff
        """
        pass

    @abstractmethod
    def __rsub__(self, op: Union[float, complex, np.float64, np.complex128]) -> "coeff":
        """Subtract a coeff from a scalar

        :param op: The scalar that the current coeff is to be subtracted from
        :type op: float | complex | np.float64 | np.complex128
        :return: The coeff type storing the result
        :rtype: coeff
        """
        pass

    @abstractmethod
    def __isub__(
        self, op: Union[float, complex, np.float64, np.complex128, "coeff"]
    ) -> "coeff":
        """Inplace subtraction of a coeff from a scalar

        :param op: The scalar to be subtracted
        :type op: float | complex | np.float64 | np.complex128 | coeff
        :return: The coeff type storing the result
        :rtype: coeff
        """
        pass

    @abstractmethod
    def __div__(self, v: Union[float, complex, np.float64, np.complex128]) -> "coeff":
        """Functions for dividing a coeff by a scalar.

        :param v: The scalar to divide the sPOP by
        :type v: float | complex | np.float64 | np.complex128
        :return: A type storing the result
        :rtype: coeff
        """
        pass

    @abstractmethod
    def __idiv__(self, v: Union[float, complex, np.float64, np.complex128]) -> "coeff":
        """Functions for inplace division of a coeff by a scalar

        :param v: The object to divide the coeff by
        :type v: float | complex | np.float64 | np.complex128
        :return: A coeff type storing the result
        :rtype: coeff
        """
        pass

    @abstractmethod
    def __mul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, "coeff", OPBase],
    ) -> Union["coeff", "sNBO", "sSOP"]:
        """Functions for multiplying a coeff by a scalar or operator type.
        The return type depends on the the type of the other object and is either:

            * :class:`coeff` : If v is float | complex | np.float64 | np.complex128 | :class:`coeff`
            * :class:`sNBO` : If v is :class:`sOP` | :class:`sPOP` | :class:`sNBO`
            * :class:`sSOP` : If v is :class:`sSOP`

        :param v: The object to multiply the sPOP by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: A coeff type storing the result
        :rtype: coeff | sNBO | sSOP
        """
        pass

    @abstractmethod
    def __rmul__(
        self,
        v: Union[float, complex, np.float64, np.complex128],
    ) -> "coeff":
        """Functions for right multiplying a coeff by a scalar

        :param v: The object to multiply the coeff by
        :type v: float | complex | np.float64 | np.complex128
        :return: A coeff type storing the result
        :rtype: coeff
        """
        pass

    @abstractmethod
    def __imul__(
        self,
        v: Union[float, complex, np.float64, np.complex128],
    ) -> "coeff":
        """Functions for inplace multiplication of a coeff by a scalar

        :param v: The object to multiply the coeff by
        :type v: float | complex | np.float64 | np.complex128
        :return: A coeff type storing the result
        :rtype: coeff
        """
        pass


coeff.register(coeff_complex)
coeff.register(coeff_real)


class sNBO(OPBase):
    """A class for handling an n-body operator string"""

    def __new__(cls, *args, dtype=np.complex128):
        """Construct a new n-body operator object

        :param `*args`: A variable list for specifying the coefficient.  Valid options are

            -  Default construct the sNBO
            - op (:class:`sOP`) - Construct NBO from single site operator
            - pop (:class:`sPOP`) - Construct NBO from product operator
            - arg (float ), op (:class:`sOP`) - Construct NBO as a product of a constant and a single site operator
            - arg (dtype), pop (:class:`sPOP`) - Construct NBO as a product of a constant and a product operator
            - arg (:class:`coeff`), op (:class:`sOP`) - Construct NBO as a product of a coefficient and a single site operator
            - arg (:class:`coeff`), pop (:class:`sPOP`) - Construct NBO as a product of a coefficient and a product operator
            - nbo (:class:`sNBO`) - Construct NBO from another NBO

        :param dtype: The internal variable type for the n-body operator string.
        :type dtype: {np.float64, np.complex128, float, complex}, optional

        :returns: The n-body operator object
        :rtype: sNBO
        """
        if dtype == np.complex128 or dtype is complex:
            return sNBO_complex(*args)
        elif dtype == np.float64 or dtype is float:
            return sNBO_real(*args)
        else:
            raise RuntimeError("Invalid dtype for sNBO")

    
    def assign(self, o: "sNBO"):
        """Assign the value of the sNBO from another

        :param o: The sNBO to copy into this one
        :type o: sNBO
        """
        pass

    
    def __copy__(self):
        """Function implementing shallow copy of the sNBO object"""
        pass

    
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the sNBO object"""
        pass

    
    def clear(self):
        """Clear all internal storage used by the sNBO object"""
        pass

    
    def insert_front(self, o: sOP):
        """Insert a sOP object at the front of this product.

        :param o: The sOP to insert
        :type o: sOP
        """

    
    def insert_back(self, o: sOP):
        """Insert a sOP object at the back of this product.

        :param o: The sOP to insert
        :type o: sOP
        """

    
    def nmodes(self) -> int:
        """Returns the number of modes that the sNBO acts on

        :return: The number of modes that the sNBO acts on
        :rtype: int
        """

    @property
    
    def ops(self) -> list[sOP]:
        """Returns the list of sOP operators

        :return: The list of sOP operators
        :rtype: list[sOP]
        """

    @property
    
    def pop(self) -> sPOP:
        """Returns the product operator associated with the sNBO

        :return: The product operator associated with the sNBO
        :rtype: sPOP
        """
    
    def __add__(self, op: OPBase) -> "sSOP":
        """Add a symbolic operator type to the current sNBO to obtain a sSOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __sub__(self, op: OPBase) -> "sSOP":
        """Subtract a symbolic operator type to the current sNBO to obtain a sSOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __div__(self, v: Union[float, complex, np.float64, np.complex128]) -> "sNBO":
        """Functions for dividing a sNBO by a scalar .

        :param v: The scalar to divide the sNBO by
        :type v: float | complex | np.float64 | np.complex128
        :return: An n-body operator type storing the result
        :rtype: sNBO
        """
        pass

    
    def __mul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, coeff, OPBase],
    ) -> Union["sNBO", "sSOP"]:
        """Functions for multiplying a sNBO by a scalar or operator type.
        The return type depends on the the type of the other object and is either:

            * :class:`sNBO` : If v is float | complex | np.float64 | np.complex128 | :class:`coeff` | :class:`sOP` | :class:`sPOP` | :class:`sNBO`
            * :class:`sSOP` : If v is :class:`sSOP`

        :param v: The object to multiply the sNBO by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: An operator type storing the result
        :rtype: sNBO | sSOP
        """
        pass

    
    def __rmul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, coeff, OPBase],
    ) -> Union["sNBO", "sSOP"]:
        """Functions for right multiplying a sNBO by a scalar or operator type.
        The return type depends on the the type of the other object and is either:

            * :class:`sNBO` : If v is float | complex | np.float64 | np.complex128 | :class:`coeff` | :class:`sOP` | :class:`sPOP` | :class:`sNBO`
            * :class:`sSOP` : If v is :class:`sSOP`

        :param v: The object to multiply the sNBO by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: An operator type storing the result
        :rtype: sNBO | sSOP
        """
        pass

    
    def __str__(self) -> str:
        """Return the string representation of the sNBO object

        :return: The string representation of the sNBO
        :rtype: str
        """
        pass

    
    def __iter__(self):
        """Return an iterator over the sOP objects in the sNBO"""
        
    @property
    def coeff(self) -> coeff:
        """Returns the coefficient of the sNBO

        :return: The coefficient of the sNBO
        :rtype: coeff
        """

sNBO.register(sNBO_complex)
sNBO.register(sNBO_real)


class sSOP(OPBase):
    """Class for handling sum-of-product operator string definitions"""

    def __new__(
        cls,
        *args,
        dtype: Union[float, complex, np.float64, np.complex128] = np.complex128,
    ) -> "sSOP":
        """A function for constructing a sum-of-product string operator

        :param `*args`: A variable list for specifying the coefficient.  Valid options are

            -  Default construct the sSOP
            - op (str) - Construct the sSOP from a string defining a sOP
            - op (:class:`sOP`) - Construct sSOP from single site operator
            - pop (:class:`sPOP`) - Construct sSOP from product operator
            - nbo (:class:`sNBO`) - Construct sSOP from an sNBO
            - sop (:class:`sSOP`) - Construct sSOP from another sSOP

        :param dtype: The internal variable type for the sum-of-product string operator.
        :type dtype: {np.float64, np.complex128}, optional

        :returns: The sum-of-product string operator
        :rtype: sSOP
        """
        if dtype == np.complex128 or dtype is complex:
            return sSOP_complex(*args)
        elif dtype == np.float64 or dtype is float:
            return sSOP_real(*args)
        else:
            raise RuntimeError("Invalid dtype for sSOP")

    
    def assign(self, o: "sSOP"):
        """Assign the value of the sSOP from another

        :param o: The sSOP to copy into this one
        :type o: sSOP
        """
        pass

    
    def __copy__(self):
        """Function implementing shallow copy of the sSOP object"""
        pass

    
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the sSOP object"""
        pass

    
    def clear(self):
        """Clear all internal storage used by the sSOP object"""
        pass

    
    def reserve(self, n: int):
        """Allocate the internal buffer for the sSOP object so that it is capable of storing n sPOP terms

        :param n: The number of elements that should be reserved.
        :type n: int
        """

    
    def nmodes(self) -> int:
        """Returns the number of modes that the sSOP acts on

        :return: The number of modes that the sSOP acts on
        :rtype: int
        """

    
    def nterms(self) -> int:
        """Returns the number of terms in the sSOP

        :return: The number of terms in the sSOP
        :rtype: int
        """

    
    def __len__(self) -> int:
        """Returns the number of terms in the sSOP

        :return: The number of terms in the sSOP
        :rtype: int
        """

    
    def __iter__(self):
        """Return an iterator over the sNBO objects in the sSOP"""

    @property
    
    def label(self) -> str:
        """A string labelling the operator

        :return: The string labelling the operator
        :rtype: str
        """
        pass

    @property
    
    def terms(self) -> list[sNBO]:
        """The sNBO terms in the sSOP

        :return: The sNBO terms in the sSOP
        :rtype: list[sNBO]
        """
        pass

    
    def __setitem__(self, i: int, v: sNBO):
        """Set the value of the ith sNBO term to v

        :param i: index of the term in the sSOP
        :type i: int
        :param v: The sNBO value to be used as the new value of this term
        :type v: sNBO
        """

    
    def __getitem__(self, i: int) -> sNBO:
        """Return the value of the ith sNBO term

        :param i: index of the term in the sSOP
        :type i: int
        :returns: The sNBO value to be used as the new value of this term
        :rtype: sNBO
        """

    
    def __str__(self) -> str:
        """Return the string representation of the sSOP object

        :return: The string representation of the sSOP
        :rtype: str
        """
        pass

    
    def __add__(self, op: OPBase) -> "sSOP":
        """Add a symbolic operator type to the current sSOP to obtain a sSOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __iadd__(self, op: OPBase) -> "sSOP":
        """Add inplace a symbolic operator type to the current sSOP to obtain a sSOP

        :param op: The operator to be added to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __sub__(self, op: OPBase) -> "sSOP":
        """Subtract a symbolic operator type to the current sSOP to obtain a sSOP

        :param op: The operator to be subtracted to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __isub__(self, op: OPBase) -> "sSOP":
        """Subtract inplace a symbolic operator type to the current sSOP to obtain a sSOP

        :param op: The operator to be subtracted to the current operator
        :type op: OPBase
        :return: The sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __div__(self, v: Union[float, complex, np.float64, np.complex128]) -> "sSOP":
        """Functions for dividing a sSOP by a scalar .

        :param v: The scalar to divide the sSOP by
        :type v: float | complex | np.float64 | np.complex128
        :return: A sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __idiv__(self, v: Union[float, complex, np.float64, np.complex128]) -> "sSOP":
        """Functions in place division a sSOP by a scalar .

        :param v: The scalar to divide the sSOP by
        :type v: float | complex | np.float64 | np.complex128
        :return: A sum-of-product operator storing the result
        :rtype: sSOP
        """
        pass

    
    def __mul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, coeff, OPBase],
    ) -> "sSOP":
        """Functions for multiplying a sSOP by a scalar or operator type.

        :param v: The object to multiply the sSOP by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: A sum-of-product operator type storing the result
        :rtype: sSOP
        """
        pass

    
    def __imul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, coeff, OPBase],
    ) -> "sSOP":
        """Functions for inplace multiplyication of a sSOP by a scalar or operator type.

        :param v: The object to multiply the sSOP by
        :type v: float | complex | np.float64 | np.complex128 | coeff | OPBase
        :return: A sum-of-product operator type storing the result
        :rtype: sSOP
        """
        pass

    
    def __rmul__(
        self,
        v: Union[float, complex, np.float64, np.complex128, coeff],
    ) -> "sSOP":
        """Functions for right multiplying a sSOP by a scalar type.

        :param v: The object to multiply the sSOP by
        :type v: float | complex | np.float64 | np.complex128 | coeff
        :return: A sum-of-product operator type storing the result
        :rtype: sSOP
        """
        pass


sSOP.register(sSOP_complex)
sSOP.register(sSOP_real)
