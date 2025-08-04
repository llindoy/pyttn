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

from pyttn.ttnpp import operator_dictionary_complex
from pyttn.ttns.operators.siteOperatorsExt import site_operator

try:
    from pyttn.ttnpp import operator_dictionary_real

    _real_opdict = True
except ImportError:
    _real_opdict = False


class OperatorDictionary(metaclass=ABCMeta):
    def __new__(
        cls,
        *args,
        dtype: Union[float, complex, np.float64, np.complex128] = np.complex128,
    ) -> "OperatorDictionary":
        """Factory function for constructing a user defined operator dictionary.

        :param `*args`: Variable length list of arguments.
        :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
        :type dtype: {np.float64, np.complex128}, optional

        :returns: The operator dictionary object
        :rtype: operator_dictionary_real or operator_dictionary_complex
        """

        if _real_opdict:
            if dtype == np.complex128 or dtype is complex:
                return operator_dictionary_complex(*args)
            elif dtype == np.float64 or dtype is float:
                return operator_dictionary_real(*args)
            else:
                raise RuntimeError("Invalid dtype for OperatorDictionary")

        else:
            if dtype == np.complex128 or dtype is complex:
                return operator_dictionary_complex(*args)
            else:
                raise RuntimeError("Invalid dtype for OperatorDictionary")

    @abstractmethod
    def assign(self, o: "OperatorDictionary"):
        """Assign the value of the operator dictionary from another

        :param o: The operator dictionary to copy into this one
        :type o: OperatorDictionary
        """
        pass

    @abstractmethod
    def __copy__(self):
        """Function implementing shallow copy of the operator dictionary object"""
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        """Function implementing deep copy of the operator dictionary object"""
        pass

    @abstractmethod
    def clear(self):
        """Clear all internal storage used by the operator dictionary object"""
        pass

    @abstractmethod
    def resize(self, size: int):
        """Resize the operator dictionary object so that it can describe a system with size modes

        :param size: The number of modes
        :type size: int
        """

    @abstractmethod
    def __setitem__(self, mode: int, el: dict[str, 'site_operator']):
        """Set the operator dictionary for mode mode

        :param mode: The mode to be set
        :type mode: int
        :param el: A dictionary of strings and site_operator objects containing the operators that act on this mode
        :type el: dict[str, site_operator]
        """
        pass

    @abstractmethod
    def __getitem__(self, mode: int) -> dict[str, 'site_operator']:
        """Get the operator dictionary for mode mode

        :param mode: The mode to be set
        :type mode: int
        :returns: A dictionary of strings and site_operator objects containing the operators that act on this mode
        :rtype: dict[str, site_operator]
        """
        pass

    @abstractmethod
    def site_dictionary(self, mode: int) -> dict[str, 'site_operator']:
        """Get the operator dictionary for mode mode

        :param mode: The mode to be set
        :type mode: int
        :returns: A dictionary of strings and site_operator objects containing the operators that act on this mode
        :rtype: dict[str, site_operator]
        """
        pass

    @abstractmethod
    def insert(self, mode: int, label: str, op: 'site_operator'):
        """Insert the operator op with label label into the site operator dictionary associated with mode mode

        :param mode: The mode to consider
        :type mode: int
        :param label: The label used to represent the operator
        :type label: str
        :param op: The site_operator object
        :type op: site_operator
        """
        pass

    @abstractmethod
    def __call__(self, mode: int, label: str) -> 'site_operator':
        """Return the operator with label label in the dictionary of mode mode

        :param mode: The mode to consider
        :type mode: int
        :param label: The label used to represent the operator
        :type label: str
        :returns: The site_operator object
        :rtype: site_operator
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of modes in the operator dictionary

        :return: The number of modes in the dictionary
        :rtype: int
        """
        pass

    @abstractmethod
    def nmodes(self) -> int:
        """Returns the number of modes in the operator dictionary

        :return: The number of modes in the dictionary
        :rtype: int
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the operator dictionary object

        :return: The string representation of the operator dictionary
        :rtype: str
        """
        pass

    @abstractmethod
    def backend(self) -> str:
        """Returns the backend type of the operator dictionary

        :return: The backend type of the object
        :rtype: str
        """
        pass


OperatorDictionary.register(operator_dictionary_complex)
if _real_opdict:
    OperatorDictionary.register(operator_dictionary_real)


operator_dictionary = OperatorDictionary
