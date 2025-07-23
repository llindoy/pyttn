"""A module with classes for setting ntree elements from leaves to root."""

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

import abc
from typing import Optional

import numpy as np

from pyttn.ttnpp import ntreeNode


class BondDimensionSetter(metaclass=abc.ABCMeta):
    """Abstract base class for setting bond dimensions.

    Abstract base class for objects that set the bond dimension variables
    stored in an ntree object dependent on the values in the children of the node.
    """

    @abc.abstractmethod
    def __call__(self, node: ntreeNode) -> None:
        """Default call function.

        :param node: The node which will have its bond dimension set
        :type node: ntreeNode
        """


class NodeSumSetter(BondDimensionSetter):
    """A BondDimensionSetter object that sets the node to the sum of its children."""

    def __call__(self, node: ntreeNode) -> None:
        """Apply the NodeSumSetter class to an ntreeNode.

        Sets the value of the bond dimension variable in the current node to the sum
        of the bond dimensions associated with the child nodes

        :param node: The node which will have its bond dimension set
        :type node: ntreeNode
        """
        ind = 0
        for i in range(node.size()):
            ind += node.at(i).value
        node.value = ind


class NodeIncrementSetter(BondDimensionSetter):
    """A class for setting the node by adding to the value of its children."""
    def __init__(
        self,
        increment_value: int,
        combination: str = "mean",
        maxchi: Optional[int] = None,
    ) -> None:
        """Construct a NodeIncrementSetter object.

        :param increment_value: The value to increment the node value by
        :type increment_value: int
        :param combination: How to combine the values of the child nodes to get the
             value to increment by, defaults to "mean"
        :type combination: {"mean", "min", "max"}, optional
        :param maxchi: The maximum allowed chi throughout the network
        :type maxchi: int, optional
        :raises RuntimeError: Raises an error if the combination variable is not in
             the allowed set of values
        """
        self.__increment_value = increment_value
        self.__maxchi = maxchi
        if combination not in ["mean", "min", "max"]:
            message="Invalid combination rule for NodeIncrementSetter"
            raise RuntimeError(message)
        self.__child_combination = combination

    def __call__(self, node: ntreeNode) -> None:
        """Apply the NodeIncrementSetter.

        Set the value of the bond dimension variable in the current node to be
        bond_dim = func(child_bond_dims) + inc

        :param node: The node which will have its bond dimension set
        :type node: ntreeNode
        """
        _vars = [node.at(i).value for i in range(node.size())]

        if self.__child_combination == "mean":
            node.value = int(np.mean(_vars)) + self.__increment_value
        elif self.__child_combination == "min":
            node.value = min(_vars) + self.__increment_value
        elif self.__child_combination == "max":
            node.value = max(_vars) + self.__increment_value
        else:
            message = "Invalid combination rule for NodeIncrementSetter"
            raise RuntimeError(message)

        if self.__maxchi is not None:
            node.value = min(node.value, self.__maxchi)
