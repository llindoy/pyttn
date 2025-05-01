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
import numpy as np


class BondDimensionSetter(metaclass=abc.ABCMeta):
    """Abstract base class for objects that set the bond dimension variables
    stored in an ntree object dependent on the values in the children of the node.
    """
    @abc.abstractmethod
    def __call__(self, node):
        """The default call function 

        :param node: The node which will have its bond dimension set
        :type node: ntreeNode
        """
        pass


class NodeSumSetter(BondDimensionSetter):
    """An implementation of the BondDimensionSetter abstract base class that set a given node
    value based on the values of the nodes below the current node
    """

    def __call__(self, node):
        """Set the value of the bond dimension variable in the current node to the sum 
        of the bond dimensions associated with the child nodes

        :param node: The node which will have its bond dimension set
        :type node: ntreeNode
        """
        ind = 0
        for i in range(node.size()):
            ind += node.at(i).value
        node.value = ind


class NodeIncrementSetter(BondDimensionSetter):
    def __init__(self, increment_value, combination="mean"):
        """An implementation of the BondDimensionSetter abstract base class that set a given node
        as an incremented value of a function of the children of the nodes

        :param increment_value: The value to increment the node value by
        :type increment_value: int
        :param combination: How to combine the values of the child nodes to get the value to increment by, defaults to "mean"
        :type combination: {"mean", "min", "max"}, optional
        :raises RuntimeError: Raises an error if the combination variable is not in the allowed set of values
        """
        self.__increment_value = increment_value
        if combination not in ["mean", "min", "max"]:
            raise RuntimeError(
                "Invalid combination rule for NodeIncrementSetter")
        self.__child_combination = combination

    def __call__(self, node):
        """Set the value of the bond dimension variable in the current node to be
        bond_dim = func(child_bond_dims) + inc

        :param node: The node which will have its bond dimension set
        :type node: ntreeNode
        """
        vars = []
        for i in range(node.size()):
            vars.append(node.at(i).value)
        if self.__child_combination == "mean":
            node.value = int(np.mean(vars)) + self.__increment_value
        elif self.__child_combination == "min":
            node.value = min(vars) + self.__increment_value
        elif self.__child_combination == "max":
            node.value = max(vars) + self.__increment_value
        else:
            raise RuntimeError(
                "Invalid combination rule for NodeIncrementSetter")
