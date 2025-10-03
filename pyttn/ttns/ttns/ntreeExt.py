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

from pyttn.ttnpp import ntree as _ntree
from pyttn.ttnpp import ntreeNode as _ntreeNode
from pyttn.ttnpp import ntreeBuilder as _ntreeBuilder

from abc import ABCMeta, abstractmethod
from typing import Callable, Iterator, Optional, Union


class ntreeNode(metaclass=ABCMeta):
    """A class for handling nodes in an ntree object"""

    def __new__(cls) -> "ntreeNode":
        """Constructs a new ntreeNode object

        :return: The ntreeNode object
        :rtype: ntreeNode
        """
        return _ntreeNode()

    @abstractmethod
    def level(self) -> int:
        """Returns the level of the node in the tree.

        :return: The level of the node in the tree.  root = 0, child of root = 1, etc.
        :rtype: int
        """
        pass

    @abstractmethod
    def nleaves(self) -> int:
        """Returns the number of leaf nodes below this node

        :return: The number of leaves in the subtree that this node is the root of
        :rtype: int
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Returns the number of children of the node

        :return: The number of children of this node
        :rtype: int
        """
        pass

    @abstractmethod
    def subtree_size(self) -> int:
        """Returns the number of nodes in the subtree that this node is the root of

        :return: The number of nodes in the subtree that this node is the root of
        :rtype: int
        """
        pass

    @abstractmethod
    def empty(self) -> bool:
        """Returns whether or not the node has any children.

        :return: Whether or not the node has any children.
        :rtype: int
        """
        pass

    @abstractmethod
    def is_root(self) -> bool:
        """Returns whether or not the node is the root node.

        :return: Whether or not the node is the root node.
        :rtype: int
        """
        pass

    @abstractmethod
    def is_leaf(self) -> bool:
        """Returns whether or not the node is a leaf node.

        :return: Whether or not the node is a leaf node.
        :rtype: int
        """
        pass

    @abstractmethod
    def is_local_basis_transformation(self) -> bool:
        """Return whether or not a node is a local basis transformation node, that is a node that is the parent of a leaf and only has one child.

        :return: Whether or not the node is a local basis transformation node.
        :rtype: int
        """
        pass

    @abstractmethod
    def leaf_indices(self) -> list[list[int]]:
        """Returns the indices associated with the leaves below this node

        :return: The indices associated with the leaves below this node
        :rtype: list[list[int]]
        """
        pass

    @abstractmethod
    def index(self) -> list[int]:
        """Returns the index associated with this node

        :return: The index associated with this node
        :rtype: list[int]
        """
        pass

    @abstractmethod
    def depth(self) -> int:
        """Returns the depth of this node (0 for root, +1 for each layer away from root)

        :return: The depth of the node
        :rtype: int
        """
        pass

    @abstractmethod
    def parent(self) -> "ntreeNode":
        """Returns the parent node of this node.  Throws a C++ exception if this node is the root node

        :return: The parent of the current node
        :rtype: ntreeNode
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear any internal storage of the node"""
        pass

    @property
    @abstractmethod
    def value(self) -> int:
        """Retun the value stored in the node

        :return: The value stored in the node
        :rtype: int
        """
        pass

    @property
    @abstractmethod
    def data(self) -> int:
        """Retun the value stored in the node

        :return: The value stored in the node
        :rtype: int
        """
        pass

    @abstractmethod
    def at(self, i: Union[int, list[int]]) -> "ntreeNode":
        """Access a child node of the current node.

        This functions supports two ways of defining the index either:

            * an integer defining the index of the child of the current node
            * or a list of integers defining the traversal path starting at the current node.

        :param i: The index of the node
        :type i: Union[int, list[int]]
        :return: The node at the requested index
        :rtype: ntreeNode
        """
        pass

    @abstractmethod
    def __getitem__(self, i: Union[int, list[int]]) -> "ntreeNode":
        """Access a child node of the current node.

        This functions supports two ways of defining the index either:

            * an integer defining the index of the child of the current node
            * or a list of integers defining the traversal path starting at the current node.

        :param i: The index of the node
        :type i: Union[int, list[int]]
        :return: The node at the requested index
        :rtype: ntreeNode
        """
        pass

    @abstractmethod
    def front(self) -> "ntreeNode":
        """Access the first child of the current node
        :return: The first child of the current node
        :rtype: ntreeNode
        """
        pass

    @abstractmethod
    def back(self) -> "ntreeNode":
        """Access the last child of the current node
        :return: The last child of the current node
        :rtype: ntreeNode
        """
        pass

    @abstractmethod
    def insert(self, x: Union[int, "ntreeNode"]):
        """Insert a new node as the last child of the current node

        :param x: Either a value used to define a new node or a node to be inserted as a child of the current node
        :type x: ntreeNode
        """
        pass

    @abstractmethod
    def insert_front(self, x: Union[int, "ntreeNode"]):
        """Insert a new node as the first child of the current node

        :param x: Either a value used to define a new node or a node to be inserted as a child of the current node
        :type x: ntreeNode
        """
        pass

    @abstractmethod
    def remove(self, i: int):
        """Remove the ith child of the node from the tree

        :param i: The index of the child to be removed
        :type i: int
        """
        pass


ntreeNode.register(_ntreeNode)


class ntree(metaclass=ABCMeta):
    """A class for handling a generic tree structure that stores an integer at each node.

    Within pyTTN this class is used to represent the topology of the tree tensor network, with the integers stored at a node corresponding
    to the number of single-particle functions (basis functions) associated with this node.  Alternatively, this
    can be viewed as the bond-dimension connecting the node with its parent.  In the current version we require the root node
    to have a value of 1, which corresponds to a topology representing a single state.
    """

    def __new__(cls, *args) -> "ntree":
        """Constructs a new ntree object

        :param `*args`: Variable length list of arguments. This function can handle the following lists of arguments

            - Default construct ttn object
            - ntree (:class:`ntree`) - Copy construct the ntree object
            - str (:class:`str`) - Construct  an ntree from a string representation of a tree.

        For the string based construction we use a set of nested parentheses to represent a tree e.g.:
        (1(2(3)(4))(5(6)(7))

        corresponds to the tree

        .. code-block:: text

                  1
                 x_x
                x___x
               2_____5
              x_x___x_x
             x___x_x___x
            3____4_6____7

        :return: The ntree object
        :rtype: ntree
        """
        return _ntree(*args)

    @abstractmethod
    def assign(self, tree: "ntree"):
        """Assign the value of this ntree using another ntree

        :param tree: Input ntree to construct a copy of
        :type tree: ntree
        """
        pass

    @abstractmethod
    def empty(self) -> bool:
        """Whether or not the tree has any elements

        :return: Returns whether or not the ntree object is empty.  That is it contains no nodes.
        :rtype: bool
        """
        pass

    @abstractmethod
    def nleaves(self) -> int:
        """Return the number of leaf nodes in the ntree

        :return: The number of leaf nodes in the ntree
        :rtype: int
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the total number of nodes in the ntree

        :return: The total number of nodes in the ntree
        :rtype: int
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of nodes in the ntree

        :return: The total number of nodes in the ntree
        :rtype: int
        """
        pass

    @abstractmethod
    def load_from_string(self, tree: str):
        """Load an ntree object from a string representation of a tree.  Here we use a set of nested parentheses to represent
        a tree e.g.:

        (1(2(3)(4))(5(6)(7))

        corresponds to the tree

        .. code-block:: text

                  1
                 x_x
                x___x
               2_____5
              x_x___x_x
             x___x_x___x
            3____4_6____7
        

        :param tree: A string representation of a tree
        :type tree: str

        """
        pass

    @abstractmethod
    def insert(self, val: int):
        """Insert a new root node into the tree.  If the tree already contains a root node this will throw an exception.

        :param val: The data to store at the root node
        :type val: int
        """
        pass

    @abstractmethod
    def insert_at(self, inds: list[int], val: int):
        """Insert an element as a child of a specific node in the tree.  Here the position is defined by a list of numbers that define which child to traverse down at each node of the tree.
        If the list is empty this attempts to insert a new root.

        :param inds: The index of the node where we aim to insert a child.
        :type inds: list[int]
        :param val: The data to store at the root node
        :type val: int
        """
        pass

    @abstractmethod
    def leaf_indices(self) -> list[list[int]]:
        """Returns the indices associated with the leaves of the tree

        :return: The indices associated with the leaves of the tree
        :rtype: list[list[int]]
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear any internal storage of the ntree"""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        """An iterator over the nodes in the tree
        :return: An iterator for traversing the tree in depth first search order
        :rtype: Iterator

        """
        pass

    @abstractmethod
    def dfs(self) -> Iterator:
        """A depth first search iterator over the nodes in the tree

        :return: An iterator for traversing the tree in depth first search order
        :rtype: Iterator
        """
        pass

    @abstractmethod
    def bfs(self) -> Iterator:
        """A breadth first search iterator over the nodes in the tree

        :return: An iterator for traversing the tree in breadth first search order
        :rtype: Iterator
        """
        pass

    @abstractmethod
    def post_order_dfs(self) -> Iterator:
        """A post order depth first search iterator over the nodes in the tree

        :return: An iterator for traversing the tree in post order depth first search order
        :rtype: Iterator
        """
        pass

    @abstractmethod
    def euler_tour(self) -> Iterator:
        """A euler tour traversal of the tree

        :return: An iterator for performing an euler tour over the tree
        :rtype: Iterator
        """
        pass

    @abstractmethod
    def leaves(self) -> Iterator:
        """An iterator over the leaves of the tree, traversing the leaves in the order given by a depth first search traversal

        :return: An iterator over the leaves of the tree
        :rtype: Iterator
        """
        pass

    @abstractmethod
    def __call__(self) -> ntreeNode:
        """Returns the root node of the tree

        :return: The root node of the tree
        :rtype: ntreeNode
        """
        pass

    @abstractmethod
    def root(self) -> ntreeNode:
        """Returns the root node of the tree

        :return: The root node of the tree
        :rtype: ntreeNode
        """
        pass

    @abstractmethod
    def __getitem__(self: Union[int, list[int]]) -> ntreeNode:
        """Access the node at a given index from the tree

        This functions supports two ways of defining the index either:

            * an integer defining the index of the child of the root node
            * or a list of integers defining the traversal path starting at the current node.

        :param i: The index of the node
        :type i: Union[int, list[int]]
        :return: The node at the requested index
        :rtype: ntreeNode
        """
        pass

    @abstractmethod
    def at(self: Union[int, list[int]]) -> ntreeNode:
        """Access the node at a given index from the tree

        This functions supports two ways of defining the index either:

            * an integer defining the index of the child of the root node
            * or a list of integers defining the traversal path starting at the current node.

        :param i: The index of the node
        :type i: Union[int, list[int]]
        :return: The node at the requested index
        :rtype: ntreeNode
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Returns a string representation of the in the nested parentheses format.

        :return: A string representation of the tree
        :rtype: str
        """
        pass

    @abstractmethod
    def as_json(self) -> str:
        """Returns a json string representing the tree

        :return: A json string representing the tree
        :rtype: str
        """
        pass


ntree.register(_ntree)


class ntreeBuilder:
    """A class providing several helper functions for setting up ntree objects and validating their structures"""

    @staticmethod
    def sanitise(tree: ntree, remove_bond_matrices: Optional[bool] = True):
        """Sanitise an ntree to ensure that it corresponds to a valid structure for TTN and if specified does not have any bond matrices.

        :param tree: The tree to sanitise.  This tree is overwritten by the sanitised structure.
        :type tree: ntree
        :param remove_bond_matrice: A flag as to whether or not to remove bond matrices., defaults to True
        :type remove_bond_matrice: Optional[bool], optional
        """
        _ntreeBuilder.sanitise(tree, remove_bond_matrices=remove_bond_matrices)

    @staticmethod
    def insert_basis_nodes(tree: ntree):
        """Insert local basis nodes within the ntree object to ensure the structure is in the format expected by the TTN class.

        :param tree: The tree to edit.  This tree is overwritten in this process.
        :type tree: ntree
        """
        _ntreeBuilder.insert_basis_nodes(tree)

    @staticmethod
    def sanitise_bond_dimensions(tree: ntree):
        """Alter the bond dimensions within the tree such that we do not have any bonds where we have a larger dimension than the number of states.

        :param tree: The tree to edit.  This tree is overwritten in this process.
        :type tree: ntree
        """
        _ntreeBuilder.sanitise_bond_dimensions(tree)

    @staticmethod
    def collapse_bond_matrices(tree: ntree):
        """Edit the tree collapsing all bond matrices into adjacent tensors.

        :param tree: The tree to edit.  This tree is overwritten in this process.
        :type tree: ntree
        """
        _ntreeBuilder.collapse_bond_matrices(tree)

    @staticmethod
    def mlmctdh_tree(
        dims: list[int],
        degree: int,
        chi: Union[int, Callable[[int], int]],
        lhd: Optional[Union[int, list[int]]] = None,
        include_local_basis_transformation: Optional[bool] = True,
    ) -> ntree:
        """Construct a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d,
        using a function that depends on the distance from the root to specify the internal bond dimensions.

        :param dims: The local Hilbert space dimensions of each leaf of the tree
        :type dims: list[int]
        :param degree: The degree of the tree structure.  2 corresponds to a balanced binary tree.
        :type degree: int
        :param chi: An integer or function specifying the bond dimension as a function of depth.
        :type chi: Union[int, Callable[[int], int]]
        :param lhd: An integer or list defining the local hilbert space dimension for transformation nodes
        :type lhd: Optional[Union[int, list[int]]]
        :param include_local_basis_transformation: A flag as to or not to include a set of bond matrices.
            at the leaves of the tree defining a local basis transformation. This argument is only used if lhd is None defaults to True
        :type include_local_basis_transformation: Optional[bool]
        :return: A balanced tree with the structure determined by the users inputs
        :rtype: ntree
        """
        if isinstance(lhd, (int, list)):
            return _ntreeBuilder.mlmctdh_tree(dims, degree, chi, lhd)
        else:
            if lhd is None:
                return _ntreeBuilder.mlmctdh_tree(
                    dims,
                    degree,
                    chi,
                    include_local_basis_transformation=include_local_basis_transformation,
                )
            else:
                raise RuntimeError("Invalid arguments for ntreeBuilder.mlmtcdh_tree")
            
    @staticmethod
    def mlmctdh_subtree(
        node: ntreeNode,
        dims: list[int],
        degree: int,
        chi: Union[int, Callable[[int], int]],
        lhd: Optional[Union[int, list[int]]] = None,
        include_local_basis_transformation: Optional[bool] = True,
    ) -> ntree:
        """Append a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d,
        using a function that depends on the distance from the root to specify the internal bond dimensions.
        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes.

        :param node: The node to append the subtree too
        :type node: ntreeNode
        :param dims: The local Hilbert space dimensions of each leaf of the tree
        :type dims: list[int]
        :param degree: The degree of the tree structure.  2 corresponds to a balanced binary tree.
        :type degree: int
        :param chi: An integer or function specifying the bond dimension as a function of depth.
        :type chi: Union[int, Callable[[int], int]]
        :param lhd: An integer or list defining the local hilbert space dimension for transformation nodes
        :type lhd: Optional[Union[int, list[int]]]
        :param include_local_basis_transformation: A flag as to or not to include a set of bond matrices.
            at the leaves of the tree defining a local basis transformation. This argument is only used if lhd is None defaults to True
        :type include_local_basis_transformation: Optional[bool]
        """
        if isinstance(lhd, (int, list)):
            _ntreeBuilder.mlmctdh_subtree(node, dims, degree, chi, lhd)
        else:
            if lhd is None:
                _ntreeBuilder.mlmctdh_subtree(
                    node,
                    dims,
                    degree,
                    chi,
                    include_local_basis_transformation=include_local_basis_transformation,
                )
            else:
                raise RuntimeError("Invalid arguments for ntreeBuilder.mlmtcdh_subtree")

    @staticmethod
    def mps_tree(
        dims: list[int],
        chi: Union[int, Callable[[int], int]],
        lhd: Optional[Union[int, list[int], Callable[[int], int]]] = None,
    ) -> ntree:
        """Construct a new mps ntree for representing a tensor with dimensions specified in a list using a
        function that depends on the distance from the root to specify the internal bond dimensions. Here
        the tree also automatically includes local optimised bases with the number of states being set by
        a second function.

        :param dims: The local Hilbert space dimensions of each leaf of the tree
        :type dims: list[int]
        :param chi: An integer or function specifying the bond dimension as a function of depth.
        :type chi: Union[int, Callable[[int], int]]
        :param lhd: An integer or list defining the local hilbert space dimension for transformation nodes
        :type lhd: Optional[Union[int, list[int], Callable[[int], int]]]
        :return: A MPS tree with the structure determined by the users inputs
        :rtype: ntree
        """
        if isinstance(lhd, list):
            return _ntreeBuilder.mps_tree(dims, chi, lhd)
        else:
            if lhd is None:
                return _ntreeBuilder.mps_tree(dims, chi)
            else:
                return _ntreeBuilder.mps_tree(dims, chi, lhd)

    @staticmethod
    def mps_subtree(
        node: ntreeNode,
        dims: list[int],
        chi: Union[int, Callable[[int], int]],
        lhd=None,
    ) -> ntree:
        """Append a new balanced ntree for representing a tensor with dimensions specified in a list, of degree d,
        using a function that depends on the distance from the root to specify the internal bond dimensions.
        Additionally, this function inserts a local basis transformation with a user specified dimension to each of the leaf nodes.

        :param node: The node to append the subtree too
        :type node: ntreeNode
        :param dims: The local Hilbert space dimensions of each leaf of the tree
        :type dims: list[int]
        :param chi: An integer or function specifying the bond dimension as a function of depth.
        :type chi: Union[int, Callable[[int], int]]
        :param lhd: An integer or list defining the local hilbert space dimension for transformation nodes
        :type lhd: Optional[Union[int, list[int]]]
        """
        if isinstance(lhd, list):
            _ntreeBuilder.mps_subtree(node, dims, chi, lhd)
        else:
            if lhd is None:
                _ntreeBuilder.mps_subtree(node, dims, chi)
            else:
                _ntreeBuilder.mps_subtree(node, dims, chi, lhd)
