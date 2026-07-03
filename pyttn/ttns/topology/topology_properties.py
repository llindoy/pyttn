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

from typing import Callable, Optional, Union
import numpy as np
from pyttn.ttnpp import ntree, ntreeNode, ntreeBuilder

from .bond_setter import BondDimensionSetter


def __get_value(chi, level):
    if isinstance(chi, (list, np.ndarray)):
        return chi[level]
    elif isinstance(chi, BondDimensionSetter):
        return 0
    elif callable(chi):
        return chi(level)
    else:
        return chi


def __setup_bond_properties_internal(
    node,
    chi,
    dims,
    level,
    chi_local_transform,
    node_list,
    nindex,
    update_leaves=True,
    update_internal_nodes=True,
):
    if node.is_leaf():
        if update_leaves:
            if isinstance(dims, (list, np.ndarray)):
                node.value = dims[node.value]
            elif isinstance(dims, int):
                node.value = dims
            else:
                raise RuntimeError("Invalid dims argument.")
    elif node.is_root() and update_internal_nodes:
        node.value = 1
    elif node.is_local_basis_transformation() and update_internal_nodes:
        chiv = __get_value(chi_local_transform, node.at(0).value)
        if chiv is None:
            if isinstance(dims, (list, np.ndarray)):
                node.value = dims[node.at(0).value]
            elif isinstance(dims, int):
                node.value = dims
        else:
            node.value = chiv
    elif update_internal_nodes:
        node.value = __get_value(chi, level)

    # append the current node to the node_list
    node_list.append(nindex)

    for i in range(node.size()):
        __setup_bond_properties_internal(
            node.at(i),
            chi,
            dims,
            level + 1,
            chi_local_transform,
            node_list,
            nindex + [i],
            update_leaves=update_leaves,
            update_internal_nodes=update_internal_nodes,
        )


def __update_interior_nodes(root, chi, node_list):
    for nind in reversed(node_list):
        node = root.at(nind)
        if (
            (not node.is_leaf())
            and (not node.is_root())
            and (not node.is_local_basis_transformation())
        ):
            chi(node)


def set_dims(root: Union[ntree, ntreeNode], dims: Union[int, list[int], np.ndarray]) -> None:
    """Set the the local hilbert space dimension in the topology tree (or in a subtree with root defined by the node root).

    :param root: The topology tree (or subtree) for which properties are to be set.

       * If root is a ntree we will edit the properties of all nodes in the entire tree.
       * If root is a ntreeNode we will edit the properties of all nodes in the subtree that root is the root of.

    :type root: ntree or ntreeNode
    :param dims: The local Hilbert space dimensions of each mode
    :type dims: int, list, np.ndarray
    """
    node_list = []
    if isinstance(root, ntree):
        __setup_bond_properties_internal(
            root(), None, dims, 0, None, node_list, [], update_internal_nodes=False
        )
    elif isinstance(root, ntreeNode):
        __setup_bond_properties_internal(
            root, None, dims, 0, None, node_list, [], update_internal_nodes=False
        )
    else:
        raise RuntimeError("Invalid root type for setup bond properties")


def set_bond_dimensions(
    root: Union[ntree, ntreeNode],
    chi: Union[int, Callable[[int], int], list[int], np.ndarray, BondDimensionSetter],
    chi_local_transform: Optional[
        Union[int, Callable[[int], int], list[int], np.ndarray]
    ] = None,
) -> None:
    """Set the the internal values stored in the topology tree (or in a subtree with root defined by the node root).
    This function will set the bond dimension and dimensionality of the local Hilbert space isometry.

    :param root: The topology tree (or subtree) for which properties are to be set.

       * If root is a ntree we will edit the properties of all nodes in the entire tree.
       * If root is a ntreeNode we will edit the properties of all nodes in the subtree that root is the root of.

    :type root: ntree or ntreeNode
    :param chi: The bond dimension information to be set for internal nodes of the tree.  The manner in whichthe bond dimensions are set depends on the type passed to chi.

       * If chi is an integer then all interior bond dimensions are set to this value
       * If chi is Callable then the interior bond dimension are set to chi(L) where L is the depth of the tree from the root
       * If chi is a list or np.ndarray then chi[L] where L is the depth of the tree from the root
       * Finally if chi is an BondDimensionSetter object then we iterate over the tree from leaves to root and set the value of the node using the BondDimensionSetter

    :type chi: int, Callable, list, np.ndarray, BondDimensionSetter
    :param chi_local_transform: The dimensionality of the local Hilbert space dimension, defaults to None

       * If chi_local_transform is an integer then all interior bond dimensions are set to this value
       * If chi_local_transform is Callable then the interior bond dimension are set to chi_local_transform(L) where L is the depth of the tree from the root
       * If chi_local_transform is a list or np.ndarray then chi_local_transform[L] where L is the depth of the tree from the root
       * If chi_local_transform is None then chi_local_transform will be set to the value of dim associated with the node

    :type chi_local_transform: int, Callable, list, np.ndarray, None, optional
    """
    node_list = []
    if isinstance(root, ntree):
        __setup_bond_properties_internal(
            root(),
            chi,
            None,
            0,
            chi_local_transform,
            node_list,
            [],
            update_leaves=False,
        )
    elif isinstance(root, ntreeNode):
        __setup_bond_properties_internal(
            root, chi, None, 0, chi_local_transform, node_list, [], update_leaves=False
        )
    else:
        raise RuntimeError("Invalid root type for setup bond properties")

    if isinstance(chi, BondDimensionSetter):
        # if chi is derived from a BondDimensionSetter object, then we want to iterate over the
        # tree in reverse order and update all of the interior nodes bond dimensions using this function
        if isinstance(root, ntree):
            __update_interior_nodes(root(), chi, node_list)
        elif isinstance(root, ntreeNode):
            __update_interior_nodes(root, chi, node_list)


def set_topology_properties(
    root: Union[ntree, ntreeNode],
    chi: Union[int, Callable[[int], int], list[int], np.ndarray, BondDimensionSetter],
    dims: Union[int, list[int], np.ndarray],
    chi_local_transform: Optional[
        Union[int, Callable[[int], int], list[int], np.ndarray]
    ] = None,
) -> None:
    """Set the values stored in a topology tree (or in a subtree with root defined by the node root).
    This function can set the bond dimension, local hilbert space dimension and dimensionality
    of the local Hilbert space isometry.

    :param root: The topology tree (or subtree) for which properties are to be set.

       * If root is a ntree we will edit the properties of all nodes in the entire tree.
       * If root is a ntreeNode we will edit the properties of all nodes in the subtree that root is the root of.

    :type root: ntree or ntreeNode
    :param chi: The bond dimension information to be set for internal nodes of the tree.  The manner in whichthe bond dimensions are set depends on the type passed to chi.

       * If chi is an integer then all interior bond dimensions are set to this value
       * If chi is Callable then the interior bond dimension are set to chi(L) where L is the depth of the tree from the root
       * If chi is a list or np.ndarray then chi[L] where L is the depth of the tree from the root
       * Finally if chi is an BondDimensionSetter object then we iterate over the tree from leaves to root and set the value of the node using the BondDimensionSetter

    :type chi: int, Callable, list, np.ndarray, BondDimensionSetter
    :param dims: The local Hilbert space dimensions of each mode
    :type dims: int, list, np.ndarray
    :param chi_local_transform: The dimensionality of the local Hilbert space dimension, defaults to None

       * If chi_local_transform is an integer then all interior bond dimensions are set to this value
       * If chi_local_transform is Callable then the interior bond dimension are set to chi_local_transform(L) where L is the depth of the tree from the root
       * If chi_local_transform is a list or np.ndarray then chi_local_transform[L] where L is the depth of the tree from the root
       * If chi_local_transform is None then chi_local_transform will be set to the value of dim associated with the node

    :type chi_local_transform: int, Callable, list, np.ndarray, None, optional
    """
    node_list = []
    if isinstance(root, ntree):
        __setup_bond_properties_internal(
            root(), chi, dims, 0, chi_local_transform, node_list, []
        )
    elif isinstance(root, ntreeNode):
        __setup_bond_properties_internal(
            root, chi, dims, 0, chi_local_transform, node_list, []
        )
    else:
        raise RuntimeError("Invalid root type for setup bond properties")

    if isinstance(chi, BondDimensionSetter):
        # if chi is derived from a BondDimensionSetter object, then we want to iterate over the
        # tree in reverse order and update all of the interior nodes bond dimensions using this function
        if isinstance(root, ntree):
            __update_interior_nodes(root(), chi, node_list)
        elif isinstance(root, ntreeNode):
            __update_interior_nodes(root, chi, node_list)


def build_bond_dimension_trees(tree: ntree, local_dims: list[int], bond_dims: dict) -> tuple[ntree, ntree]:
    """
    Construct minimum- and maximum-bond-dimension trees.

    The supplied topology tree is copied twice. Local basis
    transformation nodes are inserted automatically using
    ntreeBuilder.insert_basis_nodes. Interior nodes are then
    assigned the proposed minimum and maximum bond dimensions.

    :param tree: Input topology tree
    :type tree: ntree
    :param local_dims: Local Hilbert space dimensions indexed by physical-mode label
    :type local_dims: list[int]
    :param bond_dims: Output from propose_bond_dimensions
    :type bond_dims: dict

    :returns: (minimum_tree, maximum_tree)
    :rtype: tuple[ntree, ntree]
    """

    tree_min = ntree(tree)
    tree_max = ntree(tree)

    ntreeBuilder.insert_basis_nodes(tree_min)
    ntreeBuilder.insert_basis_nodes(tree_max)

    def populate(node_min, node_max):
        # Physical modes - set to local hilbert space dimension
        if node_min.is_leaf():
            phys_ind = node_min.value
            node_min.value = local_dims[phys_ind]
            node_max.value = local_dims[phys_ind]

        # Root - is trivial
        elif node_min.is_root():
            node_min.value = 1
            node_max.value = 1

        # Internal node - set to bond dimension guessed before
        else:
            key = tuple(node_min.index())

            if key not in bond_dims:
                raise ValueError(f"No bond dimension information for node {key}")

            node_min.value = bond_dims[key]["min_chi"]
            node_max.value = bond_dims[key]["max_chi"]

        for i in range(node_min.size()):
            populate(node_min.at(i), node_max.at(i))

    populate(tree_min(), tree_max())

    return tree_min, tree_max