"""Convert networkx tree objects to python trees."""

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

import networkx as nx

from pyttn.ttnpp import ntree, ntreeNode


def convert_nx_to_subtree(
    tree: nx.Graph, root: ntreeNode, root_ind: int = 0,
) -> list[int]:
    """Convert a networkx tree into a subtree of an ntree object with root at node root.

    :param tree: The networkx graph object representing the topology tree.  This
    :type tree: nx.Graph
    :param root: The ntreeNode object that will be used as the root node to attach the
         current subtree to
    :type root: ntreeNode
    :param root_ind: The index in the tree object that should be connected to root,
         defaults to 0
    :type root_ind: int, optional
    :return: An array containing the index of the physical modes found at each
         leaf index
    :rtype: list[int]
    """
    if not nx.is_tree(tree):
        message = "Failed to convert networkx graph to subtree. \
            The input graph is not a tree."
        raise RuntimeError(message)

    root_skip = root.size()
    node_dict = {root_ind: [root_skip]}
    edge_counter = {}

    node_inserted = False
    for edge in nx.dfs_edges(tree, source=root_ind):
        # if this is the first edge in the dfs list then we need to insert both nodes
        # so first insert left most node
        if not node_inserted:
            root.at([]).insert(edge[0])
            node_inserted = True

        if edge[0] not in edge_counter:
            edge_counter[edge[0]] = 0
        else:
            edge_counter[edge[0]] += 1

        node_dict[edge[1]] = node_dict[edge[0]] + [edge_counter[edge[0]]]
        root.at(node_dict[edge[0]]).insert(edge[1])

    subtree_root = root.at([root_skip])

    leaf_labels = [subtree_root.at(leaf_inds).value for leaf_inds in subtree_root.leaf_indices()
                   ]
    leaf_indices = [0 for _ in leaf_labels]
    for i in range(len(leaf_labels)):
        leaf_indices[leaf_labels[i]] = i

    return leaf_indices


def convert_nx_to_tree(tree: nx.Graph, root_ind: int = 0) -> tuple[ntree, list[int]]:
    """A function for constructing an ntree object from a networkx object.

    :param tree: The networkx graph object representing the topology tree.  This
    :type tree: nx.Graph
    :param root_ind: The index in the tree object that should be connected to root,
         defaults to 0
    :type root_ind: int, optional
    :return: An array containing the index of the physical modes found at
        each leaf index
    :rtype: list[int]
    """
    if not nx.is_tree(tree):
        message="Failed to convert networkx graph to subtree. \
            The input graph is not a tree."
        raise RuntimeError(message)

    edges = list(nx.dfs_edges(tree, source=root_ind))
    if len(edges) == 0:
        return ntree(), []

    res = ntree(str(edges[0][0]))

    node_dict = {root_ind: []}
    edge_counter = {}

    for edge in nx.dfs_edges(tree, source=root_ind):
        if edge[0] not in edge_counter:
            edge_counter[edge[0]] = 0
        else:
            edge_counter[edge[0]] += 1

        node_dict[edge[1]] = node_dict[edge[0]] + [edge_counter[edge[0]]]
        res().at(node_dict[edge[0]]).insert(edge[1])


    # construct an array where each element corresponds to a leaf of the tree and
    # stores the physical mode it corresponds to.
    leaf_labels = [leaf.value for leaf in res.leaves()]

    # For the purpose of each calculation we need to invert this array and for each
    # physical mode work out the tree node it acts on.
    leaf_indices = [0 for _ in res.leaves()]
    for i in range(len(leaf_labels)):
        leaf_indices[leaf_labels[i]] = i

    return res, leaf_indices
