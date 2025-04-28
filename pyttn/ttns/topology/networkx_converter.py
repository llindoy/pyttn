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

from pyttn.ttnpp import ntree


def convert_nx_to_subtree(tree, root, root_ind=0):
    """A function for converting a networkx graph storing a tree structure into a subtree
    of an ntree object with root at node root.

    :param tree: The networkx graph object representing the topology tree.  This
    :type tree: nx.Graph
    :param root: The ntreeNode object that will be used as the root node to attach the current subtree to
    :type root: ntreeNode
    :param root_ind: The index in the tree object that should be connected to root, defaults to 0
    :type root_ind: int, optional

    :return: An array containing the index of the physical modes found at each leaf index
    :rtype: list
    """

    if not nx.is_tree(tree):
        raise RuntimeError(
            "Failed to convert networkx graph to subtree.  The input graph is not a tree."
        )

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

        if edge[0] not in edge_counter.keys():
            edge_counter[edge[0]] = 0
        else:
            edge_counter[edge[0]] += 1

        node_dict[edge[1]] = node_dict[edge[0]] + [edge_counter[edge[0]]]
        root.at(node_dict[edge[0]]).insert(edge[1])

    subtree_root = root.at([root_skip])
    return [
        subtree_root.at(leaf_inds).value
        for leaf_inds in subtree_root.leaf_indices()
    ]


def convert_nx_to_tree(tree, root_ind=0):
    """A function for constructing an ntree object from a networkx object.

    :param tree: The networkx graph object representing the topology tree.  This
    :type tree: nx.Graph
    :param root_ind: The index in the tree object that should be connected to root, defaults to 0
    :type root_ind: int, optional
    :return: An array containing the index of the physical modes found at each leaf index
    :rtype: list
    """

    if not nx.is_tree(tree):
        raise RuntimeError(
            "Failed to convert networkx graph to subtree.  The input graph is not a tree."
        )

    res = ntree(str(tree.number_of_nodes() - 1))

    node_dict = {root_ind: []}
    edge_counter = {}

    for edge in nx.dfs_edges(tree, source=root_ind):
        if edge[0] not in edge_counter.keys():
            edge_counter[edge[0]] = 0
        else:
            edge_counter[edge[0]] += 1

        node_dict[edge[1]] = node_dict[edge[0]] + [edge_counter[edge[0]]]
        res().at(node_dict[edge[0]]).insert(edge[1])

    return res, [leaf.value for leaf in res.leaves()]
