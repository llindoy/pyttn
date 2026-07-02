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

from typing import Optional

import networkx as nx
import numpy as np

from .spectral_tree import __spectral_split_indices

def distance_matrix_to_graph(M: np.ndarray) -> nx.Graph:
    return nx.from_numpy_array(np.abs(M - np.diag(np.diag(M))))


def __insert_physical_nodes(spanning_tree, N, root_ind, max_children):
    # a function for taking a networkx tree generated from a max weight spanning tree of a graph
    # and inserts children nodes below each of the leaf nodes representing the physical tree nodes
    # additionally this shifts all indices stored in the tree so that they each have a unique label
    # and the new children nodes have the label that their parent used to have
    nindex = spanning_tree.number_of_nodes()

    # iterate over each node in the tree and determine whether we need to insert the child node
    # this is done by iterating over the tree in a depth first search order and determining if it
    # is a leaf node in the tree.  If it is we don't need to do anything, if it is not, then we will
    # insert a node beneath the present node and update its index

    # get a list of all nodes in the tree
    edges = sorted(
        nx.dfs_edges(spanning_tree, source=root_ind),
        key=lambda x: np.abs(x[0] - root_ind),
    )

    # determine the number of children each node has
    nchildren = [0 for x in range(N)]
    for e in edges:
        if e[0] < N:
            nchildren[e[0]] += 1

    # and the total number of nodes that aren't leaf nodes

    mapping = {}
    counter = 0
    for i in range(nindex):
        if i < N:
            if nchildren[i] == 0:
                mapping[i] = i
            else:
                mapping[i] = nindex + counter
                counter = counter + 1
        else:
            mapping[i] = i

    # iterate over the tree and add nodes to any n
    nx.relabel_nodes(spanning_tree, mapping=mapping, copy=False)

    for i in range(N):
        if nchildren[i] > 0:
            spanning_tree.add_edge(mapping[i], i)

    return spanning_tree, mapping[root_ind]


def __build_local_spectral_ttn(nodes, W, T, next_index, max_children):
    """
    Build a spectral TTN over nodes with strict degree constraint.
    Returns (root, next_index).
    """
    if max_children == 1:
        # order nodes using spectral ordering 
        if len(nodes) <= 1:
            return nodes[0], next_index

        # simple fallback: keep order (or sort)
        ordered = list(nodes)

        root = ordered[0]
        current = root

        for n in ordered[1:]:
            aux = next_index
            next_index += 1

            T.add_edge(current, aux)
            T.add_edge(aux, n)

            current = aux
        return root, next_index

    # leaf
    if len(nodes) == 1:
        return nodes[0], next_index

    # already small enough so make star
    if len(nodes) <= max_children:
        root = next_index
        next_index += 1
        for n in nodes:
            T.add_edge(root, n)
        return root, next_index

    groups = [nodes]

    while any(len(g) > 1 for g in groups) and len(groups) < max_children:
        # split the largest group
        largest = max(groups, key=len)
        groups.remove(largest)

        A, B = __spectral_split_indices(largest, W)
        groups.append(A)
        groups.append(B)

    def merge_groups(groups):
        # merge two smallest groups
        groups = sorted(groups, key=len)
        merged = groups[0] + groups[1]
        return groups[2:] + [merged]

    while len(groups) > max_children:
        groups = merge_groups(groups)

    root = next_index
    next_index += 1

    children_roots = []

    for g in groups:
        child, next_index = __build_local_spectral_ttn(g, W, T, next_index, max_children)
        children_roots.append(child)

    for child in children_roots:
        T.add_edge(root, child)

    assert len(children_roots) <= max_children, (f"max_children violated at node {root}: {len(children_roots)}")

    return root, next_index


def __restructure_hybrid(spanning_tree: nx.Graph,W: np.ndarray,root_index: int,max_children: Optional[int]) -> nx.Graph:
    if max_children is None:
        return spanning_tree

    T_new = nx.Graph()
    next_index = max(spanning_tree.nodes) + 1

    T_dir = nx.bfs_tree(spanning_tree, root_index)

    # build child mapping
    children_map = {n: [] for n in T_dir.nodes}
    for u, v in T_dir.edges:
        children_map[u].append(v)

    for node in T_dir.nodes:
        children = children_map[node]

        if len(children) < max_children:
            for c in children:
                T_new.add_edge(node, c)
            continue

        subtree_root, next_index = __build_local_spectral_ttn(children, W, T_new, next_index, max_children,)

        T_new.add_edge(node, subtree_root)

    return T_new

def generate_spanning_tree(
    M: np.ndarray,
    max_children: Optional[int] = None,
    max_leaf_children: Optional[int] = None,
    root_index: int = 0,
) -> tuple[nx.Graph, int]:
    """Construct a networkx graph object from the maximum weight spanning tree of some matrix M.  This function
    can optionally insert logical nodes to prevent any node having a more children than max_nchild, and can be chosen
    so that any node is the root index of the tree.

    :param M: The "distance" matrix used to define a weighted graph of the nodes to be represented as a tree
    :type M: np.ndarray
    :param max_children: The maximum allowed number of children for any node.  If this is none, the number of children will not be limited, defaults to None
    :type max_children: int, None, optional
    :param max_leaf_children: The maximum allowed number of leaf node children.  If this is none, then we don't treat leaf and internal node children separately, defaults to None
    :type max_leaf_children: int, None, optional
    :param root_index: The index of the nodes that will be set as the root of this tree, defaults to 0
    :type root_index: int, optional
    :return: A networkx graph containing the generated tree and the index of the root of the tree.
    :rtype: nx.Graph, int
    """
    if root_index > M.shape[0] or root_index < 0:
        raise RuntimeError(
            "Failed to generate spanning tree from weight matrix.  User specified root index out of bounds."
        )
    G = distance_matrix_to_graph(M)

    spanning_tree = nx.maximum_spanning_tree(G)

    spanning_tree = __restructure_hybrid(
        spanning_tree,
        M,
        root_index,
        max_children,
    )



    return __insert_physical_nodes(spanning_tree, M.shape[0], root_index, max_children)
