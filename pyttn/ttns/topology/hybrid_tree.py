# This files is part of the pyTTN package.
# (C) Copyright 2026 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np
import networkx as nx
from typing import Optional
from .spectral_tree import __spectral_split_indices
from .spanning_tree import distance_matrix_to_graph, __insert_physical_nodes

def __build_local_spectral_ttn(nodes, W, T, next_index, max_children):
    """
    Build a spectral TTN over nodes.
    Returns (root, next_index).
    """

    # leaf
    if len(nodes) == 1:
        return nodes[0], next_index

    # small set to star
    if len(nodes) <= max_children:
        root = next_index
        next_index += 1
        for n in nodes:
            T.add_edge(root, n)
        return root, next_index

    # otherwise split recursively
    A, B = __spectral_split_indices(nodes, W)

    root = next_index
    next_index += 1

    left, next_index = __build_local_spectral_ttn(A, W, T, next_index, max_children)
    right, next_index = __build_local_spectral_ttn(B, W, T, next_index, max_children)

    T.add_edge(root, left)
    T.add_edge(root, right)

    return root, next_index

def __restructure_hybrid(spanning_tree: nx.Graph,W: np.ndarray,root_index: int,max_children: Optional[int]) -> nx.Graph:
    """
    Hybrid MWST + spectral TTN refinement.
    """

    if max_children is None:
        return spanning_tree

    T_new = nx.Graph()
    next_index = max(spanning_tree.nodes) + 1

    # root the MWST
    T_dir = nx.bfs_tree(spanning_tree, root_index)

    # build child mapping
    children_map = {n: [] for n in T_dir.nodes}
    for u, v in T_dir.edges:
        children_map[u].append(v)

    for node in T_dir.nodes:
        children = children_map[node]

        if len(children) <= max_children:
            for c in children:
                T_new.add_edge(node, c)
            continue

        subtree_root, next_index = __build_local_spectral_ttn(
            children,
            W,
            T_new,
            next_index,
            max_children,
        )

        T_new.add_edge(node, subtree_root)

    return T_new