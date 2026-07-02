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

from math import ceil
from typing import Any, Generator, Optional

import networkx as nx
import numpy as np

from scipy.sparse.linalg import eigsh

def __spectral_split_indices(indices, W):
    """
    Split a set of indices into two groups using Fiedler vector.
    """

    if len(indices) <= 2:
        return indices[:1], indices[1:]

    subW = W[np.ix_(indices, indices)]

    D = np.diag(subW.sum(axis=1))
    L = D - subW

    vals, vecs = eigsh(L, k=2, which="SM")
    fiedler = vecs[:, 1]

    A = [indices[i] for i in range(len(indices)) if fiedler[i] >= 0]
    B = [indices[i] for i in range(len(indices)) if fiedler[i] < 0]

    # fallback if degenerate
    if len(A) == 0 or len(B) == 0:
        half = len(indices) // 2
        A = indices[:half]
        B = indices[half:]

    return A, B

def __build_spectral_ttn(T,nodes,W,next_index,max_children):
    """
    Recursively build TTN using spectral partitioning.
    Returns (root_node, next_index)
    """

    # leaf
    if len(nodes) == 1:
        return nodes[0], next_index

    # enforce branching factor
    if max_children is None or max_children == 2:
        # binary split
        A, B = __spectral_split_indices(nodes, W)

        root = next_index
        next_index += 1

        left, next_index = __build_spectral_ttn(
            T, A, W, next_index, max_children
        )
        right, next_index = __build_spectral_ttn(
            T, B, W, next_index, max_children
        )

        T.add_edge(root, left)
        T.add_edge(root, right)

        return root, next_index

    # K-ary branching
    # recursively split until we have ≤ max_children groups
    groups = [nodes]

    while len(groups) < max_children:
        largest = max(groups, key=len)
        groups.remove(largest)

        if len(largest) <= 1:
            groups.append(largest)
            break

        A, B = __spectral_split_indices(largest, W)
        groups.append(A)
        groups.append(B)

    root = next_index
    next_index += 1

    for g in groups:
        child, next_index = __build_spectral_ttn(T, g, W, next_index, max_children)
        T.add_edge(root, child)

    return root, next_index


def generate_spectral_tree(
    M: np.ndarray,
    max_children: Optional[int] = 2,
    root_index: int = 0,
) -> tuple[nx.Graph, int]:
    """
    Construct a tree using recursive spectral partitioning.

    :param M: Weight matrix
    :param max_children: Maximum allowed children per node
    :param root_index: Physical root node (optional preference)

    :return: (tree, root index)
    """

    N = M.shape[0]

    # build tree
    T = nx.Graph()
    next_index = N

    root, next_index = __build_spectral_ttn(
        T,
        list(range(N)),
        M,
        next_index,
        max_children
    )

    return T, root
