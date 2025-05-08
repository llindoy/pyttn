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
import numpy as np
from scipy.cluster.hierarchy import linkage


def __condense_distance_matrix(M: np.ndarray) -> np.ndarray:
    # converts a dense matrix to a condensed distance matrix form
    N = M.shape[0]

    dist = np.zeros((N * (N - 1)) // 2)
    c = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist[c] = M[i, j]
            c += 1
    return dist


def __linkage_to_nxtree(Z: np.ndarray) -> tuple[nx.Graph, int]:
    # takes the linkage matrix form returned by scipy.cluster.hierarchy.linkage
    # and converts it to a networkx tree, with the leaf nodes labelled based on their
    # index and the interior nodes uniquely labeled in increasing order towards the root

    N = Z.shape[0] + 1

    edges = []
    root_index = 0
    # now iterate over links and construct an edge list
    for i in range(Z.shape[0]):
        Z0 = int(Z[i, 0])
        Z1 = int(Z[i, 1])
        edges.append((i + N, Z0))
        edges.append((i + N, Z1))
        root_index = i + N

    T = nx.Graph()
    for e in edges:
        T.add_edge(e[1], e[0])
        T.add_edge(e[0], e[1])

    return T, root_index


def __weight_to_distance(M, distance_metric, **kwargs):
    if distance_metric == "sub":
        return np.max(M) - M
    elif distance_metric == "exp":
        if "eps" in kwargs:
            eps = kwargs["eps"]
            return np.exp(-eps * M)
        else:
            return np.exp(-M)
    elif distance_metric == "gauss":
        if "eps" in kwargs:
            eps = kwargs["eps"]
            return np.exp(-eps * M * M)
        else:
            return np.exp(-M * M)
    elif distance_metric == "lorentz":
        if "eps" in kwargs:
            eps = kwargs["eps"]
            return eps / (eps * eps + M * M)
        else:
            return 1 / (1 + M * M)
    else:
        raise RuntimeError("Failed to recognised distance metric parameter.")


def generate_hierarchical_clustering_tree(
    M: np.ndarray, distance_metric: str = "gauss", **kwargs
) -> tuple[nx.Graph, int]:
    """Construct a networkx graph object from the maximum weight spanning tree of some matrix M.  This function
    can optionally insert logical nodes to prevent any node having a more children than max_nchild, and can be chosen
    so that any node is the root index of the tree.

    :param M: The weight matrix used to define a weighted graph of the nodes to be represented as a tree.
    :type M: np.ndarray
    :param distance_metric: An arguments specifying how to transform the weight matrix to a distance matrix (default: 'gauss')
    :type distance_metric: {'sub', 'exp', 'gauss', 'lorentz'}, optional

    :param **kwargs:  Additional keyword arguments that depend on the choice of distance metric.

        - For distance_metric = 'sub': This argument is ignored
        - For distance_metric={'exp', 'gauss', 'lorentz'}: an optional eps argument can be specified defining the width of the function.

    :return: A networkx graph containing the generated tree and the index of the root of the tree.
    :rtype: nx.Graph, int
    """
    dist = __condense_distance_matrix(M)
    dist = __weight_to_distance(dist, distance_metric, **kwargs)
    Z = linkage(dist, method="ward")
    return __linkage_to_nxtree(
        Z,
    )
