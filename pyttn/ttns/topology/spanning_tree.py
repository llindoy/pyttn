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


def distance_matrix_to_graph(M):
    return nx.from_numpy_array(np.abs(M - np.diag(np.diag(M))))


def __insert_physical_nodes(spanning_tree, N, root_ind):
    # a function for taking a networkx tree generated from a max weight spanning tree of a graph
    # and inserts children nodes below each of the leaf nodes representing the physical tree nodes
    # additionally this shifts all indices stored in the tree so that they each have a unique label
    # and the new children nodes have the label that their parent used to have
    nindex = spanning_tree.number_of_nodes()

    mapping = {}
    for i in range(nindex):
        mapping[i] = N + i

    # iterate over the tree and add nodes to any n
    nx.relabel_nodes(spanning_tree, mapping=mapping, copy=False)

    for i in range(N):
        spanning_tree.add_edge(N + i, i)

    return spanning_tree, N + root_ind


def __split_high_degree_nodes(spanning_tree, N, root_index, max_nchild=None):
    if max_nchild is None:
        # if max degree has not been specified we just return the current tree
        return spanning_tree
    else:
        # in this case iterate through the tree determine if any nodes are too large and if they are we partition
        # the node into sets of the correct size.  To do this we get a DFS edges list and if there are any instances
        # of nodes with more than max_nchild children we insert sufficiently many logical nodes so that we have the
        # correct degree of connectivity
        edges = sorted(
            [edge for edge in nx.dfs_edges(spanning_tree, source=root_index)],
            key=lambda x: np.abs(x[0] - root_index),
        )

        nchildren = {}
        nodes_to_split = {}

        curr_node = None
        sind = 0

        # iterate over the edges getting the number of children associated with each node and the location of the
        # first edge associated with a nodes children in the list
        for i, e in enumerate(edges):
            if curr_node != e[0]:
                curr_node = e[0]
                sind = i

            if e[0] not in nchildren.keys():
                nchildren[e[0]] = 1
            else:
                nchildren[e[0]] += 1

            if nchildren[e[0]] > max_nchild:
                if e[0] not in nodes_to_split.keys():
                    nodes_to_split[e[0]] = sind

        # if none of the nodes are high degree we don't need to do anything
        if len(nodes_to_split) == 0:
            return spanning_tree

        # otherwise we iterate through the tree and split high degree nodes off
        else:
            # now we split any nodes that need to be split
            counter = N
            T = nx.Graph()
            # if we are at a node we need to split
            for node, ind in nodes_to_split.items():
                nchild = nchildren[node]

                # insert floor(nchild/max_nchild) logical nodes that will be full connecting
                for j in range(nchild // max_nchild):
                    T.add_edge(node, counter)
                    for k in range(max_nchild):
                        T.add_edge(counter, edges[ind + j * max_nchild + k][1])

                    counter = counter + 1

                if nchild % max_nchild == 1:
                    for j in range((nchild // max_nchild) * max_nchild, nchild):
                        T.add_edge(node, edges[ind + j][1])

                elif nchild % max_nchild > 1:
                    T.add_edge(node, counter)

                    for k in range((nchild // max_nchild) * max_nchild, nchild):
                        T.add_edge(counter, edges[ind + j][1])

                    counter = counter + 1

            for e in edges:
                if e[0] not in nodes_to_split.keys():
                    T.add_edge(e[0], e[1])
            return T


def generate_spanning_tree(M, max_nchild=None, root_index=0):
    """Construct a networkx graph object from the maximum weight spanning tree of some matrix M.  This function
    can optionally insert logical nodes to prevent any node having a more children than max_nchild, and can be chosen
    so that any node is the root index of the tree.

    :param M: The "distance" matrix used to define a weighted graph of the nodes to be represented as a tree
    :type M: np.ndarray
    :param max_nchild: The maximum allowed number of children.  If this is none, the number of children is not constrained, defaults to None
    :type max_nchild: int, None, optional
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
    spanning_tree = __split_high_degree_nodes(
        spanning_tree, M.shape[0], root_index, max_nchild=max_nchild
    )

    return __insert_physical_nodes(spanning_tree, M.shape[0], root_index)
