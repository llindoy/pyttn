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
from math import ceil


def distance_matrix_to_graph(M):
    return nx.from_numpy_array(np.abs(M - np.diag(np.diag(M))))


def __insert_physical_nodes(spanning_tree, N, root_ind):
    # a function for taking a networkx tree generated from a max weight spanning tree of a graph
    # and inserts children nodes below each of the leaf nodes representing the physical tree nodes
    # additionally this shifts all indices stored in the tree so that they each have a unique label
    # and the new children nodes have the label that their parent used to have
    nindex = spanning_tree.number_of_nodes()

    # iterate over each node in the tree and determine whether we need to insert the child node
    # this is done by iterating over the tree in a depth first search order and determining if it
    # is a leaf node in the tree.  If it is we don't need to do anything, if it is not, then we will
    # insert a node beneath the present node and update its index

    #get a list of all nodes in the tree
    edges = sorted([edge for edge in nx.dfs_edges(
        spanning_tree, source=root_ind)], key=lambda x: np.abs(x[0]-root_ind))
    
    #determine the number of children each node has
    nchildren = [0 for x in range(N)]
    for i, e in enumerate(edges):
        if e[0] < N:
            nchildren[e[0]] += 1

    #and the total number of nodes that aren't leaf nodes

    mapping = {}      
    counter = 0  
    for i in range(nindex):
        if(i < N):
            if nchildren[i] == 0:
                mapping[i] = i
            else:
                mapping[i] = nindex+counter
                counter = counter+1
        else:
            mapping[i] = i


    # iterate over the tree and add nodes to any n
    nx.relabel_nodes(spanning_tree, mapping=mapping, copy=False)

    for i in range(N):
        if nchildren[i] > 0:
            spanning_tree.add_edge(mapping[i], i)
    print(mapping)
    return spanning_tree, mapping[root_ind]


def chunks(nodes, n):
    N = ceil(len(nodes)/n)
    for i in range(0, len(nodes), N):
        yield nodes[i:i + N]


def __split_node_mps(T, node, children, nindex, handle_leaves = False):
    curr_node = node
    if handle_leaves:
        if len(children) == 1:
            T.add_edge(curr_node, children[0])
        elif len(children) == 2:
            T.add_edge(curr_node, nindex)
            T.add_edge(nindex, children[0])
            T.add_edge(nindex, children[1])
            nindex += 1
        elif len(children) > 2:
            T.add_edge(curr_node, nindex)
            curr_node = nindex
            nindex += 1
            for cind in range(len(children)-2):
                T.add_edge(curr_node, children[cind])
                T.add_edge(curr_node, nindex)
                curr_node = nindex
                nindex += 1
            T.add_edge(curr_node, children[-2])
            T.add_edge(curr_node, children[-1])
    else:
        if len(children) > 0:
            for cind in range(len(children)-1):
                T.add_edge(curr_node, children[cind])
                T.add_edge(curr_node, nindex)
                curr_node = nindex
                nindex += 1
            T.add_edge(curr_node, children[-1])
    return T, nindex

def __split_node_branching(T, node, children, max_nchild, nindex, handle_leaves = False):
    curr_node = node
    if len(children) > max_nchild:
        if handle_leaves:
            T.add_edge(curr_node, nindex)
            curr_node = nindex
            nindex += 1
        # if the children list is longer than the maximum degree, iterate over creating up to max degree chunks
        for chunk in chunks(children, max_nchild):
            if len(chunk) == 1:
                T.add_edge(curr_node, chunk[0])
            else:
                # for each chunk
                nlabel = nindex
                T.add_edge(curr_node, nlabel)
                nindex += 1
                T, nindex = __split_node_branching(T, nlabel, chunk, max_nchild, nindex)
    else:
        if len(children) > 1 and handle_leaves:
            T.add_edge(curr_node, nindex)
            curr_node = nindex
            nindex += 1
            for child in children:
                T.add_edge(curr_node, child)           
        else:
            for child in children:
                T.add_edge(curr_node, child)
    return T, nindex

def __split_node(T, node, children, max_nchild, nindex, handle_leaves = False):
    if max_nchild is None: 
        for i in range(len(children)):
            T.add_edge(node, children[i])
    elif max_nchild > 1:
        T, nindex = __split_node_branching(T, node, children, max_nchild, nindex, handle_leaves=handle_leaves)
    elif max_nchild == 1:
        T, nindex = __split_node_mps(T, node, children, nindex, handle_leaves=handle_leaves)
    else:
        for i in range(len(children)):
            T.add_edge(node, children[i])

    return T, nindex


def __split_high_degree_nodes(spanning_tree, N, root_index, max_nchild=None, max_nleaves=None):
    if max_nchild is None and max_nleaves is None:
        # if max degree has not been specified we just return the current tree
        return spanning_tree
    else:
        # in this case iterate through the tree determine if any nodes are too large and if they are we partition
        # the node into sets of the correct size.  To do this we get a DFS edges list and if there are any instances
        # of nodes with more than max_nchild children we insert sufficiently many logical nodes so that we have the
        # correct degree of connectivity
        edges = sorted([edge for edge in nx.dfs_edges(
            spanning_tree, source=root_index)], key=lambda x: np.abs(x[0]-root_index))

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

            if curr_node not in nchildren.keys():
                nchildren[curr_node] = 1
            else:
                nchildren[curr_node] += 1

            if nchildren[curr_node] > max_nchild:
                if curr_node not in nodes_to_split.keys():
                    nodes_to_split[curr_node] = sind

        for i, e in enumerate(edges):
            if curr_node != e[1]:
                curr_node = e[1]

            if curr_node not in nchildren.keys():
                nchildren[curr_node]=0

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

                print([nchildren[edges[ind+i][1]] for i in range(nchild)])
                # get a list containing all of its children
                leaf_children = [edges[ind+i][1] for i in range(nchild) if nchildren[edges[ind+i][1]]==0]

                internal_children = [edges[ind+i][1] for i in range(nchild) if nchildren[edges[ind+i][1]]>0]
                print(edges[ind], leaf_children, internal_children)

                if max_nchild is None:
                    max_nchild = max_nleaves    

                if max_nleaves is None:
                    T, counter = __split_node(T, node, leaf_children+internal_children, max_nchild, counter)
                else:           
                    T, counter = __split_node(T, node, internal_children, max_nchild, counter)
                    T, counter = __split_node(T, node, leaf_children, max_nleaves, counter, handle_leaves=True)

            for e in edges:
                if e[0] not in nodes_to_split.keys():
                    T.add_edge(e[0], e[1])
            return T


def generate_spanning_tree(M, max_internal_children=None, max_leaf_children=None, root_index=0):
    """Construct a networkx graph object from the maximum weight spanning tree of some matrix M.  This function
    can optionally insert logical nodes to prevent any node having a more children than max_nchild, and can be chosen
    so that any node is the root index of the tree.

    :param M: The "distance" matrix used to define a weighted graph of the nodes to be represented as a tree
    :type M: np.ndarray
    :param max_internal_children: The maximum allowed number of internal node children.  If this is none, the number of internal node children will be set to the value of max_leaf_children, defaults to None
    :type max_internal_children: int, None, optional
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
    spanning_tree = __split_high_degree_nodes(
        spanning_tree, M.shape[0], root_index, max_nchild=max_internal_children, max_nleaves=max_leaf_children
    )

    return __insert_physical_nodes(spanning_tree, M.shape[0], root_index)
