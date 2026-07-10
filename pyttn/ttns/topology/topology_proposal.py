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

from typing import Callable, List, Optional

import numpy as np

from ..ttns.ntreeExt import ntree, ntreeBuilder
from ..ttns.topology_tree import TopoTree
from .interaction_graph import InteractionGraph
from .networkx_converter import convert_nx_to_tree
from .spanning_tree import generate_spanning_tree


def connected_components(W: np.ndarray) -> List[List[int]]:
    """Return the connected components of a weight matrix (nonzero entries treated as edges), as lists of node indices, ordered by their smallest member.

    :param W: Symmetric weight matrix
    :type W: np.ndarray
    :return: The connected components, each a sorted list of node indices
    :rtype: list[list[int]]
    """
    n = W.shape[0]
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            row = W[u]
            for v in range(n):
                if not visited[v] and row[v] != 0:
                    visited[v] = True
                    stack.append(v)
        components.append(sorted(comp))
    components.sort(key=lambda c: c[0])
    return components


def bridge_disconnected_components(W: np.ndarray, components: List[List[int]], bridge_weight: Optional[float] = None) -> np.ndarray:
    """Connect disconnected components of a weight matrix. 
    
    Adds weak bridge edges between connected components so that a single spanning tree can be built.

    :param W: Symmetric weight matrix
    :type W: np.ndarray
    :param components: Connected components of ``W``.
    :type components: list[list[int]]
    :param bridge_weight: Weight assigned to bridge edges.
    :type bridge_weight: float, optional
    :return: A copy of ``W`` with bridging edges added
    :rtype: np.ndarray
    """
    if bridge_weight is None:
        nonzero = np.abs(W[W != 0])
        bridge_weight = float(np.min(nonzero)) * 1e-6 if nonzero.size > 0 else 1.0

    W = W.copy()
    for comp_a, comp_b in zip(components[:-1], components[1:]):
        u, v = comp_a[0], comp_b[0]
        W[u, v] = bridge_weight
        W[v, u] = bridge_weight
    return W


def build_topology_tree(weights: np.ndarray, labels: List[str], tree_generator: Callable, **tree_kwargs) -> TopoTree:
    """Build a labelled TopoTree from a (fully connected) weight matrix and matching labels.

    :param weights: Symmetric weight matrix 
    :type weights: np.ndarray
    :param labels: Labels for each row/column 
    :type labels: list[str]
    :param tree_generator: A function used to construct a tree from a weight matrix
    :type tree_generator: Callable
    :param `**tree_kwargs`: Additional keyword arguments forwarded to ``tree_generator``
    :return: A labelled topology tree
    :rtype: TopoTree
    """
    if len(labels) == 1:
        # a single node has no edges to build a spanning tree from;
        # convert_nx_to_tree returns an empty tree in this case, so we construct
        # the trivial one-leaf tree directly instead.
        return TopoTree(ntree("0"), list(labels))

    nx_tree, root_ind = tree_generator(weights, **tree_kwargs)
    tree, leaf_indices = convert_nx_to_tree(nx_tree, root_ind)

    # leaf_indices[original_index] = position in dfs leaf order; invert it so that we can
    # relabel the dfs-ordered leaves with their original node labels.
    dfs_original_index = [0] * len(leaf_indices)
    for original_index, position in enumerate(leaf_indices):
        dfs_original_index[position] = original_index
    leaf_labels = [labels[i] for i in dfs_original_index]

    return TopoTree(tree, leaf_labels)

def _fill_backbone_children(dest_node, skeleton_node, subtree_root_iter) -> None:
    for i in range(skeleton_node.size()):
        child = skeleton_node.at(i)
        if child.is_leaf():
            dest_node.insert(next(subtree_root_iter))
        else:
            new_child = ntree(str(child.value))
            _fill_backbone_children(new_child.root(), child, subtree_root_iter)
            dest_node.insert(new_child.root())


def join_disconnected_components(components: List[List[int]], weights: np.ndarray, labels: List[str], tree_generator: Callable, degree: int = 1, **tree_kwargs) -> TopoTree:
    """Build a topology for a disconnected interaction graph.

    Constructs an independent subtree for each connected component and joins the
    resulting subtrees with a balanced ``degree``-ary backbone. This preserves the
    internal structure of each component without introducing artificial
    cross-component edges.

    :param components: The connected components of ``weights``
    :type components: list[list[int]]
    :param weights: Symmetric weight matrix 
    :type weights: np.ndarray
    :param labels: Labels for each row/column 
    :type labels: list[str]
    :param tree_generator: Function used to construct a subtree for each component.
    :type tree_generator: Callable
    :param degree: Backbone degree (1 for a linear chain, 2 for a balanced binary tree, etc.).
    :type degree: int, optional
    :param `**tree_kwargs`: Additional keyword arguments forwarded to ``tree_generator``
    :return: A labelled topology tree covering all components
    :rtype: TopoTree
    """
    if degree < 1:
        raise ValueError(f"degree must be a positive integer, got {degree}")

    subtrees = []
    for comp in components:
        comp_labels = [labels[i] for i in comp]
        comp_weights = weights[np.ix_(comp, comp)]
        subtrees.append(build_topology_tree(comp_weights, comp_labels, tree_generator, **tree_kwargs))

    n = len(subtrees)

    # the backbone's own bond dimensions are meaningless placeholders (chi=1) -
    # every node's real value is proposed later from the fully-assembled tree
    # (see MethodBuilder.build), exactly like a bath's attachment placeholder.
    if degree == 1:
        skeleton = ntreeBuilder.mps_tree([1] * n, 1)
    else:
        skeleton = ntreeBuilder.mlmctdh_tree([1] * n, degree, 1, include_local_basis_transformation=False)

    combined = ntree(str(skeleton.root().value))
    subtree_root_iter = iter(subtree.tree.root() for subtree in subtrees)
    _fill_backbone_children(combined.root(), skeleton.root(), subtree_root_iter)

    combined_labels = [label for subtree in subtrees for label in subtree.leaf_labels]
    return TopoTree(combined, combined_labels)


def propose_topology_from_graph(graph: InteractionGraph, tree_generator: Callable = generate_spanning_tree, disconnected_strategy: str = "join", degree: int = 1, bridge_weight: Optional[float] = None, **tree_kwargs) -> TopoTree:
    """Build a topology tree from an interaction graph.

    If the graph is disconnected, components are either connected with weak bridge edges (``"weak_link"``) or converted into independent subtrees and joined by a ``degree``-ary backbone (``"join"``).

    :param graph: Interaction Graph
    :type graph: InteractionGraph
    :param tree_generator: Function used to construct a tree from a weight matrix.
    :type tree_generator: Callable, optional
    :param disconnected_strategy: Strategy for handling disconnected graphs.
    :type disconnected_strategy: {"weak_link", "join"}, optional
    :param degree: Backbone degree used when ``disconnected_strategy="join"``
    :type degree: int, optional
    :param bridge_weight: Edge weight used to bridge components when
        ``disconnected_strategy="weak_link"``, defaults to a small fraction of the
        smallest existing edge weight (or 1.0 if there are none)
    :type bridge_weight: float, optional
    :param `**tree_kwargs`: Additional keyword arguments forwarded to ``tree_generator``
    :return: A labelled topology tree covering every node of ``graph``
    :rtype: TopoTree
    """
    labels, weights = graph.weight_matrix()

    if len(labels) <= 1:
        return TopoTree(ntree("0"), list(labels)) if labels else TopoTree(ntree(), [])

    components = connected_components(weights)

    if len(components) == 1:
        return build_topology_tree(weights, labels, tree_generator, **tree_kwargs)

    if disconnected_strategy == "weak_link":
        bridged = bridge_disconnected_components(weights, components, bridge_weight=bridge_weight)
        return build_topology_tree(bridged, labels, tree_generator, **tree_kwargs)

    if disconnected_strategy == "join":
        return join_disconnected_components(components, weights, labels, tree_generator, degree, **tree_kwargs)

    raise ValueError(f"Unknown disconnected_strategy '{disconnected_strategy}'; expected 'weak_link' or 'join'.")
