
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

from pyttn.ttns.sop.interaction_hypergraph import build_interaction_hypergraph
from itertools import combinations

from .system_information import SystemInfo
from .operator_builder import lCSOP

class InteractionGraph:
    """A simple undirected graph structure for representing interactions between composite modes."""
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, label):
        """
        Add a node for a composite mode if it doesn't already exist.
        
        :param label: The composite mode label
        :type label: str
        """
        if label not in self.nodes:
            self.nodes[label] = {"terms": []}

    def add_edge(self, u, v):
        """
        Add an edge between two composite modes if it doesn't already exist.

        :param u: The first composite mode label
        :type u: str
        :param v: The second composite mode label
        :type v: str
        :returns: The edge key
        :rtype: frozenset
        """
        key = frozenset((u, v))
        if key not in self.edges:
            self.edges[key] = {"weight": 0.0, "terms": []}
        return key
    
    def weight_matrix(self) -> tuple[list[str], np.ndarray]:
        """
        Return the adjacency matrix of the graph with edge weights.

        :returns: A tuple containing the list of node labels and the corresponding weight matrix
        :rtype: tuple[list[str], np.ndarray]
        """
        labels = sorted(self.nodes.keys())
        n = len(labels)
        W = np.zeros((n, n), dtype=float)

        label_to_index = {label: i for i, label in enumerate(labels)}

        for (u, v), data in self.edges.items():
            i, j = label_to_index[u], label_to_index[v]
            W[i, j] = data["weight"]
            W[j, i] = data["weight"]  # undirected graph

        return labels, W

    def get_edge(self, u, v):
        """Return the edge data for the edge between u and v, or None if it doesn't exist.
        
        :param u: The first composite mode label
        :type u: str
        :param v: The second composite mode label
        :type v: str
        :returns: The edge data dictionary or None
        :rtype: dict or None
        """
        return self.edges.get(frozenset((u,v)))

    def neighbors(self, u):
        """Return a list of neighbors for a given node u.
        
        :param u: The composite mode label
        :type u: str
        :returns: List of neighboring composite mode labels
        :rtype: list[str]
        """
        return [list(e - {u})[0] for e in self.edges if u in e]

def hypergraph_to_graph(H, scaling: str = "uniform", store_term_data: bool = False,):
   
    """
    Project an interaction hypergraph to an interaction graph.

    Nodes correspond to composite modes.
    Edges connect composites that appear together in a hyperedge.

    Edge weights are obtained by distributing hyperedge weights
    according to the chosen scaling scheme.

    :param H: The interaction hypergraph
    :type H: InteractionHypergraph
    :param scaling: The scaling scheme used to distribute hyperedge weights
        among pairwise edges ("uniform", "linear", or "none")
    :type scaling: str
    :param store_term_data: Whether to propagate contributing term data
        to nodes and edges
    :type store_term_data: bool
    :returns: The constructed interaction graph
    :rtype: InteractionGraph
    """


    G = InteractionGraph()

    for node in H.nodes:
        G.add_node(node)
        if store_term_data:
            G.nodes[node]["terms"].extend(H.nodes[node]["terms"])

    for nodes, data in H.hyperedges.items():
        nodes = list(nodes)
        k = len(nodes)

        if k < 2:
            continue

        w = data["weight"]

        # --------------------------------------------------
        # scaling choice
        # --------------------------------------------------
        if scaling == "uniform":
            factor = w / (k * (k - 1) / 2)

        elif scaling == "linear":
            factor = w / (k - 1)

        elif scaling == "none":
            factor = w

        else:
            raise ValueError(f"Unknown scaling: {scaling}")

        # --------------------------------------------------
        # distribute to pairwise edges
        # --------------------------------------------------
        for u, v in combinations(nodes, 2):
            edge_key = G.add_edge(u, v)
            G.edges[edge_key]["weight"] += factor

            if store_term_data:
                G.edges[edge_key]["terms"].extend(data["terms"])

    return G


def build_interaction_graph(op: lCSOP, sysinfo : SystemInfo, store_term_data : bool = False, scaling: str = "uniform") -> InteractionGraph:
    """
    Build an interaction graph from an lCSOP and SystemInfo.

    Nodes correspond to composite modes.
    Edges connect composites that appear in the same term.

    Edge weights = sum |coeff|^2 over contributing terms.

    :param op: The lCSOP operator
    :type op: lCSOP
    :param sysinfo: The SystemInfo defining composite and primitive modes
    :type sysinfo: SystemInfo
    :param store_term_data: Whether to store contributing term data in edges and nodes
    :type store_term_data: bool
    :param scaling: The scaling scheme used to distribute hyperedge weights
        among pairwise edges ("uniform", "linear", or "none")
    :type scaling: str
    :returns: The constructed interaction graph 
    :rtype: InteractionGraph
    """

    H = build_interaction_hypergraph(
        op,
        sysinfo,
        store_term_data=store_term_data
    )

    return hypergraph_to_graph(
        H,
        scaling=scaling,
        store_term_data=store_term_data
    )
