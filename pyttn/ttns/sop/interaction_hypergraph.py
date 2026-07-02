
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

from .system_information import SystemInfo
from .operator_builder import lCSOP
class InteractionHypergraph:
    """Hypergraph structure for representing multi-body interactions."""

    def __init__(self):
        self.nodes = {}
        self.hyperedges = {}  # key: frozenset of nodes

    def add_node(self, label):
        """
        Add a node for a mode if it doesn't already exist.

        :param label: The mode label
        :type label: str
        """
        if label not in self.nodes:
            self.nodes[label] = {"terms": []}

    def add_hyperedge(self, nodes):
        """ 
        Add a hyperedge connecting a set of modes if it doesn't already exist.

        :param nodes: The mode labels forming the hyperedge
        :type nodes: iterable[str]
        :returns: The hyperedge key
        :rtype: frozenset
        """

        key = frozenset(nodes)
        if key not in self.hyperedges:
            self.hyperedges[key] = {"weight": 0.0, "terms": []}
        return key

    def get_hyperedge(self, nodes):
        """
        Return the hyperedge data for a given set of nodes.

        :param nodes: The mode labels forming the hyperedge
        :type nodes: iterable[str]
        :returns: The hyperedge data dictionary or None
        :rtype: dict or None
        """

        return self.hyperedges.get(frozenset(nodes))

    def edges(self):
        """
        Return the list of hyperedges with their weights.

        :returns: A list of (node_set, weight) pairs
        :rtype: list[tuple[set[str], float]]
        """
        return [(set(nodes), data["weight"]) for nodes, data in self.hyperedges.items()]

    def node_labels(self):
        """
        Return the sorted list of node labels.

        :returns: List of node labels
        :rtype: list[str]
        """

        return sorted(self.nodes.keys())


def build_interaction_hypergraph(
    op: lCSOP,
    sysinfo: SystemInfo,
    store_term_data: bool = False,
) -> InteractionHypergraph:
    """
    Build an interaction hypergraph from an lCSOP and SystemInfo.

    Nodes correspond to composite modes.
    Hyperedges connect all composite modes that appear in the same term.

    Hyperedge weights = sum |coeff|^2 over contributing terms.

    :param op: The lCSOP operator
    :type op: lCSOP
    :param sysinfo: The SystemInfo defining composite and primitive modes
    :type sysinfo: SystemInfo
    :param store_term_data: Whether to store contributing term data in nodes and hyperedges
    :type store_term_data: bool
    :returns: The constructed interaction hypergraph
    :rtype: InteractionHypergraph
    """


    H = InteractionHypergraph()

    # map primitive modes → composite modes
    prim_to_comp = {}
    for comp_label, prims in sysinfo.items():
        for p in prims:
            prim_to_comp[p] = comp_label

    opdict = op.sop.get_operator_dictionary()

    for term, coeff in op.sop:
        pop = term.as_sPOP(opdict)

        comps_in_term = set()

        for opi in pop:
            label = op.index_to_label[opi.mode]

            if label not in prim_to_comp:
                raise ValueError(f"Primitive mode '{label}' not found")

            comp = prim_to_comp[label]
            comps_in_term.add(comp)

        if len(comps_in_term) == 0:
            continue

        w = abs(coeff(0))**2
        for c in comps_in_term:
            H.add_node(c)
            if store_term_data:
                H.nodes[c]["terms"].append(coeff * term)

        key = H.add_hyperedge(comps_in_term)
        H.hyperedges[key]["weight"] += w

        if store_term_data:
            H.hyperedges[key]["terms"].append(coeff * term)

    return H