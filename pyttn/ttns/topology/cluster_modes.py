from typing import Dict, List, Set
import math

from .interaction_graph import build_interaction_graph
from ..sop.system_information import SystemInfo


def cluster_modes_graph(op, sysinfo: SystemInfo, max_lhd: int, scaling: str = "uniform", score: str = "weight", cluster_prefix: str = "C",):
    """
    Construct a new SystemInfo by clustering strongly interacting composite modes.

    param op: The operator used to define the interaction structure
    :type op: lCSOP
    :param sysinfo: The SystemInfo object defining the current composite and primitive mode decomposition
    :type sysinfo: SystemInfo
    :param max_lhd: Maximum allowed local Hilbert space dimension of any generated composite mode
    :type max_lhd: int
    :param score: Scoring metric used when selecting candidate merges

        - ``"weight"``: Use the total interaction weight between clusters
        - ``"weight_per_dim"``: Weight divided by the merged local Hilbert
        space dimension
        - ``"weight_per_logdim"``: Weight divided by the logarithm of the
        merged local Hilbert space dimension

    :type score: str, optional
    :param cluster_prefix: Prefix used when generating labels for new composite modes
    :type cluster_prefix: str, optional

    :returns: A new SystemInfo object containing the clustered composite modes
    :rtype: SystemInfo
    """

    G = build_interaction_graph(op, sysinfo, store_term_data=False,scaling=scaling)

    # local Hilbert space dimensions of current composite modes
    comp_dims = {}

    for comp_label, prims in sysinfo.items():
        d = 1
        for p in prims.values():
            d *= p.lhd
        comp_dims[comp_label] = d

    # initialise clusters
    clusters: Dict[int, Set[str]] = {i: {label} for i, label in enumerate(G.nodes.keys())}
    cluster_dims = {i: comp_dims[label] for i, label in enumerate(G.nodes.keys())}
    label_to_cluster = {label: i for i, label in enumerate(G.nodes.keys())}

    # function for computing interaction weight between two clusters

    def cluster_weight(ca, cb):
        w = 0.0
        for edge, data in G.edges.items():
            u, v = tuple(edge)

            a = label_to_cluster[u]
            b = label_to_cluster[v]

            if {a, b} == {ca, cb}:
                w += data["weight"]

        return w

    # greedily merge strongest coupled clusters that don't violate the hilbert space dimension constraint
    while True:
        best_pair = None
        best_score = -1.0

        active = list(clusters.keys())

        for i in range(len(active)):
            for j in range(i + 1, len(active)):

                ca = active[i]
                cb = active[j]

                merged_dim = cluster_dims[ca] * cluster_dims[cb]

                if merged_dim > max_lhd:
                    continue

                w = cluster_weight(ca, cb)

                if w <= 0.0:
                    continue

                if score == "weight":
                    s = w

                elif score == "weight_per_dim":
                    s = w / merged_dim

                elif score == "weight_per_logdim":
                    s = w / math.log(max(merged_dim, 2))

                else:
                    raise ValueError(f"Unknown score '{score}'")

                if s > best_score:
                    best_score = s
                    best_pair = (ca, cb)

        if best_pair is None:
            break

        ca, cb = best_pair

        new_cluster = clusters[ca] | clusters[cb]

        clusters[ca] = new_cluster
        cluster_dims[ca] *= cluster_dims[cb]

        for node in clusters:
            label_to_cluster[node] = ca

        del clusters[cb]
        del cluster_dims[cb]

    # convert cluster -> primitive labels

    groups = {}

    for i, (_, comp_labels) in enumerate(clusters.items()):
        primitive_labels = []
        for comp in sorted(comp_labels):
            primitive_labels.extend(
                sysinfo[comp].keys()
            )

        groups[f"{cluster_prefix}{i}"] = primitive_labels

    return sysinfo.group_modes(groups)