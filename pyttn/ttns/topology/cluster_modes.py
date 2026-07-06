"""A module with classes for setting ntree elements from leaves to root."""

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

from .interaction_graph import build_interaction_graph
from ..sop.system_information import SystemInfo

def cluster_from_weight_matrix(W: np.ndarray, local_dims: dict[str, int], max_lhd: int, score: str = "weight") -> list[set[int]]:
    """
    Cluster modes based on a weighted adjacency matrix.

    :param W: Symmetric weight matrix
    :type W: np.ndarray
    :param local_dims: Local Hilbert-space dimension of each mode
    :type local_dims: dict[str, int]
    :param max_lhd: Maximum local Hilbert-space dimension permitted for any merged cluster
    :type max_lhd: int
    :param score: Merge scoring metric
    :type score: str
    :returns: Mapping clusters to indices
    :rtype: list[set[int]]
    """

    clusters = { i: {i}  for i in range(W.shape[0])}
    cluster_dims = {i: local_dims[i] for i in range(W.shape[0])}
    
    while True:
        best_pair = None
        best_score = -np.inf

        active = list(clusters.keys())

        for ia in range(len(active)):
            for ib in range(ia + 1, len(active)):
                ca = active[ia]
                cb = active[ib]

                merged_dim = (cluster_dims[ca] * cluster_dims[cb])

                if merged_dim > max_lhd:
                    continue

                # total interaction between clusters
                w = sum(W[i, j] for i in clusters[ca] for j in clusters[cb])

                if w <= 0:
                    continue

                if score == "weight":
                    s = w

                elif score == "weight_per_dim":
                    s = w / merged_dim

                elif score == "weight_per_logdim":
                    s = w / np.log(max(merged_dim, 2))

                else:
                    raise ValueError(f"Unknown score '{score}'")

                if s > best_score:
                    best_score = s
                    best_pair = (ca, cb)

        if best_pair is None:
            break

        ca, cb = best_pair

        clusters[ca] |= clusters[cb]
        cluster_dims[ca] *= cluster_dims[cb]

        del clusters[cb]
        del cluster_dims[cb]

    return clusters


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
    labels, W = G.weight_matrix()

    local_dims = [ sysinfo.local_dim(label) for label in labels]

    clusters = cluster_from_weight_matrix(W, local_dims, max_lhd=max_lhd, score=score)
    groups = {}

    for i, inds in enumerate(clusters.values()):
            primitive_labels = []
            for i in sorted(inds):
                primitive_labels.extend(sysinfo[labels[i]].keys())
            groups[f"{cluster_prefix}{i}"] = primitive_labels

    return sysinfo.group_modes(groups)


    