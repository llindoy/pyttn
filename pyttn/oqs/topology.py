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

from typing import Callable, Dict, List, Optional

import numpy as np

from pyttn import ntree

from ..ttns.sop.system_information import SystemInfo
from ..ttns.sop.labelled_SOP import lCSOP
from ..ttns.ttns.topology_tree import TopoTree
from ..ttns.topology import InteractionGraph, build_interaction_graph, cluster_modes_graph, generate_spanning_tree, propose_topology_from_graph
from ..ttns.topology.interaction_terms import interaction_terms

from .baths.bath import BathSpec
from .model import OQSModel


def _prim_to_comp(sysinfo: SystemInfo) -> Dict[str, str]:
    """Map every primitive label in a SystemInfo to the composite label that owns it."""
    return {p: c for c, prims in sysinfo.items() for p in prims}


def _lowest_common_ancestor(paths: List[List[int]]) -> List[int]:
    """Return the path to the lowest common ancestor of a set of node paths."""
    if not paths:
        return []
    lca = list(paths[0])
    for path in paths[1:]:
        common = []
        for a, b in zip(lca, path):
            if a != b:
                break
            common.append(a)
        lca = common
    return lca

def propose_system_topology(model: OQSModel, tree_generator: Callable = generate_spanning_tree, cluster_max_lhd: Optional[int] = None, system_info: Optional[SystemInfo] = None, system_generator: Optional[lCSOP] = None, disconnected_strategy: str = "join", degree: int = 1, bridge_weight: Optional[float] = None, **tree_kwargs) -> TopoTree:
    """Propose a tree topology for the system degrees of freedom of an OQSModel.
    Builds a weighted interaction graph from the system generator and uses it to construct a topology tree.  This function only considers system modes, ignoring all bath degrees of freedom.

    If the interaction graph is disconnected, components are handled the ``disconnected_strategy`` (either bridged through inclusion of artificial weak links or joined after tree structures are generated for each disconnected subgraph).

    :param model: The open quantum system model to propose a topology for
    :type model: OQSModel
    :param tree_generator: A function used to construct a tree from a weight matrix, defaults to :func:`generate_spanning_tree`
    :type tree_generator: Callable, optional
    :param cluster_max_lhd: Cluster composite modes so no leaf exceeds this local Hilbert-space dimension.
    :type cluster_max_lhd: int, optional
    :param system_info: System information to use in place of ``model.system_info`` (e.g. for Liouville-space promoted SystemInfo), defaults to None
    :type system_info: SystemInfo, optional
    :param system_generator: System Generator to use in place of ``model.system_generator`` (e.g. a Liouville-space promoted Generator), defaults to None
    :type system_generator: lCSOP, optional
    :param disconnected_strategy: How to handle a disconnected interaction graph
    :type disconnected_strategy: {"weak_link", "join"}, optional
    :param degree:  Degree of tree used when joining disconnected subgraphs
    :type degree: int, optional
    :param bridge_weight: Edge weight used to bridge components when ``disconnected_strategy="weak_link"``
    :type bridge_weight: float, optional
    :param `**tree_kwargs`: Additional keyword arguments forwarded to ``tree_generator``
    :return: A labelled topology tree covering the system composite modes
    :rtype: TopoTree
    """
    sysinfo = system_info if system_info is not None else model.system_info
    op = system_generator if system_generator is not None else model.system_generator

    #cluster modes
    if cluster_max_lhd is not None:
        sysinfo = cluster_modes_graph(op, sysinfo, max_lhd=cluster_max_lhd)

    #build graph
    graph = build_interaction_graph(op, sysinfo)

    #propose topology
    return propose_topology_from_graph(graph, tree_generator, disconnected_strategy=disconnected_strategy, degree=degree, bridge_weight=bridge_weight, **tree_kwargs)


def attach_bath_placeholders(model: OQSModel, topology: TopoTree) -> TopoTree:
    """Attach a placeholder leaf for each bath in an OQSModel to a system topology.

    Each bath is represented by a single leaf labelled with its tag, attached at4the lowest common ancestor of the system modes it couples to.  This function is best applied when baths couple locally to systems.
    In the case that you have many baths coupling to many system modes this may lead to very high degree trees at the root.  In this case alternative approaches for attaching the baths are recommended.

    :param model: The open quantum system model 
    :type model: OQSModel
    :param topology: The topology to augment
    :type topology: TopoTree
    :return: The augmented topology tree
    :rtype: TopoTree
    """
    prim_to_comp = _prim_to_comp(model.system_info)
    leaf_paths = topology.leaf_paths()
    existing = set(topology.leaf_order())

    for spec in model.baths:
        #skip any baths that are already in the tree
        if spec.tag in existing:
            continue

        #otherwise find which system sites it couples to
        prim_labels = set()
        for op in spec.coupling_ops:
            prim_labels |= op.sites()

        #and use this to determine which composite sites it couples to
        comps = set()
        for p in prim_labels:
            if p not in prim_to_comp:
                raise ValueError(f"Bath '{spec.tag}' couples to unknown primitive '{p}'.")
            comps.add(prim_to_comp[p])

        #make sure they are all in the tree
        missing = comps - set(leaf_paths)
        if missing:
            raise ValueError(f"Bath '{spec.tag}' couples to composite mode(s) {sorted(missing)} " "that are not present as leaves in the supplied topology.")

        #and now insert the bath node at the lowest common ancestor of its system nodes.  
        lca_path = _lowest_common_ancestor([leaf_paths[c] for c in comps])
        topology.insert_subtree(lca_path, subtree=ntree("1"), subtree_labels=[spec.tag])
        existing.add(spec.tag)

    return topology


def default_bath_weight(spec: BathSpec, sysinfo: SystemInfo) -> Dict[str, float]:
    """Compute default bath-system coupling weights.

    Each coupling contributes:

    .. math::
        \\sqrt{\\lambda \\, |g(0)|^2} = \\sqrt{\\lambda}\\, |g(0)|

    to the composite modes it interacts with.  Here :math:`\\lambda` is the bath's reorganisation energy and :math:`g(0)` is the term's scalarcoefficient evaluated at :math:`t=0`. 

    :param spec: Bath details
    :type spec: BathSpec
    :param sysinfo: System information defining composite modes
    :type sysinfo: SystemInfo
    :raises AttributeError: If the bath does not provide ``reorganisation_energy()``
    :return: Mapping from composite mode label to edge weight
    :rtype: dict[str, float]
    """
    if not hasattr(spec.bath, "reorganisation_energy"):
        raise AttributeError(f"Bath '{spec.tag}' ({type(spec.bath).__name__}) has no 'reorganisation_energy()' method; " "supply a custom bath_weight callable to compute its coupling strength.")
    lam = spec.bath.reorganisation_energy()

    weights: Dict[str, float] = {}
    for op in spec.coupling_ops:
        for comps, w, _, _ in interaction_terms(op, sysinfo):
            contribution = np.sqrt(lam * w) / len(comps)
            for c in comps:
                weights[c] = weights.get(c, 0.0) + contribution
    return weights


def build_joint_interaction_graph(model: OQSModel, sysinfo: Optional[SystemInfo] = None, system_generator: Optional[lCSOP] = None, bath_weight: Optional[Callable[[BathSpec, SystemInfo], Dict[str, float]]] = None) -> InteractionGraph:
    """Build an interaction graph for both system modes and baths.

    Adds one node per bath to the system interaction graph and connects it to the4composite modes it couples to. Edge weights are computed by ``bath_weight``

    :param model: The open quantum system model 
    :type model: OQSModel
    :param sysinfo: System information to use in place of ``model.system_info`` (e.g. a Liouville-space promoted SystemInfo), defaults to None
    :type sysinfo: SystemInfo, optional
    :param system_generator: System Generator to use in place of ``model.system_generator`` (e.g. a Liouville-space promoted Generator), defaults to None
    :type system_generator: lCSOP, optional
    :param bath_weight: Callable that computes bath-to-system edge weights.
    :type bath_weight: Callable, optional
    :return: The combined system+bath interaction graph
    :rtype: InteractionGraph
    """
    sysinfo = sysinfo if sysinfo is not None else model.system_info
    op = system_generator if system_generator is not None else model.system_generator
    weight_fn = bath_weight if bath_weight is not None else default_bath_weight

    graph = build_interaction_graph(op, sysinfo)

    for spec in model.baths:
        graph.add_node(spec.tag)
        for comp, w in weight_fn(spec, sysinfo).items():
            if comp not in graph.nodes:
                raise ValueError(f"Bath '{spec.tag}' couples to composite mode '{comp}' which is not part of the system interaction graph.")
            key = graph.add_edge(spec.tag, comp)
            graph.edges[key]["weight"] += w

    return graph


def propose_joint_topology(model: OQSModel, tree_generator: Callable = generate_spanning_tree, cluster_max_lhd: Optional[int] = None, bath_weight: Optional[Callable[[BathSpec, SystemInfo], Dict[str, float]]] = None, system_info: Optional[SystemInfo] = None, system_generator: Optional[lCSOP] = None, graph: Optional[InteractionGraph] = None, disconnected_strategy: str = "join", degree: int = 1, bridge_weight: Optional[float] = None, **tree_kwargs) -> TopoTree:
    """Propose a topology spanning both system and bath degrees of freedom.

    Builds a joint interaction graph containing both bath and system information to construct a topology tree using the provided ``tree_generator``. Each bath is represented as a single leaf node placeholder that is to be filled in at a later stage. 

    :param model: The open quantum system model 
    :type model: OQSModel
    :param tree_generator: Tree-construction function
    :type tree_generator: Callable, optional
    :param cluster_max_lhd:  Maximum leaf local Hilbert-space dimension after optional clustering.
    :type cluster_max_lhd: int, optional
    :param bath_weight: Callable that computes bath-system edge weights.
    :type bath_weight: Callable, optional
    :param system_info:  Override ``model.system_info``, defaults to None
    :type system_info: SystemInfo, optional
    :param system_generator: Override ``model.system_generator``, defaults to None
    :type system_generator: lCSOP, optional
    :param graph: Pre-built joint interaction graph., defaults to None
    :type graph: InteractionGraph, optional
    :param disconnected_strategy: How to handle disconnected graphs.
    :type disconnected_strategy: {"weak_link", "join"}, optional
    :param degree:  Degree of tree used when joining disconnected subgraphs
    :type degree: int, optional
    :param bridge_weight: Bridge edge weight for ``"weak_link"`` mode.
    :type bridge_weight: float, optional
    :param `**tree_kwargs`: Additional arguments passed to ``tree_generator``.
    :return:  A labelled topology tree containing system modes and bath leaves.
    :rtype: TopoTree
    """
    sysinfo = system_info if system_info is not None else model.system_info
    op = system_generator if system_generator is not None else model.system_generator

    if graph is None:
        if cluster_max_lhd is not None:
            sysinfo = cluster_modes_graph(op, sysinfo, max_lhd=cluster_max_lhd)
        graph = build_joint_interaction_graph(model, sysinfo=sysinfo, system_generator=op, bath_weight=bath_weight)

    return propose_topology_from_graph(graph, tree_generator, disconnected_strategy=disconnected_strategy, degree=degree, bridge_weight=bridge_weight, **tree_kwargs)


def propose_topology(model: OQSModel, system_topology: Optional[TopoTree] = None, bath_placement: str = "attach", bath_weight: Optional[Callable[[BathSpec, SystemInfo], Dict[str, float]]] = None, **kwargs) -> TopoTree:
    """Propose a full topology for an OQSModel, including leaves/placeholders for baths.

    Two bath-placement strategies are supported: ``"attach"``, which adds bath placeholders to a generated system topology, and ``"joint"``, which includes bath nodes directly in the interaction graph before topology generation. Bath internal structure is not expanded.

    :param model: The open quantum system model
    :type model: OQSModel
    :param system_topology: Optional user-supplied system topology, defaults to None
    :type system_topology: TopoTree, optional
    :param bath_placement:  Bath-placement strategy, defaults to "attach"
    :type bath_placement: {"attach", "joint"}, optional
    :param bath_weight: Bath-system edge-weight function used in ``"joint"`` mode, defaults to func:`default_bath_weight`
    :type bath_weight: Callable, optional
    :param `**kwargs`: Additional arguments forwarded to the topology generator.
    :return: A topology tree containing system modes and bath placeholders.
    :rtype: TopoTree
    """
    if system_topology is not None:
        return attach_bath_placeholders(model, system_topology)

    if bath_placement == "attach":
        topology = propose_system_topology(model, **kwargs)
        return attach_bath_placeholders(model, topology)
    elif bath_placement == "joint":
        return propose_joint_topology(model, bath_weight=bath_weight, **kwargs)
    else:
        raise ValueError(f"Unknown bath_placement '{bath_placement}'; expected 'attach' or 'joint'.")
