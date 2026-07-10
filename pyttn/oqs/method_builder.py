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

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from pyttn import boson_mode, fermion_mode, ntree, ntreeBuilder, ttn

from ..ttns.sop.labelled_SOP import lCSOP
from ..ttns.sop.labelled_sSOP import lSOP
from ..ttns.sop.labelled_operator_dictionary import LabelledOperatorDictionary
from ..ttns.sop.super_operator import SuperOp
from ..ttns.sop.system_information import SystemInfo
from ..ttns.topology import bond_dimension_inputs, build_interaction_graph, compute_tree_cut_metrics, propose_bond_dimensions
from ..ttns.ttns.topology_tree import TopoTree

from .baths.bath import BathSpec
from .baths.bosonic_bath import BosonicBath
from .baths.discretised_bath import DiscreteBosonicBath, DiscreteFermionicBath, SplitDiscreteFermionicBath
from .baths.exponential_fit_bath import ExpFitBosonicBath, ExpFitFermionicBath
from .baths.fermionic_bath import FermionicBath
from .model import OQSModel
from .topology import attach_bath_placeholders, build_joint_interaction_graph, propose_joint_topology, propose_system_topology


#TODO: Refactor this so that we have some classes that handle aspects of this and have specialisations for constructing the different methods.  Currently this class just does too much on its own.

class Method(Enum):
    """Identifies the open quantum system simulation method to build a generator for."""

    UNITARY = "unitary"
    TEDOPA = "tedopa"
    HEOM = "heom"
    PSEUDOMODE = "pseudomode"


_LIOUVILLE_METHODS = {Method.HEOM, Method.PSEUDOMODE}
_DEFAULT_GEOM = {Method.UNITARY: "star", Method.TEDOPA: "chain"}
_DEFAULT_DEGREE = {Method.UNITARY: 2, Method.TEDOPA: 1, Method.HEOM: 2, Method.PSEUDOMODE: 2}

@dataclass
class BuildResult:
    """Result of building an OQSModel for a simulation method.

    Bundles the compiled generator, topology information, mode mappings, and
    any auxiliary objects required for simulation and observable evaluation.

    :ivar generator: Compiled generator.
    :ivar system_modes: Physical modes.
    :ivar topology: Topology tree.
    :ivar capacity: Capacity tree.
    :ivar site_map: Label-to-mode mapping.
    :ivar baths: Built bath objects.
    :ivar trace_state: Trace state, if applicable.
    :ivar jordan_wigner_ordering: Jordan-Wigner ordering, if applicable.
    :ivar system_info: Compiled system information.
    """
    generator: object
    system_modes: object
    topology: TopoTree
    capacity: TopoTree
    site_map: Dict[str, int]
    baths: Dict[str, object]
    trace_state: object = None
    jordan_wigner_ordering: Optional[List[str]] = None
    system_info: Optional[SystemInfo] = None

    def jordan_wigner(self, op: Union[lSOP, lCSOP]) -> Union[lSOP, lCSOP]:
        """Apply the same Jordan-Wigner transformation used when building:attr:`generator`.

        This ensures the transformed operator is consistent with the generated modeland states.

        :param op: Operator to transform.
        :type op: lSOP | lCSOP
        :return: Jordan-Wigner transformed operator.
        :rtype: lSOP | lCSOP
        :raises ValueError: If no Jordan-Wigner ordering is available.
        """
        if self.jordan_wigner_ordering is None:
            raise ValueError("This build has no Jordan-Wigner ordering to reuse (non-fermionic model?).")
        return op.jordan_wigner(self.jordan_wigner_ordering, self.system_info, tol=1e-15)


def _normalise_method(method: Union[str, Method]) -> Method:
    if isinstance(method, Method):
        return method
    return Method(str(method))


def _resolve_params(spec: BathSpec, method: Method) -> dict:
    if method.value in spec.params:
        return spec.params[method.value]
    return spec.params


def _is_fermionic(bath) -> bool:
    if isinstance(bath, FermionicBath):
        return True
    if isinstance(bath, BosonicBath):
        return False
    raise TypeError(f"Cannot determine whether bath object {bath!r} is bosonic or fermionic.")


class MethodBuilder:
    """Build simulation data structures for an OQSModel.

    Constructs the topology, bath representations, generator, and auxiliary
    objects required to run a particular simulation method.
    """
    def __init__(self, model: OQSModel):
        """Create a builder for an open quantum system model.

        :param model: Model to build simulations from.
        :type model: OQSModel
        """
        self.model = model

    def build(self, method: Union[str, Method], topology: Optional[TopoTree] = None, min_chi: int = 8, max_chi: int = 64, opdict: Optional[LabelledOperatorDictionary]=None, bath_placement: str = "attach", bath_weight: Optional[Callable[[BathSpec, SystemInfo], Dict[str, float]]] = None, jordan_wigner_ordering: Optional[Union[List[str], str]] = None, jordan_wigner_tol: float = 1e-15, **kwargs) -> BuildResult:
        """Build a simulation ready method of the model.  
        
        Generates a topology, expands the bath placeholders and explicit bath subtrees, constructs the system bath generator, proposes bond dimensions and creates any method specific auxiliary object.
        
        :param method: The simulation method to build for
        :type method: str or Method
        :param topology: Optional user-supplied system topology.
        :type topology: TopoTree, optional
        :param min_chi: Minimum proposed bond dimension, defaults to 8
        :type min_chi: int, optional
        :param max_chi: Maximum proposed bond dimension, defaults to 64
        :type max_chi: int, optional
        :param opdict: Optional operator dictionary used during Liouville-space promotion.
        :type opdict: Optional[LabelledOperatorDictionary]
        :param jordan_wigner_ordering: jordan_wigner_ordering: Jordan-Wigner ordering for fermionic models.
        :type jordan_wigner_ordering: list[str] or {"tree", "auto"}, optional
        :param jordan_wigner_tol: Numerical tolerance for pruning small terms when applying the Jordan-Wigner transform, defaults to 1e-15
        :type jordan_wigner_tol: float, optional
        :param bath_placement: Bath-placement strategy used when proposing a topology.
        :type bath_placement: {"attach", "joint"}, optional
        :param bath_weight: Bath-system edge-weight function used in ``"joint"`` mode.
        :type bath_weight: Callable, optional
        :param `**kwargs`: Additional arguments forwarded to topology generation.
        :return: The built topology, capacity tree, generator and supporting data
        :rtype: BuildResult
        """
        model = self.model
        model.validate()
        method = _normalise_method(method)

        for spec in model.baths:
            if spec.nchannels() not in (1, 2):
                raise NotImplementedError(f"Bath '{spec.tag}' has {spec.nchannels()} coupling operators; only 1 (symmetric) or 2 (asymmetric raising/lowering) coupling operators are supported.")

        promoted, effective_system_info, effective_generator = self._resolve_effective_model(method)

        if topology is not None:
            full_topology = attach_bath_placeholders(model, topology)
        elif bath_placement == "attach":
            system_topology = propose_system_topology(model, system_info=effective_system_info, system_generator=effective_generator, **kwargs)
            full_topology = attach_bath_placeholders(model, system_topology)
        elif bath_placement == "joint":
            # built once for the joint topology proposal, which needs each
            # bath's coupling weight to place it in the system interaction
            # graph - bond dimensions are proposed separately below, from the
            # final fully-expanded tree.
            graph = build_joint_interaction_graph(model, sysinfo=effective_system_info, system_generator=effective_generator, bath_weight=bath_weight)
            full_topology = propose_joint_topology(model, system_info=effective_system_info, system_generator=effective_generator, graph=graph, **kwargs)
        else:
            raise ValueError(f"Unknown bath_placement '{bath_placement}'; expected 'attach' or 'joint'.")

        topo_min = TopoTree(ntree(full_topology.tree), list(full_topology.leaf_labels))
        topo_max = TopoTree(ntree(full_topology.tree), list(full_topology.leaf_labels))

        raw_baths: Dict[str, object] = {}
        bath_leaf_labels: Dict[str, List[str]] = {}
        pinned_min: set = set()
        pinned_max: set = set()
        for spec in model.baths:
            raw_bath = self._build_raw_bath(spec, method)
            raw_baths[spec.tag] = raw_bath

            params = _resolve_params(spec, method)
            degree = params.get("degree", _DEFAULT_DEGREE[method])
            lhd = params.get("lhd", None)
            chi0 = params.get("chi0")
            chi = params.get("chi")

            # use bath specific chis if present else default to 2
            labels, path_min = self._expand_bath(topo_min, spec.tag, raw_bath, degree, chi0 if chi0 is not None else 2, lhd)
            _, path_max = self._expand_bath(topo_max, spec.tag, raw_bath, degree, chi if chi is not None else 2, lhd, labels=labels)
            bath_leaf_labels[spec.tag] = labels

            if chi0 is not None:
                pinned_min.update(self._subtree_paths(topo_min.tree, path_min))
            if chi is not None:
                pinned_max.update(self._subtree_paths(topo_max.tree, path_max))

        merged_sysinfo = self._merged_system_info(effective_system_info, model.baths, raw_baths, bath_leaf_labels, method)

        ordering = topo_min.leaf_order()
        raw = merged_sysinfo.build_system_modes(ordering)
        site_map = raw["primitive_label_to_index"]
        nmodes = len(site_map)

        H = effective_generator.compile(site_map, nmodes)

        for spec in model.baths:
            raw_bath = raw_baths[spec.tag]
            prim_labels = self._bath_primitive_labels(raw_bath, bath_leaf_labels[spec.tag], method)
            binds = [site_map[lx] for lx in prim_labels]
            params = _resolve_params(spec, method)

            if method in _LIOUVILLE_METHODS:
                Sp, Sm = self._compile_liouville_coupling(spec, model, effective_system_info, promoted, site_map, nmodes, opdict)
                H = raw_bath.add_system_bath_generator(H, Sp, Sm=Sm, method=method.value, binds=binds)
            else:
                Sp = self._compile_hilbert(spec.coupling_ops[0], site_map, nmodes)
                Sm = self._compile_hilbert(spec.coupling_ops[1], site_map, nmodes) if spec.nchannels() == 2 else None
                geom = params.get("geom", _DEFAULT_GEOM[method])
                H = raw_bath.add_system_bath_hamiltonian(H, Sp, Sm, geom=geom, binds=binds)

        #propose bond dimensions using the bond-cut metrics on the fully expanded tree
        index_to_label = {idx: label for label, idx in site_map.items()}
        full_graph = build_interaction_graph(lCSOP(H, index_to_label), merged_sysinfo)

        for i, path in enumerate(topo_min.tree.leaf_indices()):
            topo_min.tree.at(path).value = i
        M, local_dims = bond_dimension_inputs(topo_min, merged_sysinfo, full_graph)
        metrics = compute_tree_cut_metrics(topo_min.tree, M, local_dims)
        bond_dims = propose_bond_dimensions(metrics, min_chi=min_chi, max_chi=max_chi)

        self._set_system_node_values(topo_min, merged_sysinfo, bond_dims, "min_chi", skip_paths=pinned_min)
        self._set_system_node_values(topo_max, merged_sysinfo, bond_dims, "max_chi", skip_paths=pinned_max)

        # Enforce that the proposed bond dimension trees are valid
        ntreeBuilder.sanitise(topo_min.tree)
        ntreeBuilder.sanitise(topo_max.tree)

        # if we are dealing with liouville space methods we need to build the trace tree for evaluating expectation values.
        trace_state = None
        if method in _LIOUVILLE_METHODS:
            trace_state = self._build_trace_state(topo_min, raw_baths, bath_leaf_labels, method)

        resolved_jw_ordering = None
        if any(_is_fermionic(spec.bath) for spec in model.baths):
            resolved_jw_ordering = self._resolve_jordan_wigner_ordering(jordan_wigner_ordering, ordering)
            H = lCSOP(H, index_to_label).jordan_wigner(resolved_jw_ordering, merged_sysinfo, tol=jordan_wigner_tol).compile(site_map, nmodes)

        return BuildResult(generator=H, system_modes=raw["system_modes"], topology=topo_min, capacity=topo_max, site_map=site_map, baths=raw_baths, trace_state=trace_state, jordan_wigner_ordering=resolved_jw_ordering, system_info=merged_sysinfo)

    @staticmethod
    def _resolve_jordan_wigner_ordering(jordan_wigner_ordering, tree_leaf_order: List[str]) -> List[str]:
        """Resolve a Jordan-Wigner ordering specification into a label ordering."""
        if jordan_wigner_ordering is None:
            raise ValueError("Jordan_wigner_ordering=[...] must be supplied (or 'tree' to reuse the topology's own leaf order).")
        if jordan_wigner_ordering == "tree":
            return list(tree_leaf_order)
        if jordan_wigner_ordering == "auto":
            raise NotImplementedError("Automatic Jordan-Wigner ordering estimation is not yet implemented; pass an explicit ordering list, or 'tree' to reuse the topology leaf order.")
        return list(jordan_wigner_ordering)

    def _build_trace_state(self, topo_min: TopoTree, raw_baths: Dict[str, object], bath_leaf_labels: Dict[str, List[str]], method: Method):
        """Build the rank-1 trace/identity state for Liouville-space methods."""
        trace_topology = TopoTree(self._collapse_bond_dimensions(topo_min.tree), list(topo_min.leaf_labels))

        system_labels = set(self.model.system_info.composite_labels())
        leaf_to_bath = {label: tag for tag, labels in bath_leaf_labels.items() for label in labels}
        bath_state_iters = {tag: iter(raw_baths[tag].identity_product_state(method=method.value)) for tag in bath_leaf_labels}

        product = []
        for label in trace_topology.leaf_order():
            if label in system_labels:
                d = self.model.system_info.local_dim(label)
                product.append(np.identity(d, dtype=np.complex128).flatten())
            else:
                product.append(next(bath_state_iters[leaf_to_bath[label]]))

        trace_state = ttn(trace_topology.tree, dtype=np.complex128)
        trace_state.set_product(product)
        return trace_state

    @staticmethod
    def _collapse_bond_dimensions(tree):
        """Return a copy of a tree with all internal bond dimensions set to 1"""
        collapsed = ntree(tree)
        for node in collapsed.dfs():
            if not node.is_leaf():
                node.value = 1
        return collapsed

    def _resolve_effective_model(self, method: Method):
        """Return the system information and generator used for the build.

        Automatically promotes Hilbert-space models to Liouville space when required by the selected method.
        """
        model = self.model
        if method in _LIOUVILLE_METHODS:
            if model.hilbert_space():
                lsys = model.system_info.liouville_space(grouping="paired", suffix="~")
                generator, _ = SuperOp.commutator(model.system_generator, model.system_info, lsys)
                return True, lsys, generator
            return False, model.system_info, model.system_generator

        if not model.hilbert_space():
            raise ValueError(f"Method '{method.value}' requires a Hilbert-space OQSModel.")
        return False, model.system_info, model.system_generator

    def _build_raw_bath(self, spec: BathSpec, method: Method):
        params = _resolve_params(spec, method)
        decomposition = params.get("decomposition")
        if decomposition is None:
            raise ValueError(f"Bath '{spec.tag}' is missing a 'decomposition' parameter for method '{method.value}'.")
        truncation = params.get("truncation")
        fermionic = _is_fermionic(spec.bath)
        channels = params.get("channels")

        if channels is not None and channels != "filled_empty":
            raise ValueError(f"Bath '{spec.tag}': unknown channels '{channels}'; expected 'filled_empty' or None.")
        if channels is not None and not fermionic:
            raise ValueError(f"Bath '{spec.tag}': channels='{channels}' is only supported for fermionic baths.")

        if channels == "filled_empty" and method in (Method.UNITARY, Method.TEDOPA):
            Ef = params.get("Ef", 0.0)
            decomp_filled, decomp_empty = decomposition if isinstance(decomposition, tuple) else (decomposition, decomposition)
            gf, wf = spec.bath.discretise(decomp_filled, Ef=Ef, sigma="+")
            ge, we = spec.bath.discretise(decomp_empty, Ef=Ef, sigma="-")
            raw_bath = SplitDiscreteFermionicBath(gf, wf, ge, we, attachment=params.get("attachment", "branch"), ordering=params.get("ordering", "filled_first"))
        elif channels == "filled_empty":
            raise NotImplementedError(f"Bath '{spec.tag}': channels='filled_empty' is not yet supported for method '{method.value}'.")
        elif method in (Method.UNITARY, Method.TEDOPA):
            g, w = spec.bath.discretise(decomposition)
            raw_bath = DiscreteFermionicBath(g, w) if fermionic else DiscreteBosonicBath(g, w)
        else:
            if fermionic and method is Method.HEOM:
                raise NotImplementedError(f"Bath '{spec.tag}': fermionic HEOM baths are not yet supported by the underlying pyTTN bath machinery.")
            dk, zk = spec.bath.expfit(decomposition)[:2]
            raw_bath = ExpFitFermionicBath(dk, zk) if fermionic else ExpFitBosonicBath(dk, zk)

        if truncation is not None:
            raw_bath.truncate_modes(truncation)
        else:
            raw_bath.truncate_modes()
        raw_bath.system_information()
        return raw_bath

    def _expand_bath(self, topology: TopoTree, tag: str, raw_bath, degree: int, chi: int, lhd, labels: Optional[List[str]] = None) -> tuple:
        """Replace a bath placeholder leaf with an explicit bath subtree.
        
        :return: Bath leaf labels and the attachment-node path.
        :rtype: tuple[list[str], list[int]]
        """
        pos = topology.leaf_order().index(tag)
        path = topology.tree.leaf_indices()[pos]
        node = topology.tree.at(path)
        # this placeholder's own value was a leaf-dimension placeholder; once it gains
        # children it instead represents the bond dimension of the system<->bath cut.
        node.value = chi
        # add_bath_tree's own return value only reports leaves under the first of the
        # newly created top-level branches when the tree-builder splits directly at the
        # attachment node (e.g. degree>1 with no pre-existing children) - the tree
        # itself is built correctly, but the reported indices are incomplete. Since
        # `node` was a pure leaf before this call, every leaf now beneath it is new.
        raw_bath.add_bath_tree(node, degree, chi, lhd)
        n_new = len(node.leaf_indices())

        if labels is None:
            labels = [f"{tag}_{i}" for i in range(n_new)]
        elif len(labels) != n_new:
            raise RuntimeError(f"Bath '{tag}' produced {n_new} modes here but {len(labels)} previously; " "topology and capacity trees are inconsistent.")

        topology.leaf_labels[pos:pos + 1] = labels
        return labels, path

    @staticmethod
    def _subtree_paths(tree, root_path) -> set:
        """Return the (absolute) paths of every node - internal and leaf - at or below ``root_path``."""
        root_path = tuple(root_path)
        node = tree.root() if len(root_path) == 0 else tree.at(list(root_path))
        paths = set()

        def walk(n, prefix):
            paths.add(prefix)
            for i in range(n.size()):
                walk(n.at(i), prefix + (i,))

        walk(node, root_path)
        return paths

    def _set_system_node_values(self, topology: TopoTree, sysinfo: SystemInfo, bond_dims: dict, key: str, skip_paths: Optional[set] = None) -> None:
        skip_paths = skip_paths or set()
        leaf_paths = topology.leaf_paths()
        path_to_label = {tuple(p): lx for lx, p in leaf_paths.items()}
        for node in topology.tree.dfs():
            path = tuple(node.index())
            if node.is_leaf():
                label = path_to_label.get(path)
                if label is not None and label in sysinfo:
                    node.value = sysinfo.local_dim(label)
                continue
            if path in skip_paths:
                continue
            if path in bond_dims:
                node.value = bond_dims[path][key]

    def _bath_primitive_groups(self, raw_bath, method: Method) -> List[List[int]]:
        """Return the primitive-mode indices represented by each bath leaf."""
        if method in _LIOUVILLE_METHODS:
            return raw_bath._composite_modes
        return [[i] for i in range(len(raw_bath.primitive_mode_dims))]

    def _bath_primitive_labels(self, raw_bath, leaf_labels: List[str], method: Method) -> List[str]:
        """Flatten a bath's leaf labels into per-primitive labels, in raw mode order."""
        groups = self._bath_primitive_groups(raw_bath, method)
        flat: List[str] = []
        for leaf_label, group in zip(leaf_labels, groups):
            if len(group) == 1:
                flat.append(leaf_label)
            else:
                flat.extend(f"{leaf_label}_{j}" for j in range(len(group)))
        return flat

    def _merged_system_info(self, system_info: SystemInfo, bath_specs, raw_baths, bath_leaf_labels, method: Method) -> SystemInfo:
        merged = SystemInfo(system_info.as_dict())
        for spec in bath_specs:
            raw_bath = raw_baths[spec.tag]
            leaf_labels = bath_leaf_labels[spec.tag]
            groups = self._bath_primitive_groups(raw_bath, method)
            prim_dims = raw_bath.mode_dims if method in _LIOUVILLE_METHODS else raw_bath.primitive_mode_dims
            fermionic = _is_fermionic(spec.bath)

            for leaf_label, group in zip(leaf_labels, groups):
                if len(group) == 1:
                    dim = prim_dims[group[0]]
                    merged[leaf_label] = fermion_mode() if fermionic else boson_mode(dim)
                else:
                    prims = {}
                    for j, p in enumerate(group):
                        dim = prim_dims[p]
                        prims[f"{leaf_label}_{j}"] = fermion_mode() if fermionic else boson_mode(dim)
                    merged[leaf_label] = prims
        return merged

    def _compile_hilbert(self, op: lCSOP, site_map: Dict[str, int], nmodes: int):
        return op.expand().compile(site_map, nmodes, backend="ssop")

    def _compile_liouville_pair(self, op: lCSOP, hsys: SystemInfo, lsys: SystemInfo, site_map: Dict[str, int], nmodes: int, opdict):
        left, _ = SuperOp.left(op, hsys, lsys, opdict)
        right, _ = SuperOp.right(op, hsys, lsys, opdict)
        return [self._compile_hilbert(left, site_map, nmodes), self._compile_hilbert(right, site_map, nmodes)]

    def _compile_liouville_coupling(self, spec: BathSpec, model: OQSModel, effective_system_info: SystemInfo, promoted: bool, site_map: Dict[str, int], nmodes: int, opdict):
        if promoted:
            Sp = self._compile_liouville_pair(spec.coupling_ops[0], model.system_info, effective_system_info, site_map, nmodes, opdict)
            Sm = self._compile_liouville_pair(spec.coupling_ops[1], model.system_info, effective_system_info, site_map, nmodes, opdict) if spec.nchannels() == 2 else Sp
            return Sp, Sm

        # advanced/manual path: model.representation was already LIOUVILLE, so
        # coupling_ops are expected to already be expressed in Liouville space,
        # supplied positionally as [left-acting, right-acting].
        if spec.nchannels() != 2:
            raise ValueError(f"Bath '{spec.tag}': for a pre-built Liouville-space OQSModel, coupling_ops must contain exactly 2 operators [left-acting, right-acting].")
        Sp = [self._compile_hilbert(spec.coupling_ops[0], site_map, nmodes), self._compile_hilbert(spec.coupling_ops[1], site_map, nmodes)]
        return Sp, Sp
