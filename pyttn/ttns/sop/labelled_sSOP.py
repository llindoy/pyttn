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

from .sSOPExt import OPBase, sOP, sSOP, sNBO
from .SOPExt import SOP
from .system_information import SystemInfo

from typing import Dict, Optional, TYPE_CHECKING, Union, Set

if TYPE_CHECKING:
    from .labelled_SOP import lCSOP

import copy

import numpy as np


def _merge_labels(a, b):
    labels = a.sites() | b.sites()
    site_map = { label: i for i, label in enumerate(sorted(labels)) }
    idx_to_label = { i: label for label, i in site_map.items() }
    return site_map, idx_to_label

class lSOP:
    """A container for symbolic operator expressions with label metadata.

    This class stores an operator expression together with a mapping from placeholder indices to user-defined site labels. It is used as an intermediate representation prior to compilation into a physical SOP.
    """

    def __init__(self, expr : OPBase, index_to_label : Dict[int, str]):
        """Initialise a labelled operator object.

        :param expr: The operator expression
        :type expr: OPBase
        :param index_to_label: Mapping from placeholder indices to site labels
        :type index_to_label: dict[int, str]
        """
        self._expr = sSOP(expr)
        self.index_to_label = index_to_label

    def sites(self) -> Set[str]:
        """Return the set of site labels appearing in the operator.

        :return: The set of site labels
        :rtype: set[str]
        """

        return set(self.index_to_label.values())

    @property
    def expr(self) -> sSOP:
        """Return the underlying operator expression.

        :return: The operator expression
        :rtype: sSOP
        """
        return self._expr

    @property
    def dtype(self) -> np.dtype:
        return self._expr.dtype

    def complex_dtype(self) -> bool:
        return self._expr.complex_dtype()

    def __repr__(self):
        """Return a string representation of the labelled operator.

        :return: String representation of the operator
        :rtype: str
        """
        return f"LabelledSOP({self._expr}, sites={self.sites()})"

    def to_sSOP(self, site_map : Dict[str, int]) -> sSOP:
        """Construct a concrete sSOP with physical indices from the current lSOP object

        This function replaces all placeholder indices in a labelled operator with the corresponding physical indices defined by `site_map`.

        :param site_map: Mapping from site labels to physical mode indices
        :type site_map: dict[str, int]
        :return: A relabelled sum-of-product operator acting on physical indices
        :rtype: sSOP

        :raises ValueError: If a site label in the operator is not found in `site_map`
        """
        new_sop = sSOP()
        for term in self._expr:
            new_nbo = sNBO()
            new_nbo.coeff = copy.deepcopy(term.coeff)

            for op in term.ops:
                site_label = self.index_to_label[op.mode]

                if site_label not in site_map:
                    raise ValueError(f"Site {site_label} not in system mapping")

                new_idx = site_map[site_label]
                new_op = sOP(op.op, new_idx, op.fermionic)
                new_nbo.insert_back(new_op)

            new_sop += new_nbo
        return new_sop

    def to_SOP(self, site_map : Dict[str, int], nmodes: int) -> SOP:
        """Construct a concrete SOP with physical indices from the current lSOP object

        This function replaces all placeholder indices in a labelled operator with the corresponding physical indices defined by `site_map`.

        :param site_map: Mapping from site labels to physical mode indices
        :type site_map: dict[str, int]
        :param nmodes: The number of modes that this operator acts upon
        :type nmodes: int
        :return: A relabelled sum-of-product operator acting on physical indices
        :rtype: SOP

        :raises ValueError: If a site label in the operator is not found in `site_map`
        """
        sop = SOP(nmodes)
        for term in self._expr:
            coeff = term.coeff
            new_nbo = sNBO()
            new_nbo.coeff = coeff
            for op in term.ops:
                site_label = self.index_to_label[op.mode]

                if site_label not in site_map:
                    raise ValueError(f"Site {site_label} not in system mapping")

                new_idx = site_map[site_label]
                new_op = sOP(op.op, new_idx, op.fermionic)
                new_nbo.insert_back(new_op)

            sop += new_nbo
        return sop

    def compile(self, site_map: Dict[str, int], nmodes: Optional[int] = None, backend : str ="sop") -> Union[sSOP, SOP]:
        """Compile the labelled operator into a concrete backend representation.

        This method converts the symbolic operator expression stored in the lSOP object into a concrete operator with physical indices according to the provided site mapping. The output format is controlled by the `backend` argument.

        Two backend representations are supported:

        - "sop": Returns a compact SOP object. This is the recommended option for large systems due to its efficiency.
        - "ssop": Returns a symbolic sSOP object.

        :param site_map: Mapping from site labels to physical mode indices
        :type site_map: dict[str, int]
        :param nmodes: The total number of modes in the system.
        :type nmodes: int, optional
        :param backend: The backend representation to compile to. Supported
                        options are "sop" and "ssop". Defaults to "sop".
        :type backend: str, optional

        :return: The compiled operator in the requested backend format
        :rtype: SOP or sSOP

        :raises ValueError:
            - If `backend` is not recognised
        """
        if backend == "sop":
            if nmodes is None:
                return self.to_SOP(site_map, len(site_map))
            return self.to_SOP(site_map, nmodes)
        elif backend == "ssop":
            return self.to_sSOP(site_map)
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    def jordan_wigner(self, ordering: list[str], sysinfo : SystemInfo, tol: float = 1e-15):
        """Perform a Jordan-Wigner transform using a user-specified label ordering.

        This method applies the Jordan-Wigner transformation according to a given ordering of site labels, then converts the result back into a labelled operator representation.

        :param ordering: List of site labels defining fermionic ordering
        :type ordering: list[str]
        :param sysinfo: System information object consistent with the ordering
        :type sysinfo: SystemInfo
        :param tol: numerical tolerance for pruning small terms
        :type tol: float

        :return: A new lSOP with Jordan-Wigner transformation applied
        :rtype: lSOP
        """

        # Build temporary label → index map
        jw_map = {label: i for i, label in enumerate(ordering)}

        # Validate
        missing = self.sites() - set(ordering)
        if missing:
            raise ValueError(f"Missing labels in ordering: {missing}")

        sys = sysinfo.build_flattened_modes(ordering)
        nmodes = sys.nmodes()

        # Compile to SOP in JW index space
        sop = self.to_SOP(jw_map, nmodes)
        # Apply JW transform
        sop = sop.jordan_wigner(sys, tol)
        # Convert back to labelled representation
        index_to_label = {i: label for label, i in jw_map.items()}
        new_expr = sSOP()

        for term in sop.expand():
            new_nbo = sNBO()
            new_nbo.coeff = term.coeff
            for op in term.ops:
                idx = op.mode

                if idx not in index_to_label:
                    raise ValueError(f"Index {idx} not in JW mapping")

                # reuse label index from original builder structure
                new_op = sOP(op.op, idx, op.fermionic)
                new_nbo.insert_back(new_op)
            new_expr +=  new_nbo

        return lSOP(new_expr, index_to_label)


    def to_lCSOP(self, ordering: Optional[list[str]] = None) -> "lCSOP":
        """
        Convert this symbolic labelled SOP (lSOP) into a compact labelled SOP (lCSOP) using a specified ordering of labels.

        :param ordering: Ordered list of site labels
        :type ordering: list[str], optional

        :return: Compiled labelled compact SOP
        :rtype: lCSOP
        """
        # deferred import to avoid a module-level circular dependency between
        # lSOP (this module) and lCSOP (.labelled_SOP), which construct each other.
        from .labelled_SOP import lCSOP

        if ordering is None:
            ordering = [ self.index_to_label[i] for i in sorted(self.index_to_label) ]

        missing = self.sites() - set(ordering)
        if missing:
            raise ValueError(f"Missing labels in ordering: {missing}")

        site_map = {label: i for i, label in enumerate(ordering)}
        nmodes = len(ordering)
        sop = self.to_SOP(site_map, nmodes)

        index_to_label = {i: label for label, i in site_map.items()}
        return lCSOP(sop, index_to_label)



    def __add__(self, other : Union["lSOP", "lCSOP"]) -> "lSOP":
        """
        Add two labelled operator expressions.
        The site-label mappings of both operators are merged and the underlying symbolic expressions are combined.
        If ``other`` is an ``lCSOP`` object, it is first expanded into its symbolic representation.

        :param other: Operator to add
        :type other: lSOP or lCSOP
        :return: Sum of the two labelled operators
        :rtype: lSOP
        """
        # deferred import - see note in to_lCSOP()
        from .labelled_SOP import lCSOP

        if isinstance(other, lCSOP):
            other = other.expand()

        if not isinstance(other, lSOP):
            return NotImplemented

        site_map, idx_to_label = _merge_labels(self, other)

        expr = self.to_sSOP(site_map)
        expr += other.to_sSOP(site_map)

        return lSOP(expr, idx_to_label)

    def __iadd__(self, other : Union["lSOP", "lCSOP"]) -> "lSOP":
        """
        Add another labelled operator in-place.

        :param other: Operator to add
        :type other: lSOP or lCSOP
        :return: Updated operator
        :rtype: lSOP
        """
        tmp = self + other
        self._expr = tmp._expr
        self.index_to_label = tmp.index_to_label
        return self

    def __sub__(self, other : Union["lSOP", "lCSOP"]) -> "lSOP":
        """
        Subtract another labelled operator.

        :param other: Operator to subtract
        :type other: lSOP or lCSOP
        :return: Difference of the two operators
        :rtype: lSOP
        """
        return self + (-other)

    def __isub__(self, other : Union["lSOP", "lCSOP"]) -> "lSOP":
        """
        Subtract another labelled operator in-place.

        :param other: Operator to add
        :type other: lSOP or lCSOP
        :return: Updated operator
        :rtype: lSOP
        """
        return self.__iadd__(-other)

    def __neg__(self) -> "lSOP":
        """
        Return the additive inverse of the operator.

        :return: Negated operator
        :rtype: lSOP
        """
        return self * (-1)

    def __mul__(self, other : Union[int, float, complex, "lSOP"]) -> "lSOP":
        """
        Multiply the operator by a scalar or another operator

        :param other: Scalar or other operator multiplier
        :type other: int, float, complex, numpy scalar or lSOP
        :return: Scaled operator
        :rtype: lSOP
        """
        if np.isscalar(other):
            sop = copy.deepcopy(self._expr)
            sop *= other
            return lSOP(sop,self.index_to_label.copy())
        elif isinstance(other, lSOP):
            return self@other
        return NotImplemented

    def __rmul__(self, other :  Union[int, float, complex]) -> "lSOP":
        """
        Multiply the operator by a scalar from the left

        :param other: Scalar multiplier
        :type other: int, float, complex, or numpy scalar
        :return: Scaled operator
        :rtype: lSOP
        """
        if np.isscalar(other):
            return self * other
        else:
            return other @ self

    def __truediv__(self, other : Union[int, float, complex]) -> "lSOP":
        """
        Divide the operator by a scalar
        :param other: Scalar multiplier
        :type other: int, float, complex, or numpy scalar
        :return: Scaled operator
        :rtype: lSOP
        """
        if not np.isscalar(other):
            return NotImplemented
        return self * (1.0 / other)

    def __matmul__(self, other : "lSOP") -> "lSOP":
        """
        Form the operator product of two labelled operators.
        The site-label mappings of both operators are merged

        :param other: Operator to multiply with
        :type other: lSOP
        :return: Operator product
        :rtype: lSOP
        """
        if not isinstance(other, lSOP):
            return NotImplemented

        if self.index_to_label == other.index_to_label:
            return lSOP(self._expr * other._expr, self.index_to_label.copy())

        site_map, idx_to_label = _merge_labels(self, other)
        expr_a = self.to_sSOP(site_map)
        expr_b = other.to_sSOP(site_map)

        expr = expr_a * expr_b

        return lSOP(expr, idx_to_label)
