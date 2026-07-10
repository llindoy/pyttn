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

from .sSOPExt import sOP, sNBO
from .SOPExt import SOP
from .system_information import SystemInfo
from .labelled_sSOP import lSOP, _merge_labels

from typing import Dict, Optional, Union, Set

import copy

import numpy as np

class lCSOP:
    """
    Container for compiled labelled SOP operators.

    This class stores a compact SOP together with a mapping from
    mode indices to user-defined site labels. It provides efficient
    execution and avoids symbolic overhead.
    """

    def __init__(self, sop: SOP, index_to_label: Dict[int, str]):
        """
        Initialise a labelled compact SOP.

        :param sop: The compiled SOP operator
        :type sop: SOP
        :param index_to_label: Mapping from mode indices to labels
        :type index_to_label: dict[int, str]
        """
        self._expr = sop
        self.index_to_label = index_to_label

    def sites(self) -> Set[str]:
        """
        Return the set of site labels appearing in the operator.

        :return: The set of site labels
        :rtype: set[str]
        """
        return set(self.index_to_label.values())

    def nmodes(self) -> int:
        """
        Return the number of modes.

        :return: Number of modes
        :rtype: int
        """
        return self._expr.nmodes()

    def nterms(self) -> int:
        """
        Return the number of terms.

        :return: Number of terms
        :rtype: int
        """
        return self._expr.nterms()

    @property
    def dtype(self) -> np.dtype:
        return self._expr.dtype

    def complex_dtype(self) -> bool:
        return self._expr.complex_dtype()

    def backend(self) -> str:
        return self._expr.backend()


    @property
    def operator_dictionary(self):
        return self._expr.get_operator_dictionary()

    def set_operator_dictionary(self, opdict):
        self._expr.set_operator_dictionary(opdict)


    def jordan_wigner(self, ordering: list[str], sysinfo: SystemInfo, tol: float = 1e-15) -> "lCSOP":
        """Perform a Jordan-Wigner transform using a user-specified label ordering.

        This method applies the Jordan-Wigner transformation according to a given ordering of site labels, then converts the result back into a labelled operator representation.

        :param ordering: List of site labels defining fermionic ordering
        :type ordering: list[str]
        :param sysinfo: System information object consistent with the ordering
        :type sysinfo: SystemInfo
        :param tol: numerical tolerance for pruning small terms
        :type tol: float
        :return: transformed lCSOP
        :rtype: lCSOP
        """
        missing = self.sites() - set(ordering)
        if missing:
            raise ValueError(f"Missing labels in ordering: {missing}")

        sys = sysinfo.build_flattened_modes(ordering)
        nmodes = sys.nmodes()


        jw_map = {label: i for i, label in enumerate(ordering)}
        nmodes = sys.nmodes()
        sop = self.to_SOP(jw_map, nmodes)
        sop = sop.jordan_wigner(sys, tol)
        index_to_label = {i: label for label, i in jw_map.items()}

        return lCSOP(sop, index_to_label)


    def prune_zeros(self, tol: float = 1e-15):
        """
        Remove small terms.

        :param tol: tolerance
        :type tol: float
        """
        self._expr.prune_zeros(tol)

    def expand(self) -> "lSOP":
        """
        Expand into symbolic representation.

        :return: labelled symbolic operator
        :rtype: lSOP
        """
        ssop = self._expr.expand()
        return lSOP(ssop, self.index_to_label.copy())


    def to_SOP(self, site_map: Dict[str, int], nmodes: int = None) -> SOP:
        if nmodes is None:
            nmodes = len(site_map)
        new_sop = SOP(nmodes)

        opdict = self._expr.get_operator_dictionary()
        for term, coeff in self._expr:
            # convert prodOP to sPOP (C++)

            pop = term.as_sPOP(opdict)

            new_nbo = sNBO()
            new_nbo.coeff = coeff

            for op in pop:
                idx = op.mode

                if idx not in self.index_to_label:
                    raise ValueError(f"Index {idx} not in label mapping")

                label = self.index_to_label[idx]

                if label not in site_map:
                    raise ValueError(f"Site {label} not in mapping")

                new_idx = site_map[label]

                new_op = sOP(op.op, new_idx, op.fermionic)
                new_nbo.insert_back(new_op)

            new_sop += new_nbo

        return new_sop

    def compile(self, site_map: Dict[str, int], nmodes: Optional[int] = None) -> Union[SOP]:
        """Compile the labelled operator into a concrete backend representation.

        This method converts the symbolic operator expression stored in the lCSOP object into a concrete operator with physical indices according to the provided site mapping. 

        :param site_map: Mapping from site labels to physical mode indices
        :type site_map: dict[str, int]
        :param nmodes: The total number of modes in the system.
        :type nmodes: int, optional
        :return: The compiled operator in the requested backend format
        :rtype: SOP

        """
        if nmodes is None:
            return self.to_SOP(site_map, len(site_map))
        return self.to_SOP(site_map, nmodes)

    @property
    def expr(self) -> SOP:
        """
        Return the underlying SOP.

        :return: SOP object
        :rtype: SOP
        """
        return self._expr

    def __repr__(self):
        return f"LabelledCSOP(nmodes={self.nmodes()}, nterms={self.nterms()}, sites={self.sites()})"

    def __str__(self):
        return str(self._expr)

    def __add__(self, other : "lCSOP") -> "lCSOP":
        """
        Add two labelled operator expressions.
        The site-label mappings of both operators are merged and the underlying symbolic expressions are combined.

        :param other: Operator to add
        :type other:  lCSOP
        :return: Sum of the two labelled operators
        :rtype: lCSOP
        """
        if not isinstance(other, lCSOP):
            return NotImplemented

        site_map, idx_to_label = _merge_labels(self, other)

        nmodes = len(site_map)

        expr = self.to_SOP(site_map, nmodes=nmodes)
        expr += other.to_SOP(site_map, nmodes=nmodes)

        return lCSOP(expr, idx_to_label)

    def __iadd__(self, other : "lCSOP") -> "lCSOP":
        """
        Add another labelled operator in-place.

        :param other: Operator to add
        :type other: lCSOP
        :return: Updated operator
        :rtype: lCSOP
        """
        tmp = self + other
        self._expr = tmp._expr
        self.index_to_label = tmp.index_to_label
        return self

    def __sub__(self, other : "lCSOP") -> "lCSOP":
        """
        Subtract another labelled operator.

        :param other: Operator to subtract
        :type other: lCSOP
        :return: Difference of the two operators
        :rtype: lCSOP
        """
        return self + (-other)

    def __isub__(self, other : "lCSOP") -> "lCSOP":
        """
        Subtract another labelled operator in-place.

        :param other: Operator to add
        :type other: lCSOP
        :return: Updated operator
        :rtype: lCSOP
        """
        return self.__iadd__(-other)

    def __neg__(self) -> "lCSOP":
        """
        Return the additive inverse of the operator.

        :return: Negated operator
        :rtype: lCSOP
        """
        return self * (-1)

    def __mul__(self, other : Union[int, float, complex, "lSOP", "lCSOP"]) -> "lCSOP":
        """
        Multiply the operator by a scalar or another operator

        :param other: Scalar or other operator multiplier
        :type other: int, float, complex, numpy scalar, lSOP or lCSOP
        :return: Scaled operator
        :rtype: lSOP
        """
        if np.isscalar(other):
            sop = copy.deepcopy(self._expr)
            sop *= other
            return lCSOP(sop,self.index_to_label.copy())
        elif isinstance(other, (lSOP, lCSOP)):
            return self @ other

        return NotImplemented

    def __rmul__(self, other : "lCSOP") -> "lCSOP":
        """
        Multiply the operator by a scalar or operator from the left

        :param other: Scalar or other operator multiplier
        :type other: int, float, complex, numpy scalar, lSOP or lCSOP
        :return: Scaled operator
        :rtype: lSOP
        """
        if np.isscalar(other):
            return self * other
        elif isinstance(other, (lSOP, lCSOP)):
            return other @ self

        return NotImplemented

    def __truediv__(self, other : Union[int, float, complex]) -> "lCSOP":
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


    def __matmul__(self, other : Union["lSOP", "lCSOP"]) -> "lCSOP":
        """
        Form the operator product of two labelled operators.
        The site-label mappings of both operators are merged

        :param other: Operator to multiply with
        :type other: lSOP or lCSOP
        :return: Operator product
        :rtype: lCSOP
        """

        if not isinstance(other, (lSOP, lCSOP)):
            return NotImplemented

        result = self.expand() @ (other.expand() if isinstance(other, lCSOP) else other)
        ordering = [result.index_to_label[i] for i in range(len(result.index_to_label))]
        return result.to_lCSOP(ordering)

    def __rmatmul__(self, other : "lSOP") -> "lCSOP":
        """
        Left multiply lCSOP by lSOP

        :param other: Operator to multiply with
        :type other: lSOP
        :return: Operator product
        :rtype: lCSOP
        """

        if not isinstance(other, lSOP):
            return NotImplemented

        result = other @ self.expand()
        ordering = [result.index_to_label[i] for i in range(len(result.index_to_label))]
        return result.to_lCSOP(ordering)
