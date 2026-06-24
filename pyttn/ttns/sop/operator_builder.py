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

from typing import Dict, Optional, Union

class OperatorBuilder:
    """A helper class for constructing symbolic operator expressions using placeholder indices.

    This class allows operators to be defined using site labels rather than explicit
    integer indices. Internally, each new site label is assigned a unique integer index,
    which is later replaced during compilation to match the physical system layout.
    """

    def __init__(self):
        """Initialise the operator builder.

        This sets up internal mappings between site labels and placeholder indices.
        """

        self.label_to_index = {}
        self.index_to_label = {}
        self.next_index = 0

    def __get_index(self, site : str ) -> int:
        """Return the placeholder index associated with a given site label.

        If the site label has not yet been encountered, a new placeholder index
        is assigned and stored.

        :param site: The site label
        :type site: str
        :return: The placeholder index associated with the site
        :rtype: int
        """

        if site not in self.label_to_index:
            idx = self.next_index
            self.label_to_index[site] = idx
            self.index_to_label[idx] = site
            self.next_index += 1
        return self.label_to_index[site]
    

    def op(self, op_label, site):
        """Construct a non-fermionic operator acting on a labelled site.

        This creates a symbolic operator using a placeholder index corresponding
        to the given site label.

        :param op_label: The operator label (e.g. 'sx', 'sz', 'n')
        :type op_label: str
        :param site: The site label
        :type site: str
        :return: A single-site operator acting on the placeholder index
        :rtype: sOP
        """

        idx = self.__get_index(site)
        return sOP(op_label, idx)

    def fop(self, op_label, site):
        """Construct a fermionic operator acting on a labelled site.

        This creates a fermionic operator with the appropriate statistics flag set.

        :param op_label: The operator label (e.g. 'c', 'cdag')

        :type op_label: str
        :param site: The site label
        :type site: str
        :return: A single-site operator acting on the placeholder index
        :rtype: sOP
        """

        idx = self.__get_index(site)
        return sOP(op_label, idx, True)
    
    def wrap(self, expr):
        """Wrap a symbolic operator expression with label metadata.

        This converts an expression into an lSOP object which stores both
        the operator expression and the mapping from placeholder indices
        to site labels.

        :param expr: The operator expression
        :type expr: OPBase
        :return: A labelled operator object
        :rtype: lSOP
        """

        return lSOP(expr, dict(self.index_to_label))


class lSOP:
    """A container for symbolic operator expressions with label metadata.

    This class stores an operator expression together with a mapping from
    placeholder indices to user-defined site labels. It is used as an
    intermediate representation prior to compilation into a physical SOP.
    """

    def __init__(self, expr : OPBase, index_to_label : Dict[int, str]):
        """Initialise a labelled operator object.

        :param expr: The operator expression
        :type expr: OPBase
        :param index_to_label: Mapping from placeholder indices to site labels
        :type index_to_label: dict[int, str]
        """
        self.expr = sSOP(expr)
        self.index_to_label = index_to_label

    def sites(self):
        """Return the set of site labels appearing in the operator.

        :return: The set of site labels
        :rtype: set[str]
        """

        return set(self.index_to_label.values())
    
    
    def __repr__(self):
        """Return a string representation of the labelled operator.

        :return: String representation of the operator
        :rtype: str
        """
        return f"LabelledSOP({self.expr}, sites={self.sites()})"

    def to_sSOP(self, site_map : Dict[str, int]) -> sSOP:
        """Construct a concrete sSOP with physical indices from the current lSOP OBJECT

        This function replaces all placeholder indices in a labelled operator
        with the corresponding physical indices defined by `site_map`.

        :param site_map: Mapping from site labels to physical mode indices
        :type site_map: dict[str, int]
        :return: A relabelled sum-of-product operator acting on physical indices
        :rtype: sSOP

        :raises ValueError: If a site label in the operator is not found in `site_map`
        """
        new_sop = sSOP()
        for term in self.expr:
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

            new_sop += new_nbo
        return new_sop
    
    def to_SOP(self, site_map : Dict[str, int], nmodes: int) -> SOP:
        """Construct a concrete SOP with physical indices from the current lSOP OBJECT

        This function replaces all placeholder indices in a labelled operator
        with the corresponding physical indices defined by `site_map`.

        :param site_map: Mapping from site labels to physical mode indices
        :type site_map: dict[str, int]
        :param nmodes: The number of modes that this operator acts upon
        :type nmodes: int
        :return: A relabelled sum-of-product operator acting on physical indices
        :rtype: SOP

        :raises ValueError: If a site label in the operator is not found in `site_map`
        """
        sop = SOP(nmodes)
        for term in self.expr:
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

        This method converts the symbolic operator expression stored in the lSOP
        object into a concrete operator with physical indices according to the
        provided site mapping. The output format is controlled by the `backend`
        argument.

        Two backend representations are supported:

        - "sop": Returns a compact SOP object. This is the recommended option
            for large systems due to its efficiency.
        - "ssop": Returns a symbolic sSOP object. 

        :param site_map: Mapping from site labels to physical mode indices
        :type site_map: dict[str, int]
        :param nmodes: The total number of modes in the system. This is required
                    when using the "sop" backend, as the SOP object must know
                    the size of the system it acts upon.
        :type nmodes: int, optional
        :param backend: The backend representation to compile to. Supported
                        options are "sop" and "ssop". Defaults to "sop".
        :type backend: str, optional

        :return: The compiled operator in the requested backend format
        :rtype: SOP or sSOP

        :raises ValueError:
            - If `backend == "sop"` and `nmodes` is not provided
            - If `backend` is not recognised
        """
        if backend == "sop":
            if nmodes is None:
                raise ValueError("nmodes must be provided for SOP backend")
            return self.to_SOP(site_map, nmodes)
        elif backend == "ssop":
            return self.to_sSOP(site_map)
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    def jordan_wigner(self, ordering: list[str], sys, tol: float = 1e-15):
        """Perform a Jordan-Wigner transform using a user-specified label ordering.

        This method applies the Jordan-Wigner transformation according to a given
        ordering of site labels, then converts the result back into a labelled
        operator representation.

        :param ordering: List of site labels defining fermionic ordering
        :type ordering: list[str]
        :param sys: system_modes object consistent with the ordering
        :type sys: system_modes
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

        # Compile to SOP in JW index space
        sop = self.to_SOP(jw_map, nmodes=sys.nmodes())

        # Apply JW transform
        sop = sop.jordan_wigner(sys, tol)
        print(sop)
        # Convert back to labelled representation
        index_to_label = {i: label for label, i in jw_map.items()}
        print(sop.expand())
        new_expr = sSOP()

        for term in sop.expand():
            coeff = term.coeff
            new_nbo = sNBO()
            print("tval:", term)
            for op in term.ops:
                idx = op.mode

                if idx not in index_to_label:
                    raise ValueError(f"Index {idx} not in JW mapping")

                # reuse label index from original builder structure
                new_op = sOP(op.op, idx, op.fermionic)
                new_nbo.insert_back(new_op)

            new_expr += coeff * new_nbo

        return lSOP(new_expr, index_to_label)
