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

from typing import Dict, Optional, Union, Set
from contextlib import contextmanager

import copy

from functools import wraps
import numpy as np
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
    
    def wrap(self, expr, N: int = None) -> Union['lSOP', 'lCSOP']:
        """Wrap a symbolic operator expression with label metadata.

        This converts an expression into an lSOP object which stores both
        the operator expression and the mapping from placeholder indices
        to site labels.

        :param expr: The operator expression
        :type expr: OPBase
        :param N: Optional number of modes 
        :type N: int, optional
        :return: A labelled operator object
        :rtype: Union[lSOP, lCSOP]
        """

        if N is None:
            return lSOP(expr, dict(self.index_to_label))
        else:
            if isinstance(expr, SOP):
                return lCSOP(expr, dict(self.index_to_label))
            else:
                sop = SOP(N)
                sop += expr
                return lCSOP(sop, dict(self.index_to_label))

def _merge_labels(a, b):
    labels = a.sites() | b.sites()
    site_map = { label: i for i, label in enumerate(sorted(labels)) }
    idx_to_label = { i: label for label, i in site_map.items() }
    return site_map, idx_to_label

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

    def sites(self) -> Set[str]:
        """Return the set of site labels appearing in the operator.

        :return: The set of site labels
        :rtype: set[str]
        """

        return set(self.index_to_label.values())
    
    def expr(self) -> sSOP:
        """Return the underlying operator expression.

        :return: The operator expression
        :rtype: sSOP
        """
        return self.expr

    @property
    def dtype(self) -> np.dtype:
        return self.expr.dtype

    def complex_dtype(self) -> bool:
        return self.expr.complex_dtype()


    def __repr__(self):
        """Return a string representation of the labelled operator.

        :return: String representation of the operator
        :rtype: str
        """
        return f"LabelledSOP({self.expr}, sites={self.sites()})"

    def to_sSOP(self, site_map : Dict[str, int]) -> sSOP:
        """Construct a concrete sSOP with physical indices from the current lSOP object

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

        This method applies the Jordan-Wigner transformation according to a given
        ordering of site labels, then converts the result back into a labelled
        operator representation.

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


    def to_lCSOP(self, ordering: list[str]) -> "lCSOP":
        """
        Convert this symbolic labelled SOP (lSOP) into a compact labelled SOP (lCSOP)
        using a specified ordering of labels.

        :param ordering: Ordered list of site labels
        :type ordering: list[str]

        :return: Compiled labelled compact SOP
        :rtype: lCSOP
        """

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
        self.expr = tmp.expr
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
    
    def __mul__(self, other : Union["lSOP", "lCSOP"]) -> "lSOP":
        """
        Multiply the operator by a scalar.

        :param other: Scalar multiplier
        :type other: int, float, complex, or numpy scalar
        :return: Scaled operator
        :rtype: lSOP
        """
        if np.isscalar(other):
            sop = copy.deepcopy(self.expr)
            sop *= other
            return lSOP(sop,self.index_to_label.copy())
        
        return NotImplemented
    
    def __rmul__(self, other : Union["lSOP", "lCSOP"]) -> "lSOP":
        """
        Multiply the operator by a scalar from the left

        :param other: Scalar multiplier
        :type other: int, float, complex, or numpy scalar
        :return: Scaled operator
        :rtype: lSOP
        """
        return self * other

    def __truediv__(self, other : Union["lSOP", "lCSOP"]) -> "lSOP":
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
            return lSOP(self.expr * other.expr, self.index_to_label.copy())

        site_map, idx_to_label = _merge_labels(self, other)
        expr_a = self.to_sSOP(site_map)
        expr_b = other.to_sSOP(site_map)

        expr = expr_a * expr_b

        return lSOP(expr, idx_to_label)

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
        self.expr = sop
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
        return self.expr.nmodes()

    def nterms(self) -> int:
        """
        Return the number of terms.

        :return: Number of terms
        :rtype: int
        """
        return self.expr.nterms()

    @property
    def dtype(self) -> np.dtype:
        return self.expr.dtype

    def complex_dtype(self) -> bool:
        return self.expr.complex_dtype()

    def backend(self) -> str:
        return self.expr.backend()


    @property
    def operator_dictionary(self):
        return self.expr.get_operator_dictionary()

    def set_operator_dictionary(self, opdict):
        self.expr.set_operator_dictionary(opdict)

 
    def jordan_wigner(self, ordering: list[str], sysinfo: SystemInfo, tol: float = 1e-15) -> "lCSOP":
        """Perform a Jordan-Wigner transform using a user-specified label ordering.

        This method applies the Jordan-Wigner transformation according to a given
        ordering of site labels, then converts the result back into a labelled
        operator representation.

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
        self.expr.prune_zeros(tol)

    def expand(self) -> "lSOP":
        """
        Expand into symbolic representation.

        :return: labelled symbolic operator
        :rtype: lSOP
        """
        ssop = self.expr.expand()
        return lSOP(ssop, self.index_to_label.copy())


    def to_SOP(self, site_map: Dict[str, int], nmodes: int = None) -> SOP:
        if nmodes is None:
            nmodes = len(site_map)
        new_sop = SOP(nmodes)

        opdict = self.expr.get_operator_dictionary()
        print(opdict, nmodes)
        for term, coeff in self.expr:
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

                print(op.fermionic)
                new_op = sOP(op.op, new_idx, op.fermionic)
                new_nbo.insert_back(new_op)
            print(new_nbo)

            new_sop += new_nbo

        return new_sop

    def compile(self, site_map: Dict[str, int], nmodes: Optional[int] = None) -> Union[SOP]:
        """Compile the labelled operator into a concrete backend representation.

        This method converts the symbolic operator expression stored in the lCSOP
        object into a concrete operator with physical indices according to the
        provided site mapping. T

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

    def expr(self) -> SOP:
        """
        Return the underlying SOP.

        :return: SOP object
        :rtype: SOP
        """
        return self.expr

    def __repr__(self):
        return f"LabelledCSOP(nmodes={self.nmodes()}, nterms={self.nterms()}, sites={self.sites()})"

    def __str__(self):
        return str(self.expr)

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
        self.expr = tmp.expr
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
    
    def __mul__(self, other : "lCSOP") -> "lCSOP":
        """
        Multiply the operator by a scalar.

        :param other: Scalar multiplier
        :type other: int, float, complex, or numpy scalar
        :return: Scaled operator
        :rtype: lCSOP
        """
        if np.isscalar(other):
            sop = copy.deepcopy(self.expr)
            sop *= other
            return lCSOP(sop,self.index_to_label.copy())
        
        return NotImplemented
    
    def __rmul__(self, other : "lCSOP") -> "lCSOP":
        """
        Multiply the operator by a scalar from the left

        :param other: Scalar multiplier
        :type other: int, float, complex, or numpy scalar
        :return: Scaled operator
        :rtype: lSOP
        """
        return self * other

    def __truediv__(self, other : "lCSOP") -> "lCSOP":
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

# Global builder context
_current_builder: Optional[OperatorBuilder] = None

# Context manager
@contextmanager
def operator_context():
    """Create a new operator builder context.

    All calls to op(...) and fop(...) inside this context will use
    the same underlying OperatorBuilder.
    """
    global _current_builder
    old = _current_builder
    _current_builder = OperatorBuilder()
    try:
        yield _current_builder
    finally:
        _current_builder = old

# Operator construction functions
def op(op_label: str, site: str):
    """Construct a non-fermionic operator within the current context."""
    if _current_builder is None:
        raise RuntimeError("op() called outside of operator context. Use @operator decorator or operator_context().")
    return _current_builder.op(op_label, site)


def fop(op_label: str, site: str):
    """Construct a fermionic operator within the current context."""
    if _current_builder is None:
        raise RuntimeError("fop() called outside of operator context. Use @operator decorator or operator_context().")
    return _current_builder.fop(op_label, site)


# Explicit wrap (optional utility)
def wrap(expr):
    """Wrap an expression into an lSOP using the current context."""
    if _current_builder is None:
        raise RuntimeError("wrap() called outside of operator context. ""Use @operator decorator.")
    return _current_builder.wrap(expr)

# Decorator (main user-facing API)

def operator(func=None, *, N=None):
    """Decorator to build a labelled operator.

    If N is provided, returns an lCSOP (compiled).
    Otherwise returns an lSOP (symbolic).
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            global _current_builder

            with operator_context() as builder:
                expr = f(*args, **kwargs)
                return builder.wrap(expr, N=N)

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator

