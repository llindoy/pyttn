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

from .sSOPExt import sOP
from .SOPExt import SOP
from .labelled_sSOP import lSOP
from .labelled_SOP import lCSOP

from typing import Optional, Union
from contextlib import contextmanager

from functools import wraps

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
        if isinstance(expr, SOP):
            #if the number of modes in expr is not equal to N we need to create a new SOP
            if expr.nmodes() == N:
                sop = expr
            else:
                sop = SOP(N)
                sop += expr
        else:
            sop = SOP(N)
            sop += expr
        return lCSOP(sop, dict(self.index_to_label))

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
    if _current_builder is not None:
        return _current_builder.op(op_label, site)
    b = OperatorBuilder()
    return b.wrap(b.op(op_label, site))

def fop(op_label: str, site: str):
    """Construct a fermionic operator within the current context."""
    if _current_builder is not None:
        return _current_builder.fop(op_label, site)
    b = OperatorBuilder()
    return b.wrap(b.fop(op_label, site))

# Explicit wrap (optional utility)
def wrap(expr):
    """Wrap an expression into an lSOP using the current context."""
    if _current_builder is None:
        raise RuntimeError("wrap() called outside of operator context. ""Use @operator decorator.")
    return _current_builder.wrap(expr)


# Decorator (main user-facing API)
def operator(func=None, *, N=None):
    """Decorator to build a labelled operator.

    If N is provided, returns an lCSOP (compiled). Otherwise returns an lSOP (symbolic).
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

#def sum(ops):
#    return reduce(lambda a,b: a+b, ops)

#def prod(ops):
#    return reduce(lambda a,b: a@b, ops)


def sum(ops):
    ops = iter(ops)

    try:
        result = next(ops)
    except StopIteration:
        raise ValueError("empty operator sum") from None

    for op in ops:
        result = result + op

    return result


def prod(ops):
    ops = iter(ops)

    try:
        result = next(ops)
    except StopIteration:
        raise ValueError("empty operator product") from None

    for op in ops:
        result = result * op

    return result
