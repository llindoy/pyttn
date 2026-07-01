# This files is part of the pyTTN package.
#(C) Copyright 2026 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Optional


from pyttn.ttnpp import symbolic_transpose as _symbolic_transpose_backend
from pyttn.ttnpp import system_modes

from .opdictExt import OperatorDictionary
from .sSOPExt import OPBase, sOP, sPOP, sSOP, sNBO

import numpy as np

def _infer_dtype_from_op(Op: OPBase):
    """
    Infer dtype (float or complex) from an operator.
    """

    # sNBO / sSOP: use coefficient
    if hasattr(Op, "coeff"):
        try:
            val = Op.coeff(0.0)  # evaluate at t=0 if time-dependent
        except Exception:
            val = Op.coeff()

        if isinstance(val, complex):
            return np.complex128
        else:
            return np.float64

    # fallback
    return np.complex128

def _infer_dtype_from_dict(opdict):
    return np.complex128 if opdict.complex_dtype() else np.float64

def _promote_dtype(op_dtype, dict_dtype=None):
    """Promote dtype: complex dominates real."""
    if dict_dtype is None:
        return op_dtype

    if dict_dtype is complex or dict_dtype == np.complex128:
        return np.complex128

    if op_dtype is complex or op_dtype == np.complex128:
        return np.complex128

    return np.float64


def symbolic_transpose(
    Op : OPBase, sys : system_modes, opdict: Optional[OperatorDictionary]=None, Lopdict: Optional[OperatorDictionary]=None
) -> OPBase:
    """
    Apply a symbolic transpose to an operator.

    This constructs the transpose of an operator using the operator
    dictionary associated with the system modes. For composite operators,
    the transpose is applied term-wise, including any required phase
    factors.

    :param Op: Input operator (sOP, sPOP, sNBO, or sSOP)
    :type Op: OPBase
    :param sys: System mode information describing the Hilbert space
    :type sys: system_modes
    :param opdict: Operator dictionary for input transformations, defaults to None
    :type opdict: Optional[OperatorDictionary], optional
    :param Lopdict: Operator dictionary for output operators, defaults to None
    :type Lopdict: Optional[OperatorDictionary], optional
    :return: Transposed operator of the same type as the input
    :rtype: tuple[OPBase, OperatorDictionary]
    """
    op_dtype = _infer_dtype_from_op(Op)

    dict_dtype = None
    if isinstance(opdict, OperatorDictionary):
        dict_dtype = _infer_dtype_from_dict(opdict)

    dtype = _promote_dtype(op_dtype, dict_dtype)

    #allocate output type.
    if isinstance(Op, sSOP):
        res = sSOP(dtype=dtype)
    elif isinstance(Op, sNBO):
        res = sNBO(dtype=dtype)
    elif isinstance(Op, (sOP, sPOP)):
        res = sNBO(dtype=dtype)
    else:
        raise TypeError(f"Unsupported Operator type: {type(Op)}")

    if isinstance(opdict, OperatorDictionary):
        if not isinstance(Lopdict, OperatorDictionary):
            Lopdict = type(opdict)(opdict.nmodes())
        _symbolic_transpose_backend.apply(Op, opdict, sys, res, Lopdict)

    else:
        _symbolic_transpose_backend.apply(Op, sys, res)

    return res, Lopdict
