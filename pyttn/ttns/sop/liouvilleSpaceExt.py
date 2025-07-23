# This files is part of the pyTTN package.
#(C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import Optional, Union

from pyttn.ttnpp import liouville_space, system_modes

from .opdictExt import operator_dictionary
from .SOPExt import SOP


def liouville_space_superoperator(
    Op : SOP, sys : system_modes, optype: str, opdict: Optional[operator_dictionary]=None, Lopdict: Optional[operator_dictionary]=None, coeff : Union[float, complex]=1.0
) -> SOP:
    """A function for taking a Hilbert space operator and system information object and constructing a Liouville space operator
    object rdependent on the argument optype.  Here we support the automatic generation of 4 different types of Liouville space operator
    these are left acting operators, right acting operators, commutator operators and anticommutator operators.

    :param Op: _description_
    :type Op: SOP
    :param sys: _description_
    :type sys: system_modes
    :param optype: _description_
    :type optype: str
    :param opdict: _description_, defaults to None
    :type opdict: Optional[operator_dictionary], optional
    :param Lopdict: _description_, defaults to None
    :type Lopdict: Optional[operator_dictionary], optional
    :param coeff: _description_, defaults to 1.0
    :type coeff: Union[float, complex], optional
    :return: _description_
    :rtype: SOP
    """
    Lop = None
    if isinstance(Op, SOP):
        Lop = SOP(sys.nprimitive_modes() * 2)
    else:
        otype = type(Op)
        Lop = otype()

    if optype == "-":
        if isinstance(opdict, operator_dictionary):
            liouville_space.commutator_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == "+":
        if isinstance(opdict, operator_dictionary):
            liouville_space.anticommutator_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == "L" or optype == "l":
        if isinstance(opdict, operator_dictionary):
            liouville_space.left_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.left_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == "R" or optype == "r":
        if isinstance(opdict, operator_dictionary):
            liouville_space.right_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.right_superoperator(Op, sys, Lop, coeff=coeff)
    else:
        raise RuntimeError("Invalid superoperator label")
    return Lop
