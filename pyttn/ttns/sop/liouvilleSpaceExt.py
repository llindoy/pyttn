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

from .opdictExt import OperatorDictionary
from .SOPExt import SOP


def liouville_space_superoperator(
    Op : SOP, sys : system_modes, optype: str, opdict: Optional[OperatorDictionary]=None, Lopdict: Optional[OperatorDictionary]=None, coeff : Union[float, complex]=1.0
) -> SOP:
    """
    Construct a Liouville-space superoperator from a Hilbert-space operator.

    The type of superoperator is determined by `optype`:
    "L"/"l" -> left action (O\\rho),
    "R"/"r" -> right action (\\rhoO, via transpose),
    "-" -> commutator ([O, \\rho]),
    "+" -> anticommutator ({O, \\rho}).

    :param Op: Hilbert-space operator to be mapped to Liouville space
    :type Op: SOP
    :param sys: System mode information describing the Hilbert space
    :type sys: system_modes
    :param optype: Superoperator type ("L", "R", "-", "+")
    :type optype: str
    :param opdict: Operator dictionary for input operator transformations, defaults to None
    :type opdict: Optional[OperatorDictionary], optional
    :param Lopdict: Operator dictionary for Liouville-space operators, defaults to None
    :type Lopdict: Optional[OperatorDictionary], optional
    :param coeff: Overall prefactor multiplying the superoperator, defaults to 1.0
    :type coeff: Union[float, complex], optional
    :return: Liouville-space superoperator acting on the doubled system
    :rtype: SOP
    """

    Lop = None
    if isinstance(Op, SOP):
        Lop = SOP(sys.nprimitive_modes() * 2)
    else:
        otype = type(Op)
        Lop = otype()

    if optype == "-":
        if isinstance(opdict, OperatorDictionary):
            liouville_space.commutator_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == "+":
        if isinstance(opdict, OperatorDictionary):
            liouville_space.anticommutator_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == "L" or optype == "l":
        if isinstance(opdict, OperatorDictionary):
            liouville_space.left_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.left_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == "R" or optype == "r":
        if isinstance(opdict, OperatorDictionary):
            liouville_space.right_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.right_superoperator(Op, sys, Lop, coeff=coeff)
    else:
        raise RuntimeError("Invalid superoperator label")
    return Lop
