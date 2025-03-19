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

from pyttn.ttnpp import liouville_space
from .SOPExt import SOP, __is_SOP
from .opdictExt import __is_operator_dictionary


def liouville_space_superoperator(
    Op, sys, optype, opdict=None, Lopdict=None, coeff=1.0
):
    r"""A function for taking a Hilbert space operator and system information object and constructing a Liouville space operator
    object rdependent on the argument optype.  Here we support the automatic generation of 4 different types of Liouville space operator
    these are left acting operators, right acting operators, commutator operators and anticommutator operators.

    To Do: Add parameters
    """
    Lop = None
    if __is_SOP(Op):
        Lop = SOP(sys.nprimitive_modes() * 2)
    else:
        otype = type(Op)
        Lop = otype()

    if optype == "-":
        if __is_operator_dictionary(opdict):
            liouville_space.commutator_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == "+":
        if __is_operator_dictionary(opdict):
            liouville_space.anticommutator_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == "L" or optype == "l":
        if __is_operator_dictionary(opdict):
            liouville_space.left_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.left_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == "R" or optype == "r":
        if __is_operator_dictionary(opdict):
            liouville_space.right_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff
            )
        else:
            liouville_space.right_superoperator(Op, sys, Lop, coeff=coeff)
    else:
        raise RuntimeError("Invalid superoperator label")
    return Lop
