from pyttn.ttnpp import liouville_space
from .SOPExt import SOP
from .opdictExt import *


def liouville_space_superoperator(Op, sys, optype, opdict=None, Lopdict=None, coeff=1.0):
    """A function for taking a Hilbert space operator and system information object and constructing a Liouville space operator
    object rdependent on the argument optype.  Here we support the automatic generation of 4 different types of Liouville space operator
    these are left acting operators, right acting operators, commutator operators and anticommutator operators.

    To Do: Add parameters
    """
    Lop = None
    if __is_SOP(Op):
        Lop = SOP(sys.nprimitive_modes()*2)
    else:
        otype = type(Op)
        Lop = otype()

    if optype == '-':
        if is_operator_dictionary(opdict):
            liouville_space.commutator_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff)
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == '+':
        if is_operator_dictionary(opdict):
            liouville_space.anticommutator_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff)
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == 'L' or optype == 'l':
        if is_operator_dictionary(opdict):
            liouville_space.left_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff)
        else:
            liouville_space.left_superoperator(Op, sys, Lop, coeff=coeff)
    elif optype == 'R' or optype == 'r':
        if is_operator_dictionary(opdict):
            liouville_space.right_superoperator(
                Op, sys, opdict, Lop, Lopdict, coeff=coeff)
        else:
            liouville_space.right_superoperator(Op, sys, Lop, coeff=coeff)
    else:
        raise RuntimeError("Invalid superoperator label")
    return Lop
