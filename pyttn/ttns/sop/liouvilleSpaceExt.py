from pyttn._pyttn import liouville_space
from .SOPExt import SOP
from .opdictExt import *

def liouville_space_superoperator(Op, sys, opdict, Lopdict, optype):
    Lop = SOP(sys.nprimitive_modes()*2)
    if optype == '-':
        if is_operator_dictionary(opdict):
            liouville_space.commutator_superoperator(Op, sys, opdict, Lop, Lopdict)
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop)
    elif optype == '+':
        if is_operator_dictionary(opdict):
            liouville_space.anticommutator_superoperator(Op, sys, opdict, Lop, Lopdict)
        else:
            liouville_space.commutator_superoperator(Op, sys, Lop)
    elif optype == 'L' or optype == 'l':
        if is_operator_dictionary(opdict):
            liouville_space.left_superoperator(Op, sys, opdict, Lop, Lopdict)
        else:
            liouville_space.left_superoperator(Op, sys, Lop)
    elif optype == 'R' or optype == 'r':
        if is_operator_dictionary(opdict):
            liouville_space.right_superoperator(Op, sys, opdict, Lop, Lopdict)
        else:
            liouville_space.right_superoperator(Op, sys, Lop)
    else:
        raise RuntimeError("Invalid superoperator label")
    return Lop
