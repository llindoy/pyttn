from pyttn._pyttn import tdvp_complex, ttn_complex, ttn_real
import numpy as np

def tdvp(A, h, krylov_dim = 16, nkrylov_step=1, numthreads=1, expansion='onesite'):
    if isinstance(A, ttn_complex) and isinstance(h, sop_operator_complex):
        if expansion == 'onesite':
            return tdvp_complex(A, h, krylov_dim, nkrylov_step, numthreads)
        elif expansion == 'subspace':
            raise ValueError("subspace expansion algorithm has not yet been implemented.")
    else:
        raise RuntimeError("Invalid input types for tdvp.")
        
