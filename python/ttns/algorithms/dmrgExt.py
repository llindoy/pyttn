from pyttn._pyttn import one_site_dmrg_complex, ttn_complex, ttn_real, sop_operator_complex
from pyttn._pyttn import multiset_one_site_dmrg_complex, ms_ttn_complex, ms_ttn_real, multiset_sop_operator_complex
import numpy as np

def dmrg(A, h, krylov_dim = 16, numthreads=1, expansion='onesite'):
    if isinstance(A, ttn_complex) and isinstance(h, sop_operator_complex):
        if expansion == 'onesite':
            return one_site_dmrg_complex(A, h, krylov_dim, numthreads)
        elif expansion == 'subspace':
            raise ValueError("subspace expansion algorithm has not yet been implemented.")
    else:
        raise RuntimeError("Invalid input types for dmrg.")

def multiset_dmrg(A, h, krylov_dim = 16, nkrylov_step=1, numthreads=1, expansion='onesite'):
    if isinstance(A, ms_ttn_complex) and isinstance(h, multiset_sop_operator_complex):
        if expansion == 'onesite':
            return multiset_one_site_dmrg_complex(A, h, krylov_dim, nkrylov_step, numthreads)
        elif expansion == 'subspace':
            raise ValueError("subspace expansion algorithm has not yet been implemented.")
    else:
        raise RuntimeError("Invalid input types for dmrg.")

def dmrg(A, h, krylov_dim = 16, nkrylov_step=1, numthreads=1, expansion='onesite'):
    if isinstance(A, ttn_complex) and isinstance(h, sop_operator_complex):
        if expansion == 'onesite':
            return one_site_dmrg_complex(A, h, krylov_dim, nkrylov_step, numthreads)
        elif expansion == 'subspace':
            raise ValueError("subspace expansion algorithm has not yet been implemented.")
    elif isinstance(A, ms_ttn_complex) and isinstance(h, multiset_sop_operator_complex):
        if expansion == 'onesite':
            return multiset_one_site_dmrg_complex(A, h, krylov_dim, nkrylov_step, numthreads)
        elif expansion == 'subspace':
            raise ValueError("subspace expansion algorithm has not yet been implemented.")
    else:
        raise RuntimeError("Invalid input types for dmrg.")
