from pyttn._pyttn import one_site_tdvp_complex, ttn_complex, sop_operator_complex, adaptive_one_site_tdvp_complex
from pyttn._pyttn import multiset_one_site_tdvp_complex, ms_ttn_complex, multiset_sop_operator_complex
import numpy as np

def single_set_tdvp(A, h, krylov_dim = 16, nkrylov_step=1, subspace_krylov_dim = 6, subspace_neigs = 2, numthreads=1, expansion='onesite'):
    
    if isinstance(A, ttn_complex) and isinstance(h, sop_operator_complex):
        if expansion == 'onesite':
            return one_site_tdvp_complex(A, h, krylov_dim, nkrylov_step, numthreads)
        elif expansion == 'subspace':
            return adaptive_one_site_tdvp_complex(A, h, krylov_dim, nkrylov_step, subspace_krylov_dim, subspace_neigs, numthreads)
    else:
        raise RuntimeError("Invalid input types for single set tdvp.")
        

def multiset_tdvp(A, h, krylov_dim = 16, nkrylov_step=1, subspace_krylov_dim = 6, subspace_neigs = 2, numthreads=1, expansion='onesite'):
    if isinstance(A, ms_ttn_complex) and isinstance(h, multiset_sop_operator_complex):
        if expansion == 'onesite':
            return multiset_one_site_tdvp_complex(A, h, krylov_dim, nkrylov_step, numthreads)
        elif expansion == 'subspace':
            raise ValueError("subspace expansion algorithm has not yet been implemented for multiset TTNs.")
    else:
        raise RuntimeError("Invalid input types for multiset tdvp.")

def tdvp(A, h, krylov_dim = 16, nkrylov_step=1, subspace_krylov_dim = 6, subspace_neigs = 2, numthreads=1, expansion='onesite'):
    if isinstance(A, ttn_complex) and isinstance(h, sop_operator_complex):
        return single_set_tdvp(A, h, krylov_dim = krylov_dim, nkrylov_step = nkrylov_step, subspace_krylov_dim=subspace_krylov_dim, subspace_neigs=subspace_neigs,numthreads = numthreads, expansion=expansion)
    elif isinstance(A, ms_ttn_complex) and isinstance(h, multiset_sop_operator_complex):
        return multiset_tdvp(A, h, krylov_dim = krylov_dim, nkrylov_step = nkrylov_step, subspace_krylov_dim=subspace_krylov_dim, subspace_neigs=subspace_neigs,numthreads = numthreads, expansion=expansion)
    else:
        raise RuntimeError("Invalid input types for tdvp.")
