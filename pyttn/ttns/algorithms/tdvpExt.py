# This files is part of the pyTTN package.
# (C) Copyright 2025 NPL Management Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import TypeAlias

from pyttn.ttnpp import (
    one_site_tdvp_complex,
    ttn_complex,
    sop_operator_complex,
    adaptive_one_site_tdvp_complex,
)
from pyttn.ttnpp import (
    multiset_one_site_tdvp_complex,
    ms_ttn_complex,
    multiset_sop_operator_complex,
)

from pyttn.ttns.ttn.ttnExt import ttn_type, ms_ttn_type
from pyttn.ttns.operators.sopOperatorExt import sop_operator_type
from pyttn.ttns.operators.mssopOperatorExt import ms_sop_operator_type


tdvp_type: TypeAlias = one_site_tdvp_complex | adaptive_one_site_tdvp_complex
ms_tdvp_type: TypeAlias = multiset_one_site_tdvp_complex

# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import one_site_tdvp_complex as one_site_tdvp_complex_cuda
    from pyttn.ttnpp.cuda import ttn_complex as ttn_complex_cuda
    from pyttn.ttnpp.cuda import sop_operator_complex as sop_operator_cuda

    from pyttn.ttnpp.cuda import (
        mulitset_one_site_tdvp_complex as multiset_one_site_tdvp_complex_cuda,
    )
    from pyttn.ttnpp.cuda import ms_ttn_complex as ms_ttn_complex_cuda
    from pyttn.ttnpp.cuda import (
        multiset_sop_operator_complex as multiset_sop_operator_cuda,
    )

    __cuda_import = True
    tdvp_type: TypeAlias = tdvp_type | one_site_tdvp_complex_cuda
    ms_tdvp_type: TypeAlias = ms_tdvp_type | multiset_one_site_tdvp_complex_cuda

except ImportError:
    __cuda_import = False


def __single_set_tdvp_blas(A, H, expansion="onesite", **kwargs):
    if isinstance(A, ttn_complex) and isinstance(H, sop_operator_complex):
        if expansion == "onesite":
            return one_site_tdvp_complex(A, H, **kwargs)
        elif expansion == "subspace":
            return adaptive_one_site_tdvp_complex(A, H, **kwargs)
    else:
        raise RuntimeError("Invalid input types for single set tdvp.")


def __single_set_tdvp_cuda(A, H, expansion="onesite", **kwargs):
    if isinstance(A, ttn_complex_cuda) and isinstance(H, sop_operator_cuda):
        if expansion == "onesite":
            return one_site_tdvp_complex_cuda(A, H, **kwargs)
    else:
        raise RuntimeError("Invalid input types for single set tdvp.")


def single_set_tdvp(
    A: ttn_type, H: sop_operator_type, expansion: str = "onesite", **kwargs
) -> tdvp_type:
    """A factory method for constructing an object used for performing single set TDVP calculations

    :param A: Tree Tensor Network that the DMRG algorithm will act on
    :type A: ttn_type
    :param H: The Hamiltonian sop operator object
    :type H: sop_operator_type
    :param expansion: A string determining the type of bond dimension expansion to be used.  Either no subspace expansion ('onesite') or energy variance based ('subspace').  (Default: 'onesite')
    :type expansion: {'onesite', 'subspace'}, optional
    :param **kwargs: Keyword arguments to pass to the DMRG engine constructor.  For details see one_site_dmrg_complex or adaptive_one_site_tdvp_complex

    :returns: The TDVP evaluation object
    :rtype: tdvp_type
    """
    if A.backend() == H.backend():
        if A.backend() == "blas":
            return __single_set_tdvp_blas(A, H, expansion=expansion, **kwargs)
        elif A.backend() == "cuda":
            return __single_set_tdvp_blas(A, H, expansion=expansion, **kwargs)
        else:
            raise RuntimeError("Invalid backend for single set tdvp")


def __multiset_tdvp_blas(A, H, expansion="onesite", **kwargs):
    if isinstance(A, ms_ttn_complex) and isinstance(H, multiset_sop_operator_complex):
        if expansion == "onesite":
            return multiset_one_site_tdvp_complex(A, H, **kwargs)
        elif expansion == "subspace":
            raise ValueError(
                "subspace expansion algorithm has not yet been implemented for multiset TTNs."
            )
    else:
        raise RuntimeError("Invalid input types for multiset tdvp.")


def __multiset_tdvp_cuda(A, H, expansion="onesite", **kwargs):
    if isinstance(A, ms_ttn_complex_cuda) and isinstance(H, multiset_sop_operator_cuda):
        if expansion == "onesite":
            return multiset_one_site_tdvp_complex_cuda(A, H, **kwargs)
    else:
        raise RuntimeError("Invalid input types for multiset tdvp.")


def multiset_tdvp(
    A: ms_ttn_type, H: ms_sop_operator_type, expansion: str = "onesite", **kwargs
) -> ms_tdvp_type:
    r"""A factory method for constructing an object used for performing multiset tdvp calculations

    :param A: Tree Tensor Network that the TDVP algorithm will act on
    :type A: ms_ttn_type
    :param H: The Hamiltonian sop operator object
    :type H: ms_sop_operator_type
    :param expansion: A string determining the type of bond dimension expansion to be used.  (Default: 'onesite')
    :type expansion: {'onesite'}, optional
    :param **kwargs: Keyword arguments to pass to the DMRG engine constructor.  For details see multiset_one_site_dmrg_complex

    :returns: The TDVP evaluation object
    :rtype: ms_tdvp_type
    """
    if A.backend() == H.backend():
        if A.backend() == "blas":
            return __multiset_tdvp_blas(A, H, expansion=expansion, **kwargs)
        elif A.backend() == "cuda":
            return __multiset_tdvp_blas(A, H, expansion=expansion, **kwargs)
        else:
            raise RuntimeError("Invalid backend for multiset set tdvp")


def tdvp(
    A: ttn_type | ms_ttn_type,
    H: sop_operator_type | ms_sop_operator_type,
    expansion: str = "onesite",
    **kwargs,
) -> tdvp_type | ms_tdvp_type:
    r"""A factory method for constructing an object used for performing either single or multi set tdvp calculations.
    Which type to construct is determined by the types of the input A and h matrices.

    :param A: Tree Tensor Network that the DMRG algorithm will act on
    :type A: ttn_type | ms_ttn_type
    :param H: The Hamiltonian sop operator object
    :type H: sop_operator_type | ms_sop_operator_type
    :param expansion: A string determining the type of bond dimension expansion to be used.  Either no subspace expansion ('onesite') or energy variance based ('subspace').  (Default: 'onesite')
    :type expansion: {'onesite', 'subspace'}, optional
    :param **kwargs: Keyword arguments to pass to the DMRG engine constructor.  For details see one_site_tdvp_complex or adaptive_one_site_tdvp_complex

    :returns: The TDVP evaluation object
    :rtype: tdvp_type | ms_tdvp_type
    """
    if isinstance(A, ttn_complex) and isinstance(H, sop_operator_complex):
        return single_set_tdvp(A, H, expansion=expansion, **kwargs)
    elif isinstance(A, ms_ttn_complex) and isinstance(H, multiset_sop_operator_complex):
        return multiset_tdvp(A, H, expansion=expansion, **kwargs)
    else:
        raise RuntimeError("Invalid input types for tdvp.")
