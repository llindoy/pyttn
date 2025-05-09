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


from typing import TypeAlias

from pyttn.ttnpp import (
    one_site_dmrg_complex,
    adaptive_one_site_dmrg_complex,
    ttn_complex,
    sop_operator_complex,
)
from pyttn.ttnpp import (
    multiset_one_site_dmrg_complex,
    ms_ttn_complex,
    multiset_sop_operator_complex,
)

from pyttn.ttns.ttn.ttnExt import ttn_type, ms_ttn_type
from pyttn.ttns.operators.sopOperatorExt import sop_operator_type
from pyttn.ttns.operators.mssopOperatorExt import ms_sop_operator_type


dmrg_type: TypeAlias = one_site_dmrg_complex | adaptive_one_site_dmrg_complex
ms_dmrg_type: TypeAlias = multiset_one_site_dmrg_complex


#TODO: Need to add on cuda backend functionality here

def single_set_dmrg(A : ttn_type, H : sop_operator_type, expansion: str="onesite", **kwargs) -> dmrg_type:
    """A factory method for constructing an object used for performing single set dmrg calculations

    :param A: Tree Tensor Network that the DMRG algorithm will act on
    :type A: ttn_type
    :param H: The Hamiltonian sop operator object
    :type H: sop_operator_type
    :param expansion: A string determining the type of bond dimension expansion to be used.  Either no subspace expansion ('onesite') or energy variance based ('subspace').  (Default: 'onesite')
    :type expansion: {'onesite', 'subspace'}, optional
    :param **kwargs: Keyword arguments to pass to the DMRG engine constructor.  For details see one_site_dmrg_complex or adaptive_one_site_dmrg_complex

    :returns: The DMRG evaluation object
    :rtype: dmrg_type
    """

    if isinstance(A, ttn_complex) and isinstance(H, sop_operator_complex):
        if expansion == "onesite":
            return one_site_dmrg_complex(A, H, **kwargs)
        elif expansion == "subspace":
            return adaptive_one_site_dmrg_complex(A, H, **kwargs)
    else:
        raise RuntimeError("Invalid input types for dmrg.")


def multiset_dmrg(A : ms_ttn_type, H : ms_sop_operator_type, expansion:str="onesite", **kwargs) -> ms_dmrg_type:
    """A factory method for constructing an object used for performing multiset dmrg calculations

    :param A: Tree Tensor Network that the DMRG algorithm will act on
    :type A: ms_ttn_type
    :param H: The Hamiltonian sop operator object
    :type H: ms_sop_operator_type
    :param expansion: A string determining the type of bond dimension expansion to be used.  (Default: 'onesite')
    :type expansion: {'onesite'}, optional
    :param **kwargs: Keyword arguments to pass to the DMRG engine constructor.  For details see multiset_one_site_dmrg_complex

    :returns: The DMRG evaluation object
    :rtype: ms_dmrg_type
    """
    if isinstance(A, ms_ttn_complex) and isinstance(H, multiset_sop_operator_complex):
        if expansion == "onesite":
            return multiset_one_site_dmrg_complex(A, H, **kwargs)
        elif expansion == "subspace":
            raise ValueError(
                "subspace expansion algorithm has not yet been implemented."
            )
    else:
        raise RuntimeError("Invalid input types for dmrg.")


def dmrg(A : ttn_type | ms_ttn_type, H : sop_operator_type | ms_sop_operator_type, expansion:str="onesite", **kwargs) -> dmrg_type | ms_dmrg_type:
    """A factory method for constructing an object used for performing either single or multi set dmrg calculations.
    Which type to construct is determined by the types of the input A and h matrices.  For details on the use of these
    DMRG objects please see the documentation associated with the possible return types.

    :param A: Tree Tensor Network that the DMRG algorithm will act on
    :type A: ttn_type | ms_ttn_type
    :param H: The Hamiltonian sop operator object
    :type H: sop_operator_type | ms_sop_operator_type
    :param expansion: A string determining the type of bond dimension expansion to be used.  Either no subspace expansion ('onesite') or energy variance based ('subspace').  (Default: 'onesite')
    :type expansion: {'onesite', 'subspace'}, optional
    :param **kwargs: Keyword arguments to pass to the DMRG engine constructor.  For details see one_site_dmrg_complex or adaptive_one_site_dmrg_complex

    :returns: The DMRG evaluation object
    :rtype: dmrg_type | ms_dmrg_type
    """

    if isinstance(A, ttn_complex) and isinstance(H, sop_operator_complex):
        return single_set_dmrg(A, H, expansion=expansion, **kwargs)
    elif isinstance(A, ms_ttn_complex) and isinstance(H, multiset_sop_operator_complex):
        return multiset_dmrg(A, H, expansion=expansion, **kwargs)
    else:
        raise RuntimeError("Invalid input types for dmrg.")
