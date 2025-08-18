/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PYTTN_TTNS_LIB_OBSERVABLES_RDM_HPP_
#define PYTTN_TTNS_LIB_OBSERVABLES_RDM_HPP_

#include "../ttn/ttn_nodes/node_traits/bool_node_traits.hpp"

#include <linalg/linalg.hpp>

#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>


namespace ttns
{
    //compute the reduced density matrix element between two 
    template <typename T, typename backend = linalg::blas_backend>
    class rdm
    {
    protected:
        using matrix_type = linalg::matrix<T, backend>;
        using observable_node = typename tree<observable_node_data<T, backend>>::node_type;
        using boolnode = typename tree<bool>::node_type;

        using real_type = typename tmp::get_real_type<T>::type;
        using size_type = typename backend::size_type;

        using me_core = rdm_engine<T, backend>;
        using op_base = typename me_core::op_base;

        using ancestor_index = typename ttn<T, backend>::ancestor_index;

    protected:
        tree<linalg::matrix<T, backend>> m_temp; 
    public:
        rdm() {}
        template <typename state_type>
        rdm(const state_type &A, bool compute_2_rdm = false) { CALL_AND_HANDLE(resize(A, compute_2_rdm), "Failed to construct rdm object.  Failed to allocate internal buffers."); }

        rdm(const rdm &o) = default;
        rdm(rdm &&o) = default;

        rdm &operator=(const rdm &o) = default;
        rdm &operator=(rdm &&o) = default;

        template <typename state_type, typename mat_type>
        void operator()(state_type &psi, size_t mode, mat_type& ret) 
        { 
            if(mode >= psi.nmodes())
            {
                RAISE_EXCEPTION("Failed to compute rdm of single mode.  Mode index out of bounds.")
            }
            if(!psi.has_orthogonality_centre())
            {
                CALL_AND_HANDLE(psi.orthogonalise(), "Failed to orthogonalise input psi tensor.");
            }

            //resize the output array to the correct size
            CALL_AND_HANDLE(ret.resize(psi.dim(mode), psi.dim(mode)), "Failed to resize output matrix.");

            //get the current orthogonality centre index
            size_t ortho_centre = psi.orthogonality_centre();

            //get the node index of the required leaf node
            size_t lid = psi.get_leaf_index(mode);

            //now shift the index to the current node as required
            CALL_AND_HANDLE(psi.set_orthogonality_centre(lid), "Failed to shift orthogonality centre to required position");

            //now we can set the value of ret.  This is very easy to do, it is just a simple matrix multiplication
            CALL_AND_HANDLE(ret = psi[lid]().as_matrix()*linalg::adjoint(psi[lid]().as_matrix()), "Failed to compute RDM contraction");

            //finally we return the orthogonality centre to where it started
            CALL_AND_HANDLE(psi.set_orthogonality_centre(ortho_centre), "Failed to shift orthogonality centre to required position");
        }

        /*
        template <typename state_type, typename mat_type>
        void operator()(state_type &psi, size_t mode1, size_t mode2, mat_type& ret) 
        { 
            if(mode2 == mode1){this->operator()(psi, mode1, ret);}
            else
            {
                if(mode2 > mode1)
                {
                    size_t temp = mode2;
                    mode2 = mode1;
                    mode1 = temp;
                }
                if(mode1 >= psi.nmodes() or mode2 >= psi.nmodes())
                {
                    RAISE_EXCEPTION("Failed to compute rdm of single mode.  Mode index out of bounds.")
                }
                CALL_AND_HANDLE(this->resize(psi, true), "Failed to resize internal buffer for rdm evaluation.");

                if(!psi.has_orthogonality_centre())
                {
                    CALL_AND_HANDLE(psi.orthogonalise(), "Failed to orthogonalise input psi tensor.");
                }

                //resize the output array to the correct size
                CALL_AND_HANDLE(ret.resize(psi.dim(mode1)*psi.dim(mode2), psi.dim(mode1)*psi.dim(mode2)), "Failed to resize output matrix.");

                //get the current orthogonality centre index
                size_t ortho_centre = psi.orthogonality_centre();

                //get the node index of the required leaf node
                size_t lid1 = psi.get_leaf_index(mode1);
                size_t lid2 = psi.get_leaf_index(mode2);

                //now shift the index to one of the requested leaf nodes
                CALL_AND_HANDLE(psi.set_orthogonality_centre(lid1), "Failed to shift orthogonality centre to required position");

                //get the path of nodes spanning the two trees to the root
                ancestor_index inds;
                psi.ancestor_indexing_leaf(lid1, inds);
                psi.ancestor_indexing(lid2, inds);

                //iterate over each node and evaluate the correct contraction
                for (const auto &pair : inds)
                {

                }

                //finally we return the orthogonality centre to where it started
                CALL_AND_HANDLE(psi.set_orthogonality_centre(ortho_centre), "Failed to shift orthogonality centre to required position");
            }
        }*/




        template <typename state_type>
        void resize(const state_type &A, bool compute_2_rdm = false)
        {
            if(compute_2_rdm)
            {
                //construt the tree for storing the temporary array used for 
                CALL_AND_HANDLE(m_temp.construct_topology(A), "Failed to construct the topology.");

                for (size_t ind = 0; ind < A.size(); ++ind)
                {
                    size_t d1 = 1;
                    size_t d2 = 1;
                    for (size_t i = 0; i < a.nmodes(); ++i)
                    {
                        if(A[ind].max_dim(i) > d1)
                        {
                            d2 = d1;
                            d1 = A[ind].max_dim(i);
                        }
                        else if(A[ind].max_dim(i) > d2)
                        {
                            d2 = A[ind].max_dim(i);
                        }
                    }                
                    //allocate the temporary array required to store the two rdm
                    m_temp[ind]().reallocate(d1*d2, d1*d2);
                }
            }
        }

        void clear()
        {
            try
            {
                m_temp.clear();
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to clear rdm object.");
            }
        }

    };

} // namespace ttns

#endif // PYTTN_TTNS_LIB_OBSERVABLES_MATRIX_ELEMENT_HPP_//
