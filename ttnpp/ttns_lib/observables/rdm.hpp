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

#include <linalg/linalg.hpp>
#include "../ttn/ttn.hpp"

#include <common/tmp_funcs.hpp>
#include <common/exception_handling.hpp>

#include <vector>
#include <algorithm>

namespace ttns
{
    //compute the reduced density matrix element between two 
    template <typename T, typename backend = linalg::blas_backend>
    class rdm
    {
    protected:
        using matrix_type = linalg::matrix<T, backend>;
        using real_type = typename tmp::get_real_type<T>::type;
        using size_type = typename backend::size_type;
        using ancestor_index = typename ttn<T, backend>::ancestor_index;

    protected:
        linalg::tensor<T, 4, backend> m_temp1;
        linalg::tensor<T, 4, backend> m_temp2;
        linalg::tensor<T, 4, backend> m_temp3;

        linalg::matrix<T, backend> m_working;
        linalg::matrix<T, backend> m_working2;
        linalg::matrix<T, backend> m_working3;

    public:
        rdm() {}
        template <typename state_type>
        rdm(const state_type &A, bool compute_2_dm = false) { CALL_AND_HANDLE(resize(A, compute_2_dm), "Failed to construct rdm object.  Failed to allocate internal buffers."); }

        rdm(const rdm &o) = default;
        rdm(rdm &&o) = default;

        rdm &operator=(const rdm &o) = default;
        rdm &operator=(rdm &&o) = default;

        void clear()
        {
            try
            {
                m_temp1.clear();
                m_temp2.clear();

            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to clear rdm object.");
            }
        }

        template <typename state_type>
        void operator()(state_type &psi, size_type mode, matrix_type& ret) 
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
            size_type ortho_centre = psi.orthogonality_centre();

            //get the node index of the required leaf node
            size_type lid = psi.get_leaf_index(mode);

            //now shift the index to the current node as required
            CALL_AND_HANDLE(psi.set_orthogonality_centre(lid), "Failed to shift orthogonality centre to required position");

            //now we can set the value of ret.  This is very easy to do, it is just a simple matrix multiplication
            CALL_AND_HANDLE(ret = psi[lid]().as_matrix()*linalg::adjoint(psi[lid]().as_matrix()), "Failed to compute RDM contraction");

            //finally we return the orthogonality centre to where it started
            CALL_AND_HANDLE(psi.set_orthogonality_centre(ortho_centre), "Failed to shift orthogonality centre to required position");
        }


    protected:
        template <typename state_type>
        size_type _two_body_density_path(const state_type& psi, size_type lid, size_type root, linalg::tensor<T, 4, backend>& temp)
        {
            size_type cur_node = lid;
            size_type next_node = psi[lid].parent().id();
            if(next_node == root){return lid;}
            size_type t2 = 0;
            
            while(next_node != root)
            {   
                size_type cid = psi[cur_node].child_id();
                auto Pt = psi[next_node]().as_rank_4(cid);

                const auto& A = psi[cur_node]().as_matrix();

                //if we are at the leaf node then we contract the current node up into its parent and contract the required parent indices
                if(cur_node == lid)
                {
                    //contract the leaf tensor in psi into its parent
                    //        |        2
                    //       -P-       |
                    //       /    =  0-x-1 
                    //      A          |
                    //      |          3
                    //resize the working array so that it can fit the intermediate result and so that the last index contains the two non-contracted indices.  
                    CALL_AND_HANDLE(m_working.resize(Pt.shape(0)*Pt.shape(2), Pt.shape(3)*A.shape(0)), "Failed to resize working array");
                    CALL_AND_HANDLE(m_working2.resize(A.shape(0)*Pt.shape(3), A.shape(0)*Pt.shape(3)), "Failed to resize working array");
                    CALL_AND_HANDLE(temp.resize(Pt.shape(0), Pt.shape(0), Pt.shape(3), Pt.shape(3)), "Failed to resize result array");

                    //now get rank 4 tensor representations of each of the matrices
                    auto working_tensor = m_working.reinterpret_shape(Pt.shape(0), Pt.shape(2), Pt.shape(3), A.shape(0));
                    auto working_tensor_2 = m_working2.reinterpret_shape(Pt.shape(3), A.shape(0), Pt.shape(3), A.shape(0));

                    CALL_AND_HANDLE(working_tensor = linalg::tensordot(Pt,psi[cur_node]().as_matrix(), std::array<int, 1>{{1}}, std::array<int, 1>{{1}}), "Failed to compute working tensor.");

                    // and contract the resulting tensor with itself
                    CALL_AND_HANDLE(m_working2 = linalg::adjoint(m_working)*m_working, "Failed to compute temp array.");

                    //finally transpose so that it is in the format temp expects
                    CALL_AND_HANDLE(temp = linalg::transpose(working_tensor_2, {0, 2, 1, 3}), "Failed to transpose working tensor into the temporary object.");

                    t2 = A.shape(0);
                }
                else
                {
                    //contract the next node with its conjugate 
                    CALL_AND_HANDLE(m_working.resize(Pt.shape(1)*Pt.shape(3), Pt.shape(1)*Pt.shape(3)), "Failed to resize working array");
                    CALL_AND_HANDLE(m_temp3 = temp, "Failed to copy temporary array into intermediate buffer.")
                    CALL_AND_HANDLE(temp.resize(Pt.shape(3), Pt.shape(3), t2, t2), "Failed to resize output array");

                    auto working_tensor = m_working.reinterpret_shape(Pt.shape(1), Pt.shape(3), Pt.shape(1), Pt.shape(3));
                    CALL_AND_HANDLE(working_tensor = linalg::tensordot(linalg::conj(Pt), Pt, std::array<int, 2>{{0, 2}}, std::array<int, 2>{{0, 2}}), "Failed to compute working tensor.");

                    //and contract the current nodes temp into the result
                    temp = linalg::tensordot(working_tensor, m_temp3, std::array<int, 2>{{0, 2}}, std::array<int, 2>{{0, 1}});
                }

                cur_node = next_node;
                next_node = psi[cur_node].parent().id();
            }
            return cur_node;
        }

        template <typename state_type>
        void _two_body_density_root(const state_type& psi, size_type mode1, size_type node1, size_type mode2, size_type node2, size_type root, linalg::tensor<T, 4, backend>& temp1, linalg::tensor<T, 4, backend>& temp2, matrix_type& ret)
        {
            //get the node index of the required leaf node
            size_type lid1 = psi.get_leaf_index(mode1);
            size_type lid2 = psi.get_leaf_index(mode2);

            //First we contract the root node into itself 
            size_type cid1 = psi[node1].child_id();
            size_type cid2 = psi[node2].child_id();

            size_type dim1 = psi.dim(mode1);
            size_type dim2 = psi.dim(mode2);
            
            //resize the output array to the correct size
            CALL_AND_HANDLE(ret.resize(dim1*dim2, dim1*dim2), "Failed to resize output matrix.");

            auto Pt = psi[root]().as_rank_5(cid1, cid2);

            CALL_AND_HANDLE(m_working.resize(Pt.shape(1)*Pt.shape(3), Pt.shape(1)*Pt.shape(3)), "Failed to resize working array");
            auto working_tensor = m_working.reinterpret_shape(Pt.shape(1), Pt.shape(3), Pt.shape(1), Pt.shape(3));

            CALL_AND_HANDLE(working_tensor = linalg::tensordot(linalg::conj(Pt), Pt, std::array<int, 3>{{0,2,4}}, std::array<int, 3>{{0,2,4}}), "Failed to contract root node");

            CALL_AND_HANDLE(m_working2.resize(Pt.shape(3)*Pt.shape(3), dim1*dim1), "Failed to resize working array");
            auto working_tensor_2 = m_working2.reinterpret_shape(Pt.shape(3), Pt.shape(3), dim1, dim1);
            //now contract the child tensors with dangling bonds into this object.
            //if the leaf index is the same as the node index.  Then we need to contract the leaf nodes into this node
            if(lid1 == node1)
            {
                m_working3.resize(Pt.shape(3)*Pt.shape(1), Pt.shape(3)*dim1);
                auto wt3 = m_working3.reinterpret_shape(Pt.shape(3), Pt.shape(1), Pt.shape(3), dim1);
                CALL_AND_HANDLE(wt3 = linalg::tensordot(working_tensor, linalg::conj(psi[lid1]().as_matrix()), std::array<int, 1>{{0}}, std::array<int, 1>{{1}}), "Failed to contract leaf into root.");
                CALL_AND_HANDLE(working_tensor_2 = linalg::tensordot(wt3, psi[lid1]().as_matrix(), std::array<int, 1>{{1}}, std::array<int, 1>{{1}}), "Failed to contract leaf into root.");
            }
            //otherwise we use the child m_temp object
            else
            {
                CALL_AND_HANDLE(working_tensor_2 = linalg::tensordot(working_tensor, temp1, std::array<int, 2>{{0,2}}, std::array<int, 2>{{0,1}}), "Failed to contract root node");
            }

            CALL_AND_HANDLE(m_working.resize(dim1*dim1, dim2*dim2), "Failed to resize working array");
            auto wt = m_working.reinterpret_shape(dim1, dim1, dim2, dim2);


            //if the leaf index is the same as the node index.  Then we need to contract the leaf nodes into this node
            if(lid2 == node2)
            {
                CALL_AND_HANDLE(m_working3.resize(Pt.shape(3)*dim1*dim1, dim2), "Failed to resize working array");
                auto working_tensor_3 = m_working3.reinterpret_shape(Pt.shape(3), dim1, dim1, dim2);
                CALL_AND_HANDLE(working_tensor_3 = linalg::tensordot(working_tensor_2, linalg::conj(psi[lid2]().as_matrix()), std::array<int, 1>{{0}}, std::array<int, 1>{{1}}), "Failed to contract root node");
                CALL_AND_HANDLE(wt = linalg::tensordot(working_tensor_3, psi[lid2]().as_matrix(), std::array<int, 1>{{0}}, std::array<int, 1>{{1}}), "Failed to contract root node");
            }
            //otherwise we use the child m_temp object
            else
            {
                CALL_AND_HANDLE(wt = linalg::tensordot(working_tensor_2,  temp2, std::array<int, 2>{{0, 1}}, std::array<int, 2>{{0, 1}}), "Failed to compute final contraction");
            }
            auto rt = ret.reinterpret_shape(dim1, dim2, dim1, dim2);
            CALL_AND_HANDLE(rt = linalg::transpose(wt, {0, 2, 1, 3}), "Failed to transpose result into ret");
        }

    public:
        template <typename state_type>
        void operator()(state_type &psi, size_type mode1, size_type mode2, matrix_type& ret) 
        { 
            if(mode2 == mode1){this->operator()(psi, mode1, ret);}
            else
            {
                if(mode1 >= psi.nmodes() or mode2 >= psi.nmodes())
                {
                    RAISE_EXCEPTION("Failed to compute rdm of single mode.  Mode index out of bounds.")
                }
                CALL_AND_HANDLE(this->resize(psi, true), "Failed to resize internal buffer for rdm evaluation.");

                if(!psi.has_orthogonality_centre())
                {
                    CALL_AND_HANDLE(psi.orthogonalise(), "Failed to orthogonalise input psi tensor.");
                }

                if(mode1 > mode2)
                {
                    size_t temp = mode1;
                    mode1 = mode2;
                    mode2 = temp;
                }

                //get the current orthogonality centre index
                size_type ortho_centre = psi.orthogonality_centre();

                //get the node index of the required leaf node
                size_type lid1 = psi.get_leaf_index(mode1);
                size_type lid2 = psi.get_leaf_index(mode2);

                //get the path of nodes spanning the two trees to the root
                ancestor_index inds1, inds2;
                psi.ancestor_indexing(lid1, inds1);
                psi.ancestor_indexing(lid2, inds2);


                std::vector<std::pair<size_t, size_t>> intersect;
                for(auto z : common::rzip(inds1, inds2))
                {
                    auto & z1 = std::get<0>(z);
                    auto & z2 = std::get<1>(z);
                    if(z1 == z2)
                    {
                        intersect.push_back(z1);
                    }
                }

                ASSERT(intersect.size() != 0, "Failed to compute RDM, the two nodes have no common ancestor.");
                size_type common_root = std::get<0>(intersect[intersect.size()-1]);
                size_type active_node_1 = 0;
                size_type active_node_2 = 0;

                //now shift the index to one of the requested leaf nodes
                CALL_AND_HANDLE(psi.set_orthogonality_centre(common_root), "Failed to shift orthogonality centre to required position");

                //now iterate from lid1 to common_root and evaluate the matrices
                CALL_AND_HANDLE(active_node_1 = _two_body_density_path(psi, lid1, common_root, m_temp1), "Failed to update path from leaf 1 to the common root.");

                //now iterate from lid2 to common_root and evaluate the matrices
                CALL_AND_HANDLE(active_node_2 = _two_body_density_path(psi, lid2, common_root, m_temp2), "Failed to update path from leaf 2 to the common root.");

                //now we want to evaluate the contraction on the root node
                CALL_AND_HANDLE(_two_body_density_root(psi, mode1, active_node_1, mode2, active_node_2, common_root, m_temp1, m_temp2, ret), "Failed to compute contraction up to root node.");

                //finally we return the orthogonality centre to where it started
                CALL_AND_HANDLE(psi.set_orthogonality_centre(ortho_centre), "Failed to shift orthogonality centre to required position");
            }
        }

        template <typename state_type>
        void resize(const state_type& A, bool compute_2_dm = false)
        {
            if(compute_2_dm)
            {
                m_temp1.clear();
                m_temp2.clear();

                //allocate the maximum sized buffers at each node of the tree
                tree<size_type> max_dims;
                max_dims.construct_topology(A); 

                for(size_type ind = 0; ind < max_dims.size(); ++ind)
                {
                    size_type rind = max_dims.size() - (ind+1);
                    if(max_dims[rind].is_leaf())
                    {
                        max_dims[rind]() = A[rind]().max_dimen();
                    }
                    else
                    {
                        size_type lmdim = 0;
                        for(size_type j = 0; j < max_dims[rind].size(); ++j)
                        {
                            if(lmdim < max_dims[rind][j]())
                            {
                                lmdim = max_dims[rind][j]();
                            }
                        }
                        max_dims[rind]() = lmdim;
                    }
                }

                size_t mdim = 0;
                //now we resize the A array so that it is large enough to fit the maximum size
                for (size_type ind = 0; ind < A.size(); ++ind)
                {
                    if(!A[ind].is_leaf())
                    {
                        max_dims[ind]() *= A[ind]().max_hrank();
                        if(max_dims[ind]() > mdim)
                        {
                            mdim = max_dims[ind]();
                        }
                    }
                }
                CALL_AND_HANDLE(m_temp1.reallocate(mdim*mdim), "Failed to allocate internal buffers for two body density matrix evaluation.");
                CALL_AND_HANDLE(m_temp2.reallocate(mdim*mdim), "Failed to allocate internal buffers for two body density matrix evaluation.");
            }
        }

    };

} // namespace ttns

#endif // PYTTN_TTNS_LIB_OBSERVABLES_MATRIX_ELEMENT_HPP_//
