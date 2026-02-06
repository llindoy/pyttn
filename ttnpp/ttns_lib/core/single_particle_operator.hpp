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

#ifndef PYTTN_TTNS_LIB_CORE_SINGLE_PARTICLE_OPERATOR_HPP_
#define PYTTN_TTNS_LIB_CORE_SINGLE_PARTICLE_OPERATOR_HPP_

#include <linalg/linalg.hpp>
#include "../ttn/ttn.hpp"
#include "../ttn/ms_ttn.hpp"
#include "../operators/sop_operator.hpp"
#include "../operators/multiset_sop_operator.hpp"
#include "matrix_element_buffer.hpp"
#include "observable_node.hpp"
#include "multiset_sop_env_node.hpp"
#include "kronecker_product_operator_helper.hpp"

namespace ttns
{
    template <typename T, typename backend>
    class single_particle_operator_engine
    {
        using kpo = kronecker_product_operator_mel<T, backend>;
        using hnode = ttn_node<T, backend>;
        using hdata = ttn_node_data<T, backend>;

        using ms_slice_node = multiset_ttn_node_slice<T, backend, true>;

        using soptype = sop_operator<T, backend>;
        using cinfnode = typename soptype::node_type;

        using ms_hnode = ms_ttn_node<T, backend>;
        using ms_hdata = multiset_node_data<T, backend>;

        using ms_soptype = multiset_sop_operator<T, backend>;
        using ms_cinfnode = typename ms_soptype::node_type;

        using op_container = typename soptype::container_type;

        using mat = linalg::matrix<T, backend>;
        using buffer_type = matrix_element_buffer<T, backend>;
        using cinftype = sttn_node_data<T>;

        using size_type = typename backend::size_type;

    public:
        template <typename spfnode, typename Atype>
        static inline void evaluate(const soptype &h, const cinfnode &cinf, Atype &&A, spfnode &hspf, buffer_type& buffer, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t /*set_var_nthreads*/ = 1)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a single particle operator given a site tensor and surrounding environment tensors.");
#endif            
            // resize the matrices in the event that the tensor objects have changed size
            if (A.is_leaf())
            {
                CALL_AND_RETHROW(evaluate_leaf_single_set(h, cinf, std::forward<Atype>(A), std::forward<Atype>(A), buffer, hspf, compute_identity, update_all, operator_sum_nthreads));
            }
            else
            {
                CALL_AND_RETHROW(evaluate_branch_single_set(cinf, std::forward<Atype>(A), std::forward<Atype>(A), buffer, hspf, compute_identity, update_all, operator_sum_nthreads));
            }
        }

        template <typename spfnode, typename Btype, typename Atype>
        static inline void evaluate(const soptype &h, const cinfnode &cinf, Btype &&B, Atype &&A, spfnode &hspf, buffer_type& buffer, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads=1, size_t /*set_var_nthreads*/=1)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a single particle operator given a site tensor and surrounding environment tensors.");
#endif        
            // resize the matrices in the event that the tensor objects have changed size
            if (A.is_leaf())
            {
                CALL_AND_RETHROW(evaluate_leaf_single_set(h, cinf, std::forward<Btype>(B), std::forward<Atype>(A), buffer, hspf, compute_identity, update_all, operator_sum_nthreads));
            }
            else
            {
                CALL_AND_RETHROW(evaluate_branch_single_set(cinf, std::forward<Btype>(B), std::forward<Atype>(A), buffer, hspf, compute_identity, update_all, operator_sum_nthreads));
            }
        }

        template <typename spfnode, typename Atype>
        static inline void evaluate(const soptype &h, const cinfnode &cinf, Atype &&A, size_t /* i */, size_t /* c */, spfnode &hspf, buffer_type& buffer, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads=1, size_t /*set_var_nthreads*/=1)
        {
            CALL_AND_RETHROW(evaluate(h, cinf, std::forward<Atype>(A), std::forward<Atype>(A), hspf, buffer, compute_identity, update_all, operator_sum_nthreads));
        }

        template <typename spfnode, typename Btype, typename Atype>
        static inline void evaluate(const soptype &h, const cinfnode &cinf, Btype &&B, Atype &&A, size_t /* i */, size_t /* c */, spfnode &hspf, buffer_type& buffer, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads=1, size_t /*set_var_nthreads*/=1)
        {
            CALL_AND_RETHROW(evaluate(h, cinf, std::forward<Btype>(B), std::forward<Atype>(A), hspf, buffer, compute_identity, update_all, operator_sum_nthreads));
        }

    public:
        template <typename spfnode>
        static inline void evaluate(const ms_soptype &h, const ms_cinfnode &cinf, const ms_hnode &A, spfnode &hspf, buffer_type &buffer, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t set_var_nthreads =1)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a single particle operator given a site tensor and surrounding environment tensors.");
#endif        
            // resize the matrices in the event that the tensor objects have changed size
            if (A.is_leaf())
            {
                CALL_AND_RETHROW(evaluate_leaf(h, cinf, A, A, buffer, hspf, compute_identity, update_all, operator_sum_nthreads, set_var_nthreads));
            }
            else
            {
                CALL_AND_RETHROW(evaluate_branch(cinf, A, A, buffer, hspf, compute_identity, update_all, operator_sum_nthreads, set_var_nthreads));
            }
        }

        template <typename spfnode>
        static inline void evaluate(const ms_soptype &h, const ms_cinfnode &cinf, const ms_hnode &B, const ms_hnode &A, spfnode &hspf, buffer_type &buffer, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t set_var_nthreads =1)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a single particle operator given a site tensor and surrounding environment tensors.");
#endif        
            // resize the matrices in the event that the tensor objects have changed size
            if (A.is_leaf())
            {
                CALL_AND_RETHROW(evaluate_leaf(h, cinf, B, A, buffer, hspf, compute_identity, update_all, operator_sum_nthreads, set_var_nthreads));
            }
            else
            {
                CALL_AND_RETHROW(evaluate_branch(cinf, B, A, buffer, hspf, compute_identity, update_all, operator_sum_nthreads, set_var_nthreads));
            }
        }

        template <typename spfnode>
        static inline void evaluate(const ms_soptype &h, const ms_cinfnode &cinf, const ms_hnode &A, size_t i, size_t c, spfnode &hspf, buffer_type &buffer, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a single particle operator given a site tensor and surrounding environment tensors.");
#endif        
            // resize the matrices in the event that the tensor objects have changed size
            if (A.is_leaf())
            {
                CALL_AND_RETHROW(evaluate_leaf(h, cinf, A, A, i, c, buffer, hspf, compute_identity, update_all, operator_sum_nthreads));
            }
            else
            {
                CALL_AND_RETHROW(evaluate_branch(cinf, A, A, i, c, buffer, hspf, compute_identity, update_all, operator_sum_nthreads));
            }
        }

        template <typename spfnode>
        static inline void evaluate(const ms_soptype &h, const ms_cinfnode &cinf, const ms_hnode &B, const ms_hnode &A, size_t i, size_t c, spfnode &hspf, buffer_type &buffer, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a single particle operator given a site tensor and surrounding environment tensors.");
#endif        
            // resize the matrices in the event that the tensor objects have changed size
            if (A.is_leaf())
            {
                CALL_AND_RETHROW(evaluate_leaf(h, cinf, B, A, i, c, buffer, hspf, compute_identity, update_all, operator_sum_nthreads));
            }
            else
            {
                CALL_AND_RETHROW(evaluate_branch(cinf, B, A, i, c, buffer, hspf, compute_identity, update_all, operator_sum_nthreads));
            }
        }

    protected:
        template <typename spftype>
        static inline void evaluate_leaf(const op_container &h, const cinftype &cinf, const hdata &B, const hdata &A, buffer_type &buffer, spftype &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a single particle operator given a leaf site tensor and surrounding environment tensors.");
#endif        
            try
            {
                CALL_AND_HANDLE(hspf.resize_matrices(B.size(1), A.size(1)), "Failed to resize the single-particle Hamiltonian operator matrices.");
                const auto &a = A.as_matrix();
                const auto &b = B.as_matrix();

                // if the A matrix and B matrix are not the same then we simply go through and compute the identity matrix
                if ((&A != &B || compute_identity) && update_all)
                {
                    CALL_AND_HANDLE(hspf.spf_id() = adjoint(b) * a, "Failed to compute the id matrix term.");
                }

#ifdef USE_OPENMP
                #pragma omp parallel for num_threads(operator_sum_nthreads) default(shared) schedule(static) 
#endif
                for (size_type ind = 0; ind < cinf.nterms(); ++ind)
                {
#ifdef USE_OPENMP
                    size_type ti = omp_get_thread_num() + tid*operator_sum_nthreads;
#else
                    size_type ti = tid*operator_sum_nthreads;
#endif                                     
                    CALL_AND_HANDLE(buffer.HA[ti].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                    // update all terms if we aren't worrying about time dependence otherwise only deal with time dependence
                    if (update_all || cinf[ind].is_time_dependent())
                    {
                        if (!cinf[ind].is_identity_spf())
                        {
                            hspf.spf(ind).fill_zeros();
                            for (size_type i = 0; i < cinf[ind].nspf_terms(); ++i)
                            {
                                auto &indices = cinf[ind].spf_indexing()[i][0];
                                CALL_AND_HANDLE(h[indices[0]][indices[1]].apply(a, buffer.HA[ti]), "Failed to apply leaf operator.");
                                CALL_AND_HANDLE(hspf.spf(ind) += cinf[ind].spf_coeff(i) * adjoint(b) * buffer.HA[ti], "Failed to apply matrix product to obtain result.");
                            }
                        }
                    }
                }
            }
            catch (const common::invalid_value &ex)
            {
                logging::error(ex.what());
                RAISE_NUMERIC("evaluating single particle operator at a leaf node.");
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate single particle operator at a leaf node.");
            }
        }

        template <typename spfnode>
        static inline void evaluate_leaf_single_set(const soptype &h, const cinfnode &cinf, const hnode &B, const hnode &A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            CALL_AND_RETHROW(evaluate_leaf(h.mode_operators(), cinf(), B(), A(), buffer, hspf(), compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
        static inline void evaluate_leaf_single_set(const soptype &h, const cinfnode &cinf, ms_slice_node &&B, ms_slice_node &&A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            CALL_AND_RETHROW(evaluate_leaf(h.mode_operators(), cinf(), B(), A(), buffer, hspf(), compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
        static inline void evaluate_leaf_single_set(const soptype &h, const cinfnode &cinf, ms_slice_node &&B, const hnode &A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            CALL_AND_RETHROW(evaluate_leaf(h.mode_operators(), cinf(), B(), A(), buffer, hspf(), compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
        static inline void evaluate_leaf_single_set(const soptype &h, const cinfnode &cinf, const hnode &B, ms_slice_node &&A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            CALL_AND_RETHROW(evaluate_leaf(h.mode_operators(), cinf(), B(), A(), buffer, hspf(), compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
        static inline void evaluate_leaf(const ms_soptype &h, const ms_cinfnode &cinf, const ms_hnode &B, const ms_hnode &A, size_t i, size_t c, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            // now we need to iterate over each of the set indices and perform the correct contractions storing them in the correct locations
            size_t j = cinf()[i][c].col();
            CALL_AND_RETHROW(evaluate_leaf(h.mode_operators()[i][c], cinf()[i][c], B(i), A(j), buffer, hspf(), compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
#ifdef USE_OPENMP
        static inline void evaluate_leaf(const ms_soptype &h, const ms_cinfnode &cinf, const ms_hnode &B, const ms_hnode &A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t set_var_nthreads =1)
#else
        static inline void evaluate_leaf(const ms_soptype &h, const ms_cinfnode &cinf, const ms_hnode &B, const ms_hnode &A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t /*set_var_nthreads*/ =1)
#endif
        {
#ifdef USE_OPENMP
#pragma omp parallel for num_threads( (buffer.buf < set_var_nthreads ? buffer.buf : set_var_nthreads)) default(shared) if (buffer.buf > 1 && cinf().size() > 1)  schedule(static)
#endif
            for (size_t row = 0; row < cinf().size(); ++row)
            {
#ifdef USE_OPENMP
                size_t tid = omp_get_thread_num();
#else
                size_t tid = 0;
#endif
                for (size_t ci = 0; ci < cinf()[row].size(); ++ci)
                {
                    size_t col = cinf()[row][ci].col();
                    CALL_AND_RETHROW(evaluate_leaf(h.mode_operators()[row][ci], cinf()[row][ci], B(row), A(col), buffer, hspf()[row][ci], compute_identity, update_all, operator_sum_nthreads, tid));
                }
            }
        }

    protected:
        template <typename spfnode>
        static inline void _evaluate_branch(const cinftype &cinf, const hdata &A, buffer_type &buffer, spfnode &hspf, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a single particle operator given a branch site tensor and surrounding environment tensors.");
#endif        
            try
            {
                CALL_AND_HANDLE(hspf().resize_matrices(A.size(1), A.size(1)), "Failed to resize the single-particle Hamiltonian operator matrices.");
                const auto &a = A.as_matrix();
#ifdef USE_OPENMP
                #pragma omp parallel for num_threads(operator_sum_nthreads) default(shared) schedule(static)
#endif
                for (size_type ind = 0; ind < cinf.nterms(); ++ind)
                {    
#ifdef USE_OPENMP
                    size_type ti = omp_get_thread_num() + tid*operator_sum_nthreads;
#else
                    size_type ti = tid*operator_sum_nthreads;
#endif                    
                    CALL_AND_HANDLE(buffer.HA[ti].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                    CALL_AND_HANDLE(buffer.temp[ti].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                    if (update_all || cinf[ind].is_time_dependent())
                    {
                        if (!cinf[ind].is_identity_spf())
                        {
                            hspf().spf(ind).fill_zeros();
                            for (size_type i = 0; i < cinf[ind].nspf_terms(); ++i)
                            {
                                CALL_AND_HANDLE(kron_prod(hspf, cinf, ind, i, A, buffer.temp[ti], buffer.HA[ti]), "Failed to apply kronecker product operator.");
                                CALL_AND_HANDLE(hspf().spf(ind) += cinf[ind].spf_coeff(i) * adjoint(a) * buffer.HA[ti], "Failed to apply matrix product to obtain result.");
                            }
                        }
                    }
                }
            }
            catch (const common::invalid_value &ex)
            {
                logging::error(ex.what());
                RAISE_NUMERIC("evaluating single particle operator at a branch node.");
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate single particle operator at a branch node.");
            }
        }

        template <typename spfnode>
        static inline void _evaluate_branch(const cinftype &cinf, const hdata &B, const hdata &A, buffer_type &buffer, spfnode &hspf, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a single particle operator given a branch site tensor and surrounding environment tensors.");
#endif     
            try
            {
                CALL_AND_HANDLE(buffer.HA[tid].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                CALL_AND_HANDLE(buffer.temp[tid].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                CALL_AND_HANDLE(hspf().resize_matrices(B.size(1), A.size(1)), "Failed to resize the single-particle Hamiltonian operator matrices.");

                const auto &b = B.as_matrix();
                if (update_all)
                {
                    CALL_AND_HANDLE(kpo::kpo_id(hspf, A, buffer.temp[tid], buffer.HA[tid]), "Failed to apply kronecker product operator.");
                    CALL_AND_HANDLE(hspf().spf_id() = adjoint(b) * buffer.HA[tid], "Failed to compute the id matrix term.");
                }

#ifdef USE_OPENMP
                #pragma omp parallel for num_threads(operator_sum_nthreads) default(shared) schedule(static)
#endif
                for (size_type ind = 0; ind < cinf.nterms(); ++ind)
                {
                    if (update_all || cinf[ind].is_time_dependent())
                    {
#ifdef USE_OPENMP
                        size_type ti2 = omp_get_thread_num() + tid*operator_sum_nthreads;
#else
                        size_type ti2 = tid*operator_sum_nthreads;
#endif
                        CALL_AND_HANDLE(buffer.HA[ti2].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                        CALL_AND_HANDLE(buffer.temp[ti2].resize(A.size(0), A.size(1)), "Failed to resize hamiltonian action object.");
                        if (!cinf[ind].is_identity_spf())
                        {
                            hspf().spf(ind).fill_zeros();
                            for (size_type i = 0; i < cinf[ind].nspf_terms(); ++i)
                            {
                                CALL_AND_HANDLE(kron_prod(hspf, cinf, ind, i, B, A, buffer.temp[ti2], buffer.HA[ti2]), "Failed to apply kronecker product operator.");
                                CALL_AND_HANDLE(hspf().spf(ind) += cinf[ind].spf_coeff(i) * adjoint(b) * buffer.HA[ti2], "Failed to apply matrix product to obtain result.");
                            }
                        }
                    }
                }
            }
            catch (const common::invalid_value &ex)
            {
                logging::error(ex.what());
                RAISE_NUMERIC("evaluating single particle operator at a branch node.");
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate single particle operator at a branch node.");
            }
        }

        template <typename spfnode>
        static inline void evaluate_branch(const cinftype &cinf, const hdata &B, const hdata &A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            if (&A == &B && !compute_identity)
            {
                CALL_AND_RETHROW(_evaluate_branch(cinf, A, buffer, hspf, update_all, operator_sum_nthreads, tid));
            }
            else
            {
                CALL_AND_RETHROW(_evaluate_branch(cinf, B, A, buffer, hspf, update_all, operator_sum_nthreads, tid));
            }
        }

    protected:
        template <typename spfnode>
        static inline void evaluate_branch_single_set(const cinfnode &cinf, const hnode &B, const hnode &A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            CALL_AND_RETHROW(evaluate_branch(cinf(), B(), A(), buffer, hspf, compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
        static inline void evaluate_branch_single_set(const cinfnode &cinf, ms_slice_node &&B, ms_slice_node &&A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            CALL_AND_RETHROW(evaluate_branch(cinf(), B(), A(), buffer, hspf, compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
        static inline void evaluate_branch_single_set(const cinfnode &cinf, ms_slice_node &&B, const hnode &A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            CALL_AND_RETHROW(evaluate_branch(cinf(), B(), A(), buffer, hspf, compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
        static inline void evaluate_branch_single_set(const cinfnode &cinf, const hnode &B, ms_slice_node &&A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            CALL_AND_RETHROW(evaluate_branch(cinf(), B(), A(), buffer, hspf, compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
        static inline void evaluate_branch(const ms_cinfnode &cinf, const ms_hnode &B, const ms_hnode &A, size_t i, size_t c, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            size_t j = cinf()[i][c].col();
            CALL_AND_RETHROW(evaluate_branch(cinf()[i][c], B(i), A(j), buffer, hspf, compute_identity, update_all, operator_sum_nthreads, tid));
        }

        template <typename spfnode>
#ifdef USE_OPENMP
        static inline void evaluate_branch(const ms_cinfnode &cinf, const ms_hnode &B, const ms_hnode &A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t set_var_nthreads =1)
#else
        static inline void evaluate_branch(const ms_cinfnode &cinf, const ms_hnode &B, const ms_hnode &A, buffer_type &buffer, spfnode &hspf, bool compute_identity = false, bool update_all = true, size_t operator_sum_nthreads = 1, size_t /*set_var_nthreads*/ =1)
#endif
        {
#ifdef USE_OPENMP
#pragma omp parallel for default(shared) if (buffer.buf > 1 && cinf().size() > 1) num_threads( (buffer.buf < set_var_nthreads ? buffer.buf : set_var_nthreads)) schedule(static)
#endif
            for (size_t row = 0; row < cinf().size(); ++row)
            {
#ifdef USE_OPENMP
                size_t tid = omp_get_thread_num();
#else
                size_t tid = 0;
#endif
                for (size_t ci = 0; ci < cinf()[row].size(); ++ci)
                {
                    size_t col = cinf()[row][ci].col();
                    ms_sop_env_slice<T, backend> hslice(hspf, row, ci);
                    CALL_AND_RETHROW(evaluate_branch(cinf()[row][ci], B(row), A(col), buffer, hslice, compute_identity, update_all, operator_sum_nthreads, tid));
                }
            }
        }

    public:
        // kronecker product operators for the operator ype
        template <typename spfnode>
        static void kron_prod(const spfnode &op, const cinftype &cinf, size_type ind, size_type ri, const hdata &A, mat &temp, mat &res)
        {
            CALL_AND_RETHROW(kpo::kron_prod([&op](size_t nu, size_t cri)
                                            { return op[nu]().spf(cri); }, cinf[ind].spf_indexing()[ri], A, temp, res));
        }

        // kronecker product operators for the operator type
        template <typename spfnode>
        static void kron_prod(const spfnode &op, const cinftype &cinf, size_type ind, size_type ri, const hdata &B, const hdata &A, mat &temp, mat &res)
        {
            ASSERT(op().has_identity(), "Cannot apply rectangular hamiltonian without having identity matrices bound");
            CALL_AND_RETHROW(kpo::kron_prod([&op](size_t nu, size_t cri)
                                            { return op[nu]().spf(cri); }, [&op](size_t nu)
                                            { return op[nu]().spf_id(); }, cinf[ind].spf_indexing()[ri], B, A, temp, res));
        }

    }; // class single_particle_operator engine
} // namespace ttns

#endif // PYTTN_TTNS_LIB_CORE_SINGLE_PARTICLE_OPERATOR_HPP_//
