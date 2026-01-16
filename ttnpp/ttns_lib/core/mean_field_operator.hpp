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

#ifndef PYTTN_TTNS_LIB_CORE_MEAN_FIELD_OPERATOR_HPP_
#define PYTTN_TTNS_LIB_CORE_MEAN_FIELD_OPERATOR_HPP_

#include "sop_env_node.hpp"
#include "kronecker_product_operator_helper.hpp"
#include "matrix_element_buffer.hpp"

namespace ttns
{

    template <typename T, typename backend>
    class mean_field_operator_engine
    {
        using kpo = kronecker_product_operator_mel<T, backend>;
        using hnode = ttn_node<T, backend>;
        using hdata = ttn_node_data<T, backend>;

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

    protected:
        static inline size_type contraction_buffer_size(const hdata &A, bool use_capacity = false)
        {
            size_type maxdim = 0;
            for (size_type mode = 0; mode < A.nmodes(); ++mode)
            {
                auto _A = A.as_rank_3(mode, use_capacity);
                size_type dim = _A.shape(0) * _A.shape(1) * _A.shape(1);
                if (dim > maxdim)
                {
                    maxdim = dim;
                }
            }
            return maxdim;
        }

    public:
        static inline size_type contraction_buffer_size(const hnode &A, bool use_capacity = false)
        {
            return contraction_buffer_size(A(), use_capacity);
        }

        static inline size_type contraction_buffer_size(const ms_hnode &A, bool use_capacity = false)
        {
            size_type maxdim = 0;
            for (size_t i = 0; i < A.nset(); ++i)
            {
                size_type dim = contraction_buffer_size(A(i), use_capacity);
                if (dim > maxdim)
                {
                    maxdim = dim;
                }
            }
            return maxdim;
        }

    public:
        template <typename opnode>
        static inline void evaluate(const cinfnode &hinf, const hnode &A, buffer_type& buffer, opnode &h, size_t operator_sum_nthreads=1, size_t /*set_var_nthreads*/=1)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a mean field operator given a site tensor and surrounding environment tensors.");
#endif
            if (!h.is_root())
            {
                size_type mode = h.child_id();
                const auto &hinf_p = hinf.parent();
                CALL_AND_RETHROW(_evaluate_term(hinf(), hinf_p(), mode, A(), buffer, h, operator_sum_nthreads));
            }
        }

#ifdef USE_OPENMP
        template <typename opnode>
        static inline void evaluate(const ms_cinfnode &hinf, const ms_hnode &A, buffer_type& buffer, opnode &h, size_t operator_sum_nthreads=1, size_t set_var_nthreads=1)
#else
        template <typename opnode>
        static inline void evaluate(const ms_cinfnode &hinf, const ms_hnode &A, buffer_type& buffer, opnode &h, size_t operator_sum_nthreads=1, size_t /*set_var_nthreads*/=1)
#endif
        {
#ifdef TRACE_LOG
            logging::trace("evaluating the value of a mean field operator given a site tensor and surrounding environment tensors.");
#endif
            if (!h.is_root())
            {
                size_type mode = h.child_id();
                const auto &hinf_p = hinf.parent();

#ifdef USE_OPENMP
#pragma omp parallel for default(shared) if (buffer.buf > 1 && hinf().size() > 1) num_threads((buffer.buf < set_var_nthreads ? buffer.buf : set_var_nthreads))
#endif
                for (size_t row = 0; row < hinf().size(); ++row)
                {
                    for (size_t ci = 0; ci < hinf()[row].size(); ++ci)
                    {
#ifdef USE_OPENMP
                        size_t tid = omp_get_thread_num();
#else
                        size_t tid = 0;
#endif
                        size_t col = hinf()[row][ci].col();
                        ms_sop_env_slice<T, backend> hslice(h, row, ci);

                        if (row == col)
                        {
                            _evaluate_term(hinf()[row][ci], hinf_p()[row][ci], mode, A(row), buffer, hslice, operator_sum_nthreads, tid);
                        }
                        else
                        {
                            _evaluate_term(hinf()[row][ci], hinf_p()[row][ci], mode, A(row), A(col), buffer, hslice, operator_sum_nthreads, tid);
                        }
                    }
                }
            }
        }

    protected:
        template <typename opnode>
        static inline void _evaluate_term(const cinftype &hinf, const cinftype &hinf_p, size_type mode, const hdata &A, buffer_type& buffer, opnode &h, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            try
            {
                const auto &h_p = h.parent();

#ifdef USE_OPENMP
                #pragma omp parallel for num_threads(operator_sum_nthreads) default(shared) schedule(dynamic, 1)
#endif
                for (size_type ind = 0; ind < hinf.nterms(); ++ind)
                {
#ifdef USE_OPENMP
                    size_type ti = omp_get_thread_num() + tid*operator_sum_nthreads;
#else
                    size_type ti = tid*operator_sum_nthreads;
#endif                                    
                    CALL_AND_HANDLE(buffer.HA[ti].resize(A.size(0), A.size(1)), "failed to resize working buffers.");
                    CALL_AND_HANDLE(buffer.temp[ti].resize(A.size(0), A.size(1)), "failed to resize working buffers.");

                    // if the mean field operator is the identity then we don't need to do anything.
                    if (!hinf[ind].is_identity_mf())
                    {
                        h().mf(ind).fill_zeros();
                        for (size_type it = 0; it < hinf[ind].nmf_terms(); ++it)
                        {
                            size_type pi = hinf[ind].mf_indexing()[it].parent_index();

                            if (!hinf_p[pi].is_identity_mf())
                            {
                                CALL_AND_HANDLE(kron_prod(h, hinf, ind, it, A, buffer.HA[ti], buffer.temp[ti]), "Failed to evaluate action of kronecker product operator.");
                                CALL_AND_HANDLE(buffer.HA[ti] = buffer.temp[ti] * trans(h_p().mf(pi)), "Failed to apply action of parent mean field operator.");
                            }
                            else
                            {
                                CALL_AND_HANDLE(kron_prod(h, hinf, ind, it, A, buffer.temp[ti], buffer.HA[ti]), "Failed to evaluate action of kronecker product operator.");
                            }

                            CALL_AND_HANDLE(buffer.temp[ti] = conj(A.as_matrix()), "Failed to compute conjugate of the A matrix.");

                            try
                            {
                                auto _A = A.as_rank_3(mode);
                                auto _HA = buffer.HA[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));
                                auto _temp = buffer.temp[ti].reinterpret_shape(_A.shape(0), _A.shape(1), _A.shape(2));

                                CALL_AND_HANDLE(h().mf(ind) += hinf[ind].mf_coeff(it) * (contract(_temp, 0, 2, _HA, 0, 2)), "Failed when evaluating the final contraction.");
                            }
                            catch (const std::exception &ex)
                            {
                                logging::error(ex.what());
                                RAISE_EXCEPTION("Failed to form temporary reinterpreted tensors and perform contraction over the outer indices to form the mean field Hamiltonian.");
                            }
                        }
                    }
                }
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate mean field operator at a node.");
            }
        }

        template <typename opnode>
        static inline void _evaluate_term(const cinftype &hinf, const cinftype &hinf_p, size_type mode, const hdata &B, const hdata &A, buffer_type& buffer, opnode &h, size_t operator_sum_nthreads = 1, size_t tid = 0)
        {
            if (&A == &B)
            {
                CALL_AND_RETHROW(return _evaluate_term(hinf, hinf_p, mode, A, buffer, h, operator_sum_nthreads, tid));
            }
            try
            {

                const auto &h_p = h.parent();
                {
                    size_type ti = tid*operator_sum_nthreads;
                    CALL_AND_HANDLE(buffer.HA[ti].resize(A.size(0), A.size(1)), "failed to resize working buffers.");
                    CALL_AND_HANDLE(buffer.temp[ti].resize(A.size(0), A.size(1)), "failed to resize working buffers.");

                    CALL_AND_HANDLE(kpo::kpo_id(h_p, A, mode, buffer.HA[ti], buffer.temp[ti]), "Failed to apply kronecker product operator.");
                    CALL_AND_HANDLE(buffer.HA[ti] = buffer.temp[ti] * trans(h_p().mf_id()), "Failed to apply action of parent mean field operator.");
                    CALL_AND_HANDLE(buffer.temp[ti] = conj(B.as_matrix()), "Failed to compute conjugate of the A matrix.");

                    try
                    {
                        auto _A = A.as_rank_3(mode);
                        auto _B = B.as_rank_3(mode);
                        auto _HA = buffer.HA[ti].reinterpret_shape(_B.shape(0), _A.shape(1), _B.shape(2));
                        auto _temp = buffer.temp[ti].reinterpret_shape(_B.shape(0), _B.shape(1), _B.shape(2));

                        CALL_AND_HANDLE(h().mf_id() = (contract(_temp, 0, 2, _HA, 0, 2)), "Failed when evaluating the final contraction.");
                    }
                    catch (const std::exception &ex)
                    {
                        logging::error(ex.what());
                        RAISE_EXCEPTION("Failed to form temporary reinterpreted tensors and perform contraction over the outer indices to form the mean field Hamiltonian.");
                    }
                }

#ifdef USE_OPENMP
                #pragma omp parallel for num_threads(operator_sum_nthreads) default(shared) schedule(dynamic, 1)
#endif
                for (size_type ind = 0; ind < hinf.nterms(); ++ind)
                {
                    // if the mean field operator is the identity then we don't need to do anything.
                    if (!hinf[ind].is_identity_mf())
                    {
#ifdef USE_OPENMP
                        size_type ti = omp_get_thread_num() + tid*operator_sum_nthreads;
#else
                        size_type ti = tid*operator_sum_nthreads;
#endif                
                        CALL_AND_HANDLE(buffer.HA[ti].resize(A.size(0), A.size(1)), "failed to resize working buffers.");
                        CALL_AND_HANDLE(buffer.temp[ti].resize(A.size(0), A.size(1)), "failed to resize working buffers.");

                        h().mf(ind).fill_zeros();
                        for (size_type it = 0; it < hinf[ind].nmf_terms(); ++it)
                        {
                            size_type pi = hinf[ind].mf_indexing()[it].parent_index();

                            if (hinf[ind].mf_indexing()[it].sibling_indices().size() == 0)
                            {
                                CALL_AND_HANDLE(kpo::kpo_id(h_p, A, mode, buffer.HA[ti], buffer.temp[ti]), "Failed to apply kronecker product operator.");
                            }
                            else
                            {
                                CALL_AND_HANDLE(kron_prod(h, hinf, ind, it, B, A, mode, buffer.HA[ti], buffer.temp[ti]), "Failed to evaluate action of kronecker product operator.");
                            }
                            if (!hinf_p[pi].is_identity_mf())
                            {
                                CALL_AND_HANDLE(buffer.HA[ti] = buffer.temp[ti] * trans(h_p().mf(pi)), "Failed to apply action of parent mean field operator.");
                            }
                            else
                            {
                                CALL_AND_HANDLE(buffer.HA[ti] = buffer.temp[ti] * trans(h_p().mf_id()), "Failed to apply action of parent mean field operator.");
                            }

                            CALL_AND_HANDLE(buffer.temp[ti] = conj(B.as_matrix()), "Failed to compute conjugate of the A matrix.");

                            try
                            {
                                auto _A = A.as_rank_3(mode);
                                auto _B = B.as_rank_3(mode);

                                auto _HA = buffer.HA[ti].reinterpret_shape(_B.shape(0), _A.shape(1), _B.shape(2));
                                auto _temp = buffer.temp[ti].reinterpret_shape(_B.shape(0), _B.shape(1), _B.shape(2));

                                CALL_AND_HANDLE(h().mf(ind) += hinf[ind].mf_coeff(it) * (contract(_temp, 0, 2, _HA, 0, 2)), "Failed when evaluating the final contraction.");
                            }
                            catch (const std::exception &ex)
                            {
                                logging::error(ex.what());
                                RAISE_EXCEPTION("Failed to form temporary reinterpreted tensors and perform contraction over the outer indices to form the mean field Hamiltonian.");
                            }
                        }
                    }
                }
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate mean field operator at a node.");
            }
        }

        template <typename opnode>
        static inline void evaluate_term(const cinftype &hinf, const cinftype &hinf_p, size_type mode, const hdata &B, const hdata &A, buffer_type& buffer, opnode &h, size_t operator_sum_nthreads = 1)
        {
#ifdef TRACE_LOG
            logging::trace("evaluating a single term of in the mean field operator given a site tensor and surrounding environment tensors.");
#endif
            if (&A == &B)
            {
                CALL_AND_RETHROW(_evaluate_term(hinf, hinf_p, mode, A, buffer, h, operator_sum_nthreads));
            }
            else
            {
                CALL_AND_RETHROW(_evaluate_term(hinf, hinf_p, mode, B, buffer, h, operator_sum_nthreads));
            }
        }

    public:
        template <typename opnode>
        static void kron_prod(const opnode &op, const cinftype &cinf, size_type ind, size_type ri, const hdata &A, mat &temp, mat &res)
        {
            CALL_AND_RETHROW(kpo::kron_prod([&op](size_t nu, size_t cri)
                                            { return op.parent()[nu]().spf(cri); }, cinf[ind].mf_indexing()[ri].sibling_indices(), A, temp, res));
        }

        // kronecker product operators for the operator type
        template <typename spfnode>
        static void kron_prod(const spfnode &op, const cinftype &cinf, size_type ind, size_type ri, const hdata &B, const hdata &A, size_type mode, mat &temp, mat &res)
        {
            ASSERT(op().has_identity(), "Cannot apply rectangular hamiltonian without having identity matrices bound");
            ASSERT(cinf[ind].mf_indexing()[ri].sibling_indices().size() != 0, "Cannot apply kron prod if all spf matrices are identity.");
            CALL_AND_RETHROW(
                kpo::kron_prod(
                    [&op](size_t nu, size_t cri)
                    { return op.parent()[nu]().spf(cri); },
                    [&op](size_t nu)
                    { return op.parent()[nu]().spf_id(); },
                    cinf[ind].mf_indexing()[ri].sibling_indices(), B, A, mode, temp, res));
        }

    }; // class mean field operator engine

} // namespace ttns

#endif // PYTTN_TTNS_LIB_CORE_MEAN_FIELD_OPERATOR_HPP_//
