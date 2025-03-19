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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_CONTRACTION_332_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_CONTRACTION_332_HPP_

#include "../../../backends/cuda_backend.hpp"
#include "../../../backends/blas_backend.hpp"
#include "../expression_base.hpp"

namespace linalg
{
    namespace expression_templates
    {

        template <typename backend>
        struct requires_working_buffer
        {
            static constexpr bool value() { return false; }
        };

#ifdef PYTTN_BUILD_CUDA
        template <>
        struct requires_working_buffer<cuda_backend>
        {
            static constexpr bool value() { return true; }
        };
#endif

        template <typename T1, typename T2>
        class tensor_contraction_332 : public tensor_contraction_expression, public expression_base<tensor_contraction_332<T1, T2>, true>, public dense_type
        {
        public:
            static_assert(is_same_value<T1, T2>::value, "Failed to construct contraction_332 object.  The two input tensors must have the same value_type.");
            static_assert(is_same_backend<T1, T2>::value, "Failed to construct contraction_332 object.  The two input tensors must have the same backend type.");
            static_assert(T1::rank == 3 && T2::rank == 3, "Failed to construct contraction_332 object.  The two input tensor must be rank 3 and rank 2.");
            static_assert(is_dense<T1>::value && is_dense<T2>::value, "Failed to construct contraction_332 object.  The two input tensor must both be dense tensors.");

            using base_type = expression_base<tensor_contraction_332<T1, T2>, true>;
            using left_type = T1;
            using right_type = T2;
            using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
            using value_ptr = typename std::add_pointer<typename std::add_const<value_type>::type>::type;
            using backend_type = typename traits<T1>::backend_type;
            using size_type = typename backend_type::size_type;
            using shape_type = std::array<size_type, 2>;
            using ttype = typename backend_type::transform_type;

        protected:
            value_type m_coeff;
            value_ptr m_Abuffer;
            value_ptr m_Bbuffer;

            size_type m_Atotsize;
            size_type m_Btotsize;

            bool m_outer;
            ttype m_opA, m_opB;
            size_type m_t, m_m, m_n, m_k, m_ldA, m_ldB;
            size_type m_strideA, m_strideB;

            value_type *m_working_buffer;
            value_type *m_conj_buffer;

            size_type m_working_buffer_size;
            size_type m_conj_buffer_size;

            bool m_allocated_working;
            bool m_allocated_conj;

        protected:
            void initialise_state(const left_type &A, const right_type &B, size_type ind, bool conjA, bool conjB)
            {
                try
                {
                    ASSERT(ind <= 2, "Failed to construct rank 3, rank 3 tensor contraction expression.  The index not being contracted is out of bounds.");
                    for (size_type i = 0; i < 3; ++i)
                    {
                        if (i != ind)
                        {
                            ASSERT(A.shape(i) == B.shape(i), "Failed to construct rank 3, rank 3 tensor contraction expression.  The contraction indices do not have the same dimensions.");
                        }
                    }
                    this->m_shape[0] = A.shape(ind);
                    this->m_shape[1] = B.shape(ind);

                    switch (ind)
                    {
                    case (0):
                    {
                        // in this case C = A B^T with optional conjugations
                        m_outer = false;
                        m_Abuffer = A.buffer();
                        m_ldA = A.shape(1) * A.shape(2);
                        m_Atotsize = A.size();
                        m_Bbuffer = B.buffer();
                        m_ldB = B.shape(1) * B.shape(2);
                        m_Btotsize = B.size();

                        m_opA = conjA ? backend_type::op_c : backend_type::op_n;
                        m_opB = conjB ? backend_type::op_h : backend_type::op_t;

                        m_m = B.shape(0);
                        m_k = A.shape(1) * A.shape(2);
                        m_n = A.shape(0);
                        break;
                    };
                    case (1):
                    {
                        // in this case C_k = A_k B_k^T with optional conjugations
                        m_outer = true;
                        m_Abuffer = A.buffer();
                        m_ldA = A.shape(2);
                        m_Atotsize = A.size();
                        m_Bbuffer = B.buffer();
                        m_ldB = B.shape(2);
                        m_Btotsize = B.size();

                        m_opA = conjA ? backend_type::op_c : backend_type::op_n;
                        m_opB = conjB ? backend_type::op_h : backend_type::op_t;

                        m_m = B.shape(1);
                        m_k = A.shape(2);
                        m_n = A.shape(1);
                        m_t = A.shape(0);
                        m_strideA = A.shape(1) * A.shape(2);
                        m_strideB = B.shape(1) * B.shape(2);
                        break;
                    };
                    case (2):
                    {
                        m_outer = false;
                        m_Abuffer = A.buffer();
                        m_ldA = A.shape(2);
                        m_Bbuffer = B.buffer();
                        m_ldB = B.shape(2);

                        m_opA = conjA ? backend_type::op_h : backend_type::op_t;
                        m_Atotsize = A.size();
                        m_opB = conjB ? backend_type::op_c : backend_type::op_n;
                        m_Btotsize = B.size();

                        m_m = B.shape(2);
                        m_k = A.shape(0) * A.shape(1);
                        m_n = A.shape(2);
                        break;
                    };
                    };
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to initialise the state of a rank 3 and rank 3 tensor contraction expression.");
                }
            }

        public:
            tensor_contraction_332(value_type coeff, const left_type &A, const right_type &B, size_type ind, bool conjA = false, bool conjB = false)
            try : base_type(shape_type{{0, 0}}), m_coeff(coeff), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_outer(false), m_strideA(0), m_strideB(0), m_working_buffer(nullptr), m_conj_buffer(nullptr), m_working_buffer_size(0), m_conj_buffer_size(0)
            {
                CALL_AND_HANDLE(initialise_state(A, B, ind, conjA, conjB), "initialise_state call failed.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct tensor_contraction_332 object.");
            }

            template <typename wbuff>
            tensor_contraction_332(value_type coeff, const left_type &A, const right_type &B, size_type ind, wbuff &working, bool conjA = false, bool conjB = false)
            try : base_type(shape_type{{0, 0}}), m_coeff(coeff), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_outer(false), m_strideA(0), m_strideB(0), m_working_buffer(nullptr), m_conj_buffer(nullptr), m_working_buffer_size(0), m_conj_buffer_size(0)
            {
                CALL_AND_HANDLE(initialise_state(A, B, ind, conjA, conjB), "initialise_state call failed.");
                CALL_AND_HANDLE(bind_workspace(working), "Failed to bind working buffer.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct tensor_contraction_332 object.");
            }

            template <typename wbuff, typename cbuff>
            tensor_contraction_332(value_type coeff, const left_type &A, const right_type &B, size_type ind, wbuff &working, cbuff &cworking, bool conjA = false, bool conjB = false)
            try : base_type(shape_type{{0, 0}}), m_coeff(coeff), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_outer(false), m_strideA(0), m_strideB(0), m_working_buffer(nullptr), m_conj_buffer(nullptr), m_working_buffer_size(0), m_conj_buffer_size(0)
            {
                CALL_AND_HANDLE(initialise_state(A, B, ind, conjA, conjB), "initialise_state call failed.");
                CALL_AND_HANDLE(bind_workspace(working), "Failed to bind working buffer.");
                CALL_AND_HANDLE(bind_conjugate_workspace(cworking), "Failed to bind conjugate working buffer.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct tensor_contraction_332 object.");
            }

            tensor_contraction_332(const tensor_contraction_332 &o, const value_type &factor)
            try : base_type(o.shape()), m_coeff(o.m_coeff * factor), m_Abuffer(o.m_Abuffer), m_Bbuffer(o.m_Bbuffer), m_Atotsize(o.m_Atotsize), m_Btotsize(o.m_Btotsize), m_outer(o.m_outer), m_opA(o.m_opA), m_opB(o.m_opB), m_t(o.m_t), m_m(o.m_m), m_n(o.m_n), m_k(o.m_k), m_ldA(o.m_ldA), m_ldB(o.m_ldB), m_strideA(o.m_strideA), m_strideB(o.m_strideB), m_working_buffer(o.m_working_buffer), m_conj_buffer(o.m_conj_buffer), m_working_buffer_size(o.m_working_buffer_size), m_conj_buffer_size(o.m_conj_buffer_size) {}
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct tensor_contraction_332 object.");
            }

            template <typename working_type>
            tensor_contraction_332 &bind_workspace(working_type &working)
            {
                static_assert(is_dense<working_type>(), "Failed to construct contraction_332 object.  The input temporary object type is not a dense tensor.");
                static_assert(std::is_same<typename T1::value_type, typename working_type::value_type>::value, "Failed to construct contraction_332 object.  The working_type tensor must have the same value_type as the input tensors.");
                static_assert(std::is_same<typename T1::backend_type, typename working_type::backend_type>::value, "Failed to construct contraction_332 object.  The working_type tensor must have the same backend_type as the input tensors.");
                try
                {
                    ASSERT(m_working_buffer == nullptr, "A working buffer has already been bound.");
                    m_working_buffer = working.buffer();
                    m_working_buffer_size = working.size();
                    return *this;
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to bind workspace buffer.");
                }
            }

            void unbind_workspace()
            {
                m_working_buffer = nullptr;
                m_working_buffer_size = 0;
            }

            template <typename conj_type>
            tensor_contraction_332 &bind_conjugate_workspace(conj_type &conj)
            {
                static_assert(is_dense<conj_type>(), "Failed to construct contraction_332 object.  The input conjorary object type is not a dense tensor.");
                static_assert(std::is_same<typename T1::value_type, typename conj_type::value_type>::value, "Failed to construct contraction_332 object.  The conj_type tensor must have the same value_type as the input tensors.");
                static_assert(std::is_same<typename T1::backend_type, typename conj_type::backend_type>::value, "Failed to construct contraction_332 object.  The conj_type tensor must have the same backend_type as the input tensors.");
                try
                {
                    ASSERT(m_conj_buffer == nullptr, "A conjugate workspace buffer has already been bound.");
                    m_conj_buffer = conj.buffer();
                    m_conj_buffer_size = conj.size();
                    return *this;
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to bind conjugate workspace buffer.");
                }
            }

            void unbind_conjugate_workspace()
            {
                m_conj_buffer = nullptr;
                m_conj_buffer_size = 0;
            }

            template <typename T3>
            void applicative(T3 &res, value_type beta = 0.0, value_type coeff_scale = 1.0)
            {
                /// static_assert(std::is_same<typename T3::value_type, typename T2::value_type>::value, "Failed to construct evaluate operator for tensor_contraction_332 object.  The result_type must have a compatible value_type.");
                static_assert(std::is_same<typename T3::backend_type, typename T2::backend_type>::value, "Failed to construct evaluate operator for tensor_contraction_332 object.  The result_type must have a compatible backend type.");
                static_assert(T3::rank == 2, "Failed to construct evaluate operator for tensor_contraction_332 object.  The result type must be a rank 3 tensor.");
                static_assert(is_dense<T3>::value, "Failed to construct evaluate operator for tensor_contraction_332 object.  The result type must be a dense tensor.");

                try
                {
                    ASSERT(res.buffer() != m_Abuffer && res.buffer() != m_Bbuffer, "Failed to compute rank 3 rank 3 tensor contraction.  The rank 3 rank 3 tensor contraction does not support inplace products.");
                    value_type coeff = m_coeff * coeff_scale;

                    if (m_opA == backend_type::op_c && m_opB == backend_type::op_c)
                    {
                        ttype opA = backend_type::op_n;
                        ttype opB = backend_type::op_n;
                        if (is_complex<value_type>::value)
                        {
                            coeff = conj(coeff);
                            if (beta != value_type(0.0))
                            {
                                beta = conj(beta);
                                CALL_AND_HANDLE(backend_type::complex_conjugate(res.size(), res.buffer(), res.buffer()), "Failed to compute complex conjugate of result");
                            }
                            CALL_AND_HANDLE(eval_contraction(res, beta, coeff, m_Abuffer, m_Bbuffer, opA, opB), "Failed to evaluate contraction for rank 3 rank 3 contraction.");
                            CALL_AND_HANDLE(backend_type::complex_conjugate(res.size(), res.buffer(), res.buffer()), "Failed to compute complex conjugate of result");
                        }
                        else
                        {
                            CALL_AND_HANDLE(eval_contraction(res, beta, coeff, m_Abuffer, m_Bbuffer, opA, opB), "Failed to evaluate contraction for rank 3 rank 3 contraction.");
                        }
                    }
                    else
                    {
                        ttype opA = m_opA;
                        ttype opB = m_opB;
                        value_ptr Abuffer = m_Abuffer;
                        value_ptr Bbuffer = m_Bbuffer;
                        if (m_opA == backend_type::op_c && m_opB != backend_type::op_c)
                        {
                            opA = backend_type::op_n;
                            if (is_complex<value_type>::value)
                            {
                                ASSERT(m_conj_buffer != nullptr, "The conjugate working space buffer has not been bound.");
                                ASSERT(m_conj_buffer_size >= m_Atotsize, "The working space array is not large enough to store temporary objects.");
                                backend_type::complex_conjugate(m_Atotsize, m_Abuffer, m_conj_buffer);
                                Abuffer = static_cast<value_ptr>(m_conj_buffer);
                            }
                        }
                        else if (m_opA != backend_type::op_c && m_opB == backend_type::op_c)
                        {
                            opB = backend_type::op_n;
                            if (is_complex<value_type>::value)
                            {
                                ASSERT(m_conj_buffer != nullptr, "The conjugate working space buffer has not been bound.");
                                ASSERT(m_conj_buffer_size >= m_Btotsize, "The working space array is not large enough to store temporary objects.");
                                backend_type::complex_conjugate(m_Btotsize, m_Bbuffer, m_conj_buffer);
                                Bbuffer = static_cast<value_ptr>(m_conj_buffer);
                            }
                        }

                        CALL_AND_HANDLE(eval_contraction(res, beta, coeff, Abuffer, Bbuffer, opA, opB), "Failed to evaluate contraction for rank 3 rank 3 contraction.");
                        Abuffer = nullptr;
                        Bbuffer = nullptr;
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate rank 3 rank 3 tensor contraction.");
                }
            }

            template <typename T3>
            void addition_applicative(T3 &res, value_type coeff_scale = 1.0) { CALL_AND_RETHROW(applicative(res, 1.0, coeff_scale)); }

            template <typename T3>
            void subtraction_applicative(T3 &res, value_type coeff_scale = 1.0) { CALL_AND_RETHROW(applicative(res, -1.0, coeff_scale)); }

        protected:
            template <typename T3>
            void eval_contraction(T3 &res, value_type beta, value_type coeff, value_ptr Abuffer, value_ptr Bbuffer, ttype opA, ttype opB)
            {
                try
                {
                    if (m_outer)
                    {
                        if (requires_working_buffer<backend_type>::value())
                        {
                            ASSERT(m_working_buffer != nullptr, "The working buffer has not been bound.");
                            ASSERT(m_working_buffer_size >= m_t * m_n * m_m, "Failed to construct rank 3, rank 3 tensor contraction with outer indices contracted.  The working buffer is not large enough to store the result.");
                        }

                        size_type ldres = res.shape(1);
                        size_type strideres = this->m_shape[0] * this->m_shape[1];
                        // first we calculate all of the matrix products using batched_gemm
                        CALL_AND_HANDLE(
                            backend_type::outer_contract(opB, opA, m_m, m_n, m_k, coeff, Bbuffer, m_ldB, m_strideB, Abuffer, m_ldA, m_strideA, beta, m_working_buffer, ldres, strideres, m_t, res.buffer()),
                            "Error when making call to batched_gemm.");
                    }
                    else
                    {
                        size_type ldres = res.shape(1);
                        CALL_AND_HANDLE(
                            backend_type::gemm(opB, opA, m_m, m_n, m_k, coeff, Bbuffer, m_ldB, Abuffer, m_ldA, beta, res.buffer(), ldres),
                            "Error when making call to gemm.");
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate contraction.");
                }
            }
        }; // tensor_contraction_332

    } // namespace expression_templates
    template <typename T1, typename T2>
    struct traits<expression_templates::tensor_contraction_332<T1, T2>>
    {
        using value_type = typename traits<T1>::value_type;
        using backend_type = typename traits<T1>::backend_type;
        using shape_type = std::array<typename backend_type::size_type, 2>;
        using const_shape_reference = const shape_type &;
        static constexpr size_t rank = 2;
    };
} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_CONTRACTION_332_HPP_//
