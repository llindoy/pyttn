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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_CONTRACTION_1_MT_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_CONTRACTION_1_MT_HPP_

#include "../expression_base.hpp"

namespace linalg
{
    namespace expression_templates
    {

        template <typename T1, typename T2>
        class tensor_contraction_1_mt : public tensor_contraction_expression, public expression_base<tensor_contraction_1_mt<T1, T2>, true>, dense_type
        {
        public:
            static_assert(is_same_value<T1, T2>::value, "Failed to construct contraction_1_mt object.  The two input tensors must have the same value_type.");
            static_assert(is_same_backend<T1, T2>::value, "Failed to construct contraction_1_mt object.  The two input tensors must have the same backend type.");
            static_assert(T1::rank == 3 && T2::rank == 2, "Failed to construct contraction_1_mt object.  The two input tensor must be rank 3 and rank 2.");
            static_assert(is_dense<T1>::value && is_dense<T2>::value, "Failed to construct contraction_1_mt object.  The two input tensor must both be dense tensors.");

            using base_type = expression_base<tensor_contraction_1_mt<T1, T2>, true>;
            using left_type = T1;
            using right_type = T2;
            using value_type = typename std::remove_cv<typename traits<T1>::value_type>::type;
            using value_ptr = typename std::add_pointer<typename std::add_const<value_type>::type>::type;

            using backend_type = typename traits<T1>::backend_type;
            using size_type = typename backend_type::size_type;
            using shape_type = std::array<size_type, 3>;
            using ttype = typename backend_type::transform_type;

        protected:
            value_type m_coeff;
            value_ptr m_Abuffer;
            value_ptr m_Bbuffer;

            value_type *m_working;
            size_type m_working_size;

            bool m_batched;
            ttype m_opA, m_opB;
            size_type m_t, m_m, m_n, m_k, m_ldA, m_ldB;
            size_type m_strideA, m_strideB;
            size_type m_Asize, m_Bsize;

        protected:
            void initialise_state(const left_type &A, const right_type &B, size_type Aind, size_type Bind, bool conjA, bool conjB)
            {
                try
                {
                    ASSERT(Aind <= 2, "The contraction index of the rank 3 tensor is out of bounds.");
                    ASSERT(Bind <= 1, "The contraction index of the rank 2 tensor is out of bounds.");
                    ASSERT(A.shape(Aind) == B.shape(Bind), "The contraction indices do not have the same dimension.");

                    // we determine the type of contraction we are attempting.  In case 0, and case 2 we can evaluate the contraction using a matrix matrix product
                    // so we will use the parameters we would use for the requested matrix matrix product.   In case 1 this must be performed using a batched gemm
                    // call and we need to be a bit more careful.

                    m_Asize = A.size();
                    m_Bsize = B.size();
                    switch (Aind)
                    {
                    case (0):
                    {
                        // in this case we have to evaluate R = op(B^T) A (with possible conjugations of B and A)
                        m_batched = false;
                        m_opA = ((Bind == 0) ? (conjB ? backend_type::op_h : backend_type::op_t) : (conjB ? backend_type::op_c : backend_type::op_n));
                        m_opB = conjA ? backend_type::op_c : backend_type::op_n;

                        m_Abuffer = B.buffer();
                        m_ldA = B.shape(1);
                        m_Bbuffer = A.buffer();
                        m_ldB = A.shape(1) * A.shape(2);

                        m_m = A.shape(1) * A.shape(2);
                        m_n = (Bind == 0 ? B.shape(1) : B.shape(0));
                        m_k = B.shape(Bind);

                        this->m_shape[0] = (Bind == 0) ? B.shape(1) : B.shape(0);
                        this->m_shape[1] = A.shape(1);
                        this->m_shape[2] = A.shape(2);
                        break;
                    };
                    case (1):
                    {
                        // set up the batch parameters
                        m_batched = true;
                        m_t = A.shape(0);
                        m_strideB = A.shape(1) * A.shape(2);

                        // in this case we are evaluating C_i = op(B)^T A_i
                        m_opA = (Bind == 0) ? (conjB ? backend_type::op_h : backend_type::op_t) : (conjB ? backend_type::op_c : backend_type::op_n);
                        m_opB = conjA ? backend_type::op_c : backend_type::op_n;

                        m_Abuffer = B.buffer();
                        m_ldA = B.shape(1);
                        m_Bbuffer = A.buffer();
                        m_ldB = A.shape(2);

                        m_m = A.shape(2);
                        m_n = (Bind == 0 ? B.shape(1) : B.shape(0));
                        m_k = B.shape(Bind);
                        this->m_shape[0] = A.shape(0);
                        this->m_shape[1] = (Bind == 0) ? B.shape(1) : B.shape(0);
                        this->m_shape[2] = A.shape(2);

                        break;
                    };
                    case (2):
                    {
                        // R = A op(B)
                        m_batched = false;
                        m_m = (Bind == 0 ? B.shape(1) : B.shape(0));
                        m_n = A.shape(0) * A.shape(1);
                        m_k = B.shape(Bind);

                        m_Abuffer = A.buffer();
                        m_ldA = A.shape(2);
                        m_Bbuffer = B.buffer();
                        m_ldB = B.shape(1);

                        m_opA = conjA ? backend_type::op_c : backend_type::op_n;
                        m_opB = (Bind == 0) ? (conjB ? backend_type::op_c : backend_type::op_n) : (conjB ? backend_type::op_h : backend_type::op_t);

                        this->m_shape[0] = A.shape(0);
                        this->m_shape[1] = A.shape(1);
                        this->m_shape[2] = (Bind == 0) ? B.shape(1) : B.shape(0);
                        break;
                    };
                    };
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to initialise the state of a rank 3 and rank 2 tensor contraction expression.");
                }
            }

        public:
            tensor_contraction_1_mt(const value_type &coeff, const left_type &A, const right_type &B, size_type Aind, size_type Bind, bool conjA = false, bool conjB = false)
            try : base_type({0, 0, 0}), m_coeff(coeff), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_working(nullptr), m_working_size(0), m_batched(false), m_strideA(0), m_strideB(0)
            {
                CALL_AND_HANDLE(initialise_state(A, B, Aind, Bind, conjA, conjB), "Failed to initialise state of the tensor_contraction_1_mt object.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct tensor_contraction_1_mt object.");
            }

            tensor_contraction_1_mt(const value_type &coeff, const right_type &A, const left_type &B, size_type Aind, size_type Bind, bool conjA = false, bool conjB = false)
            try : tensor_contraction_1_mt(coeff, B, A, Bind, Aind, conjB, conjA) {}
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct tensor_contraction_1_mt object.");
            }

            template <typename wbuff>
            tensor_contraction_1_mt(const value_type &coeff, const left_type &A, const right_type &B, size_type Aind, size_type Bind, wbuff &working, bool conjA = false, bool conjB = false)
            try : base_type({0, 0, 0}), m_coeff(coeff), m_Abuffer(A.buffer()), m_Bbuffer(B.buffer()), m_working(nullptr), m_working_size(0), m_batched(false), m_strideA(0), m_strideB(0)
            {
                CALL_AND_HANDLE(initialise_state(A, B, Aind, Bind, conjA, conjB), "Failed to initialise state of the tensor_contraction_1_mt object.");
                CALL_AND_HANDLE(bind_conjugate_workspace(working), "Failed to bind conjugate workspace buffer.");
            }
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct tensor_contraction_1_mt object.");
            }

            template <typename wbuff>
            tensor_contraction_1_mt(const value_type &coeff, const right_type &A, const left_type &B, size_type Aind, size_type Bind, wbuff &working, bool conjA = false, bool conjB = false)
            try : tensor_contraction_1_mt(coeff, B, A, Bind, Aind, working, conjB, conjA) {}
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct tensor_contraction_1_mt object.");
            }

            tensor_contraction_1_mt(const tensor_contraction_1_mt &o, const value_type &factor)
            try : base_type(o.shape()), m_coeff(o.m_coeff * factor), m_Abuffer(o.m_Abuffer), m_Bbuffer(o.m_Bbuffer), m_working(o.m_working), m_working_size(o.m_working_size), m_batched(o.m_batched), m_opA(o.m_opA), m_opB(o.m_opB), m_t(o.m_t), m_m(o.m_m), m_n(o.m_n), m_k(o.m_k), m_ldA(o.m_ldA), m_ldB(o.m_ldB), m_strideA(o.m_strideA), m_strideB(o.m_strideB), m_Asize(o.m_Asize), m_Bsize(o.m_Bsize) {}
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct tensor_contraction_1_mt object.");
            }

            template <typename conj_type>
            tensor_contraction_1_mt &bind_conjugate_workspace(conj_type &conj)
            {
                static_assert(is_dense<conj_type>(), "Failed to construct contraction_332 object.  The input conjorary object type is not a dense tensor.");
                static_assert(std::is_same<value_type, typename conj_type::value_type>::value, "Failed to construct contraction_332 object.  The conj_type tensor must have the same value_type as the input tensors.");
                static_assert(std::is_same<backend_type, typename conj_type::backend_type>::value, "Failed to construct contraction_332 object.  The conj_type tensor must have the same backend_type as the input tensors.");
                try
                {
                    ASSERT(m_working == nullptr, "A working buffer has already been bound.");
                    m_working = conj.buffer();
                    m_working_size = conj.size();
                    return *this;
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to bind conj array.");
                }
            }

            void unbind_conjugate_workspace()
            {
                m_working = nullptr;
                m_working_size = 0;
            }

            template <typename T3>
            void applicative(T3 &res, value_type beta = 0.0, value_type coeff_scale = 1.0)
            {
                static_assert(is_same_value<T2, T3>::value, "Failed to construct evaluate operator for tensor_contraction_1_mt object.  The result_type must have a compatible value_type.");
                static_assert(is_same_backend<T2, T3>::value, "Failed to construct evaluate operator for tensor_contraction_1_mt object.  The result_type must have a compatible backend type.");
                static_assert(T3::rank == 3, "Failed to construct evaluate operator for tensor_contraction_1_mt object.  The result type must be a rank 3 tensor.");
                static_assert(is_dense<T3>::value, "Failed to construct evaluate operator for tensor_contraction_1_mt object.  The result type must be a dense tensor.");

                try
                {
                    ASSERT(res.buffer() != m_Abuffer && res.buffer() != m_Bbuffer, "The tensor contraction does not support inplace products.");

                    value_type coeff = m_coeff * coeff_scale;

                    // now we need to check whether the tensor can be done using
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
                            CALL_AND_HANDLE(eval_contraction(res, beta, coeff, m_Abuffer, m_Bbuffer, opA, opB), "Failed to evaluate the required contraction.");
                            CALL_AND_HANDLE(backend_type::complex_conjugate(res.size(), res.buffer(), res.buffer()), "Failed to compute complex conjugate of result");
                        }
                        else
                        {
                            CALL_AND_HANDLE(eval_contraction(res, beta, coeff, m_Abuffer, m_Bbuffer, opA, opB), "Failed to evaluate the required contraction.");
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
                                ASSERT(m_working != nullptr, "The workspace array has not been bound.");
                                ASSERT(m_working_size >= m_Asize, "The workspace array is not large enough to store temporary objects.");
                                CALL_AND_HANDLE(backend_type::complex_conjugate(m_Asize, m_Abuffer, m_working), "Failed to compute complex conjugate array value.");
                                Abuffer = static_cast<value_ptr>(m_working);
                            }
                        }
                        else if (m_opA != backend_type::op_c && m_opB == backend_type::op_c)
                        {
                            opB = backend_type::op_n;
                            if (is_complex<value_type>::value)
                            {
                                ASSERT(m_working != nullptr, "The workspace array has not been bound.");
                                ASSERT(m_working_size >= m_Bsize, "The workspace array is not large enough to store temporary objects.");
                                CALL_AND_HANDLE(backend_type::complex_conjugate(m_Bsize, m_Bbuffer, m_working), "Failed to compute complex conjugate array value.");
                                Bbuffer = static_cast<value_ptr>(m_working);
                            }
                        }
                        CALL_AND_HANDLE(eval_contraction(res, beta, coeff, Abuffer, Bbuffer, opA, opB), "Failed to evaluate the required contraction.");
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate a contraction between a rank 2 and rank 3 tensor.");
                }
            }

            template <typename T3>
            void addition_applicative(T3 &res, value_type coeff_scale = 1.0)
            {
                CALL_AND_HANDLE(applicative(res, 1.0, coeff_scale), "Failed to add the result of tensor_contraction_1_mt to input tensor.");
            }

            template <typename T3>
            void subtraction_applicative(T3 &res, value_type coeff_scale = 1.0)
            {
                CALL_AND_HANDLE(applicative(res, -1.0, coeff_scale), "Failed to subtract the result of tensor_contraction_1_mt to input tensor.");
            }

        protected:
            template <typename T3>
            void eval_contraction(T3 &res, value_type beta, value_type coeff, value_ptr Abuffer, value_ptr Bbuffer, ttype opA, ttype opB)
            {
                try
                {
                    if (m_batched)
                    {
                        size_type strideres = res.size(1) * res.size(2);
                        size_type ldres = res.shape(2);
                        CALL_AND_HANDLE(
                            backend_type::batched_gemm(opB, opA, m_m, m_n, m_k, coeff, Bbuffer, m_ldB, m_strideB, Abuffer, m_ldA, m_strideA, beta, res.buffer(), ldres, strideres, m_t),
                            "Error when making call to batched_gemm.");
                    }
                    else
                    {
                        size_type ldres = m_m;
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
        }; // tensor_contraction_1_MT

    } // namespace expression_templates
    template <typename T1, typename T2>
    struct traits<expression_templates::tensor_contraction_1_mt<T1, T2>>
    {

        using value_type = typename traits<T1>::value_type;
        using backend_type = typename traits<T1>::backend_type;
        using shape_type = std::array<typename backend_type::size_type, 3>;
        using const_shape_reference = const shape_type &;
        static constexpr size_t rank = 3;
    };
} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_CONTRACTION_1_MT_HPP_//
