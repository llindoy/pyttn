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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_MATRIX_MULTIPLICATION_BASE_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_MATRIX_MULTIPLICATION_BASE_HPP_

#include "../../../linalg_forward_decl.hpp"

namespace linalg
{
    namespace expression_templates
    {

        template <typename impl>
        class matrix_matrix_product_base : public tensor_contraction_expression, public expression_base<matrix_matrix_product_base<impl>, true>, public dense_type
        {
        public:
            using base_type = expression_base<matrix_matrix_product_base<impl>, true>;

            using value_type = typename std::remove_cv<typename traits<impl>::value_type>::type;
            using lvalue_type = typename traits<impl>::lvalue_type;
            using rvalue_type = typename traits<impl>::rvalue_type;
            using backend_type = typename traits<impl>::backend_type;
            using size_type = typename backend_type::size_type;
            using shape_type = std::array<size_type, 2>;
            using ttype = typename backend_type::transform_type;

        protected:
            value_type m_coeff;

            // working buffer
            value_type *m_working;
            size_type m_working_size;

            ttype m_opA, m_opB;
            size_type m_m, m_n, m_k;

        protected:
            void initialise_state(const shape_type &As, const shape_type &Bs, bool transA, bool transB, bool conjA, bool conjB)
            {
                size_type ka;
                if (transA)
                {
                    m_opA = conjA ? backend_type::op_h : backend_type::op_t;
                    m_n = As[1];
                    ka = As[0];
                }
                else
                {
                    m_opA = conjA ? backend_type::op_c : backend_type::op_n;
                    m_n = As[0];
                    ka = As[1];
                }

                size_type kb;
                if (transB)
                {
                    m_opB = conjB ? backend_type::op_h : backend_type::op_t;
                    m_m = Bs[0];
                    kb = Bs[1];
                }
                else
                {
                    m_opB = conjB ? backend_type::op_c : backend_type::op_n;
                    m_m = Bs[1];
                    kb = Bs[0];
                }
                ASSERT(ka == kb, "Failed to compute matrix-matrix contraction.  The specified matrices do not have the correct shape to perform the requested contraction (incompatible contraction index size).");
                m_k = ka;
                this->m_shape[0] = m_n;
                this->m_shape[1] = m_m;
            }

        public:
            matrix_matrix_product_base(const value_type &coeff, const shape_type &As, const shape_type &Bs, bool transA = false, bool transB = false, bool conjA = false, bool conjB = false)
                : base_type(shape_type{{0, 0}}), m_coeff(coeff), m_working(nullptr), m_working_size(0)
            {
                // we start by determining all of the required parameters for the gemm call (and determine if the requested matrix-matrix product actually maps onto a gemm call.
                CALL_AND_HANDLE(initialise_state(As, Bs, transA, transB, conjA, conjB), "Failed to construct matrix_matrix_product object.  Failed to correctly initialise the matrix state.");
            }

            template <typename wbuff>
            matrix_matrix_product_base(const value_type &coeff, const shape_type &As, const shape_type &Bs, wbuff &working, bool transA = false, bool transB = false, bool conjA = false, bool conjB = false)
                : base_type(shape_type{{0, 0}}), m_coeff(coeff), m_working(nullptr), m_working_size(0)
            {
                // we start by determining all of the required parameters for the gemm call (and determine if the requested matrix-matrix product actually maps onto a gemm call.
                CALL_AND_HANDLE(initialise_state(As, Bs, transA, transB, conjA, conjB), "Failed to construct matrix_matrix_product object.  Failed to correctly initialise the matrix state.");
                CALL_AND_HANDLE(bind_conjugate_working(working), "Failed to bind working buffer.");
            }

            template <typename V, typename = typename std::enable_if<std::is_convertible<V, value_type>::value, void>::type>
            matrix_matrix_product_base(const matrix_matrix_product_base &o, const V &factor)
                : base_type(o.shape()), m_coeff(o.m_coeff * factor), m_working(o.m_working), m_working_size(o.m_working_size), m_opA(o.m_opA), m_opB(o.m_opB), m_m(o.m_m), m_n(o.m_n), m_k(o.m_k) {}

            matrix_matrix_product_base(const matrix_matrix_product_base &o, bool make_transpose = false, bool conjugate = false) : base_type(shape_type{{0, 0}}), m_working(o.m_working), m_working_size(o.m_working_size)
            {
                // using std::conj;
                if (make_transpose)
                {
                    m_coeff = conjugate ? conj(o.m_coeff) : o.m_coeff;
                    this->m_shape[0] = o.m_shape[1];
                    this->m_shape[1] = o.m_shape[0];
                    m_m = o.m_n;
                    m_n = o.m_m;
                    m_k = o.m_k;

                    switch (o.m_opA)
                    {
                    case (backend_type::op_n):
                    {
                        m_opB = conjugate ? backend_type::op_h : backend_type::op_t;
                        break;
                    };
                    case (backend_type::op_t):
                    {
                        m_opB = conjugate ? backend_type::op_c : backend_type::op_n;
                        break;
                    };
                    case (backend_type::op_c):
                    {
                        m_opB = conjugate ? backend_type::op_t : backend_type::op_h;
                        break;
                    };
                    case (backend_type::op_h):
                    {
                        m_opB = conjugate ? backend_type::op_n : backend_type::op_c;
                        break;
                    };
                    };
                    switch (o.m_opB)
                    {
                    case (backend_type::op_n):
                    {
                        m_opA = conjugate ? backend_type::op_h : backend_type::op_t;
                        break;
                    };
                    case (backend_type::op_t):
                    {
                        m_opA = conjugate ? backend_type::op_c : backend_type::op_n;
                        break;
                    };
                    case (backend_type::op_c):
                    {
                        m_opA = conjugate ? backend_type::op_t : backend_type::op_h;
                        break;
                    };
                    case (backend_type::op_h):
                    {
                        m_opA = conjugate ? backend_type::op_n : backend_type::op_c;
                        break;
                    };
                    }
                }
                else
                {
                    m_coeff = conjugate ? conj(o.m_coeff) : o.m_coeff;
                    this->m_shape = o.m_shape;
                    m_m = o.m_m;
                    m_n = o.m_n;
                    m_k = o.m_k;

                    switch (o.m_opA)
                    {
                    case (backend_type::op_n):
                    {
                        m_opA = conjugate ? backend_type::op_c : backend_type::op_n;
                        break;
                    };
                    case (backend_type::op_t):
                    {
                        m_opA = conjugate ? backend_type::op_h : backend_type::op_t;
                        break;
                    };
                    case (backend_type::op_c):
                    {
                        m_opA = conjugate ? backend_type::op_n : backend_type::op_c;
                        break;
                    };
                    case (backend_type::op_h):
                    {
                        m_opA = conjugate ? backend_type::op_t : backend_type::op_h;
                        break;
                    };
                    };
                    switch (o.m_opB)
                    {
                    case (backend_type::op_n):
                    {
                        m_opB = conjugate ? backend_type::op_c : backend_type::op_n;
                        break;
                    };
                    case (backend_type::op_t):
                    {
                        m_opB = conjugate ? backend_type::op_h : backend_type::op_t;
                        break;
                    };
                    case (backend_type::op_c):
                    {
                        m_opB = conjugate ? backend_type::op_n : backend_type::op_c;
                        break;
                    };
                    case (backend_type::op_h):
                    {
                        m_opB = conjugate ? backend_type::op_t : backend_type::op_h;
                        break;
                    };
                    };
                }
            }

            template <typename conj_type>
            impl &bind_conjugate_workspace(conj_type &conj)
            {
                static_assert(is_dense<conj_type>(), "Failed to construct contraction_332 object.  The input conjorary object type is not a dense tensor.");
                static_assert(std::is_same<value_type, typename conj_type::value_type>::value, "Failed to construct contraction_332 object.  The conj_type tensor must have the same value_type as the input tensors.");
                static_assert(std::is_same<backend_type, typename conj_type::backend_type>::value, "Failed to construct contraction_332 object.  The conj_type tensor must have the same backend_type as the input tensors.");
                try
                {
                    ASSERT(m_working == nullptr, "A working buffer has already been bound.");
                    m_working = conj.buffer();
                    m_working_size = conj.size();
                    return *(static_cast<impl *>(this));
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
                static_assert(std::is_same<typename traits<T3>::value_type, value_type>::value, "Failed to construct evaluate operator for matrix_matrix_product object.  The result_type must have a compatible value_type.");
                static_assert(std::is_same<typename traits<T3>::backend_type, backend_type>::value, "Failed to construct evaluate operator for matrix_matrix_product object.  The result_type must have a compatible backend type.");
                static_assert(is_dense_matrix<T3>::value, "Failed to construct evaluate operator for matrix_matrix_product object.  The result type must be a dense matrix.");
                CALL_AND_RETHROW(static_cast<impl *>(this)->applicative_impl(res, beta, coeff_scale));
            }

            template <typename T3>
            void addition_applicative(T3 &res, value_type coeff_scale = 1.0)
            {
                static_assert(std::is_same<typename traits<T3>::value_type, value_type>::value, "Failed to construct evaluate operator for matrix_matrix_product object.  The result_type must have a compatible value_type.");
                static_assert(std::is_same<typename traits<T3>::backend_type, backend_type>::value, "Failed to construct evaluate operator for matrix_matrix_product object.  The result_type must have a compatible backend type.");
                static_assert(is_dense_matrix<T3>::value, "Failed to construct evaluate operator for matrix_matrix_product object.  The result type must be a dense matrix.");
                CALL_AND_RETHROW(static_cast<impl *>(this)->applicative_impl(res, 1.0, coeff_scale));
            }

            template <typename T3>
            void subtraction_applicative(T3 &res, value_type coeff_scale = 1.0)
            {
                static_assert(std::is_same<typename traits<T3>::value_type, value_type>::value, "Failed to construct evaluate operator for matrix_matrix_product object.  The result_type must have a compatible value_type.");
                static_assert(std::is_same<typename traits<T3>::backend_type, backend_type>::value, "Failed to construct evaluate operator for matrix_matrix_product object.  The result_type must have a compatible backend type.");
                static_assert(is_dense_matrix<T3>::value, "Failed to construct evaluate operator for matrix_matrix_product object.  The result type must be a dense matrix.");
                CALL_AND_RETHROW(static_cast<impl *>(this)->applicative_impl(res, -1.0, coeff_scale));
            }
        }; // matrix_matrix_product_base

    } // namespace expression_templates
    template <typename impl>
    struct traits<expression_templates::matrix_matrix_product_base<impl>>
    {
        using value_type = typename traits<impl>::value_type;
        using lvalue_type = typename traits<impl>::lvalue_type;
        using rvalue_type = typename traits<impl>::rvalue_type;
        using backend_type = typename traits<impl>::backend_type;
        using shape_type = std::array<typename backend_type::size_type, 2>;
        using const_shape_reference = const shape_type &;
        static constexpr size_t rank = 2;
    };

} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_MATRIX_MULTIPLICATION_BASE_HPP_//
