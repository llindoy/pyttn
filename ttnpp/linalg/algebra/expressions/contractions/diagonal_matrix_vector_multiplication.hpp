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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_DIAGONAL_MATRIX_VECTOR_MULTIPLICATION_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_DIAGONAL_MATRIX_VECTOR_MULTIPLICATION_HPP_

#include "matrix_vector_multiplication_base.hpp"

#include <type_traits>

namespace linalg
{

    namespace expression_templates
    {
        // diagonal matrix dense vector product
        template <typename sparse_type, typename dense_type>
        class matrix_vector_product<sparse_type, dense_type,
                                    typename std::enable_if<is_dense_vector<dense_type>::value && is_diagonal_matrix_type<sparse_type>::value && is_same_backend<sparse_type, dense_type>::value && is_sparse_multiply_value<sparse_type, dense_type>::value, void>::type> : public matrix_vector_product_base<matrix_vector_product<sparse_type, dense_type, void>>
        {
        public:
            using backend_type = typename traits<dense_type>::backend_type;
            using size_type = typename backend_type::size_type;
            using left_type = sparse_type;
            using right_type = dense_type;
            using lvalue_type = typename traits<left_type>::value_type;
            using rvalue_type = typename traits<right_type>::value_type;
            using lvalue_ptr = typename std::add_pointer<typename std::add_const<lvalue_type>::type>::type;
            using rvalue_ptr = typename std::add_pointer<typename std::add_const<rvalue_type>::type>::type;

            using value_type = decltype(rvalue_type() * lvalue_type());
            using self_type = matrix_vector_product<left_type, right_type>;
            using base_type = matrix_vector_product_base<self_type>;

            static constexpr size_t rank = 1;

        protected:
            lvalue_ptr m_Abuffer;
            rvalue_ptr m_Xbuffer;

            size_type m_Asize, m_Xsize;
            size_type m_incA, m_incX;

            using base_type::m_coeff;
            using base_type::m_m;
            using base_type::m_n;
            using base_type::m_opA;
            using base_type::m_opX;
            using base_type::m_working;
            using base_type::m_working_size;

        public:
            matrix_vector_product(const value_type &coeff, const left_type &A, const right_type &X, bool transA = false, bool conjA = false, bool conjX = false)
            try : base_type(coeff, A.shape(), X.shape(), transA, conjA, conjX), m_Abuffer(A.buffer()), m_Xbuffer(X.buffer()), m_Asize(A.size()), m_Xsize(X.size()), m_incA(A.incx()), m_incX(X.incx()) {}
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");
            }

            template <typename working_type>
            matrix_vector_product(const value_type &coeff, const left_type &A, const right_type &X, working_type &working_buffer, bool transA = false, bool conjA = false, bool conjX = false)
            try : base_type(coeff, A.shape(), X.shape(), working_buffer, transA, conjA, conjX), m_Abuffer(A.buffer()), m_Xbuffer(X.buffer()), m_Asize(A.nnz()), m_Xsize(X.size()), m_incA(A.incx()), m_incX(X.incx()) {}
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");
            }

            matrix_vector_product(const matrix_vector_product &o, const value_type &factor)
            try : base_type(o, factor), m_Abuffer(o.m_Abuffer), m_Xbuffer(o.m_Xbuffer), m_Asize(o.m_Asize), m_Xsize(o.m_Xsize), m_incA(o.m_incA), m_incX(o.m_incX) {}
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");
            }

            matrix_vector_product(const matrix_vector_product &o, bool conjugate = false)
            try : base_type(o, conjugate), m_Abuffer(o.m_Abuffer), m_Xbuffer(o.m_Xbuffer), m_Asize(o.m_Asize), m_Xsize(o.m_Xsize), m_incA(o.m_incA), m_incX(o.m_incX) {}
            catch (const std::exception &ex)
            {
                std::cerr << ex.what() << std::endl;
                RAISE_EXCEPTION("Failed to construct sparse dense matrix vector product object.");
            }

            // This routine expects dense matrices in row major order.  And as such there is no need to perform the reordering of operations required for the dense dense matrix products.
            template <typename T3>
            void applicative_impl(T3 &res, value_type beta = 0.0, value_type coeff_scale = 1.0)
            {
                try
                {
                    ASSERT(res.buffer() != m_Xbuffer, "Failed to compute diagonal matrix vector product.  The matrix vector product does not support inplace products.");
                    value_type coeff = m_coeff * coeff_scale;
                    size_type incc = res.incx();

                    bool conjA = (m_opA == backend_type::op_c || m_opA == backend_type::op_h);
                    bool conjB = (m_opX == backend_type::op_c || m_opX == backend_type::op_h);
                    size_type mn = (m_n < m_m ? m_n : m_m);
                    size_type mx = (m_n > m_m ? m_n : m_m);
                    CALL_AND_HANDLE(backend_type::dgmv(conjA, conjB, mn, mx, coeff, m_Abuffer, m_incA, m_Xbuffer, m_incX, beta, res.buffer(), incc), "Failed to compute diagonal matrix vector product.  Error when making call to backend::dgmv");
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to evaluate diagonal matrix vector product.");
                }
            }
        }; // matrix_vector_product

    } // namespace expression_templates

} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_CONTRACTIONS_DIAGONAL_MATRIX_VECTOR_MULTIPLICATION_HPP_
