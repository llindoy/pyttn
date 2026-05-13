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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_EXPRESSION_CUH_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_EXPRESSION_CUH_

#include "../../../../linalg_forward_decl.hpp"
#include "../expression.hpp"
#include "storage_traits.cuh"
#include "../../../../backends/cuda/cuda_backend.hpp"

#include "applicative/addition.cuh"
#include "applicative/scalar_multiplication.cuh"
#include "applicative/complex_conjugation.cuh"
#include "applicative/exponential.cuh"
#include "applicative/hadamard.cuh"
#include "applicative/complex.cuh"

namespace linalg
{
    namespace expression_templates
    {
        template <typename T>
        class literal_type<T, cuda_backend>
        {
            static_assert(is_number<T>::value, "Failed to initialise literal type object.  The literal must be a number type.");

        public:
            using value_type = typename device_type<T, cuda_backend>::type;
            using backend_type = cuda_backend;
            literal_type(T val) : m_value(val) {}

            inline __host__ __device__ operator T() const { return T(m_value); }

            template <typename... Args>
            inline __host__ __device__ value_type operator()(Args &&...args) { return m_value; }

        private:
            value_type m_value;
        };

        template <typename T>
        class literal_type<std::complex<T>, cuda_backend>
        {
            static_assert(is_number<T>::value, "Failed to initialise literal type object.  The literal must be a number type.");

        public:
            using value_type = cuda::std::complex<T>;
            using backend_type = cuda_backend;
            literal_type(std::complex<T> val) : m_value(val) {}

            inline __host__ __device__ operator std::complex<T>() const { return std::complex<T>(m_value); }
            inline __host__ __device__ operator cuda::std::complex<T>() const { return m_value; }

            template <typename... Args>
            inline __host__ __device__ value_type operator()(Args &&...args) { return m_value; }

        private:
            value_type m_value;
        };

        /////////////////////////////////////////////////////////////////////////////////
        //   Helper objects for specialising the applicative functions of the binary   //
        //                             expression objects.                             //
        /////////////////////////////////////////////////////////////////////////////////
        namespace internal
        {
            // generic cuda wrapper
            template <typename type, typename expr>
            struct expression_applicative<type, expr, cuda_backend>
            {
                using size_type = typename traits<cuda_backend>::size_type;
                template <typename res>
                static inline void apply(res &_res, const expr &_expr)
                {
                    static_assert(std::is_base_of<type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
                    CALL_AND_HANDLE(backend_algebra<cuda_backend>::evaluate_expression_tree(_res.buffer(), _res.nelems(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
                }
                template <typename res>
                static inline void addition_assign(res &_res, const expr &_expr)
                {
                    static_assert(std::is_base_of<type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
                    CALL_AND_HANDLE(backend_algebra<cuda_backend>::evaluate_add_expression_tree(_res.buffer(), _res.nelems(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
                }
                template <typename res>
                static inline void subtraction_assign(res &_res, const expr &_expr)
                {
                    static_assert(std::is_base_of<type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
                    CALL_AND_HANDLE(backend_algebra<cuda_backend>::evaluate_sub_expression_tree(_res.buffer(), _res.nelems(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
                }
            };

            // blas wrapper for diagonal matrix return type
            template <typename expr>
            struct expression_applicative<diagonal_matrix_type, expr, cuda_backend>
            {
                using size_type = typename traits<cuda_backend>::size_type;
                template <typename res>
                static inline void apply(res &_res, const expr &_expr)
                {
                    static_assert(std::is_base_of<diagonal_matrix_type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
                    CALL_AND_HANDLE(backend_algebra<cuda_backend>::evaluate_expression_tree_strided(_res.buffer(), _res.nelems(), _res.incx(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
                }
                template <typename res>
                static inline void addition_assign(res &_res, const expr &_expr)
                {
                    static_assert(std::is_base_of<diagonal_matrix_type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
                    CALL_AND_HANDLE(backend_algebra<cuda_backend>::evaluate_add_expression_tree_strided(_res.buffer(), _res.nelems(), _res.incx(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
                }
                template <typename res>
                static inline void subtraction_assign(res &_res, const expr &_expr)
                {
                    static_assert(std::is_base_of<diagonal_matrix_type, res>::value, "Failed to instantiate expression applicative object.  The input type is not derived from the result type.");
                    CALL_AND_HANDLE(backend_algebra<cuda_backend>::evaluate_sub_expression_tree_strided(_res.buffer(), _res.nelems(), _res.incx(), _expr), "Failed to evaluate expression object.  The cuda backend evaluate expression tree raised an exception.");
                }
            };
        } // namespace internal

        /////////////////////////////////////////////////////////////////////////////////
        //               Unary expression type wrapper for cuda backend                //
        /////////////////////////////////////////////////////////////////////////////////
        template <typename vtype, template <typename> class operation>
        class unary_expression<vtype, operation, cuda_backend>
        {
        private:
            using vtraits = storage_traits<vtype>;
            typename vtraits::type m_val;
            typename vtraits::eval_type m_eval;

        public:
            using self_type = unary_expression<vtype, operation, cuda_backend>;
            using backend_type = cuda_backend;
            using size_type = typename traits<backend_type>::size_type;
            using op_type = operation<backend_type>;
            using value_type = typename device_type<typename result_type<self_type>::value_type, backend_type>::type;
            using result_type = typename result_type<self_type>::type;
            using eval_type = internal::expression_applicative<result_type, self_type, backend_type>;

            unary_expression(typename vtraits::type v) : m_val(v), m_eval(vtraits::data(m_val)) {}
            auto obj() const -> decltype(m_val) { return m_val; }

            template <typename array_type>
            void operator()(array_type &res) const { CALL_AND_HANDLE(eval_type::apply(res, *this), "Failed to evaluate unary expression."); }
            template <typename array_type>
            void addition_assign(array_type &res) const { CALL_AND_HANDLE(eval_type::addition_assign(res, *this), "Failed to evaluate unary expression."); }
            template <typename array_type>
            void subtraction_assign(array_type &res) const { CALL_AND_HANDLE(eval_type::subtraction_assign(res, *this), "Failed to evaluate unary expression."); }
            inline __device__ value_type operator[](size_type i) const { return op_type::apply(m_eval, i); }
        }; // class unary_expression

        /////////////////////////////////////////////////////////////////////////////////
        //              Binary expression type wrapper for cuda backend                //
        /////////////////////////////////////////////////////////////////////////////////
        template <typename ltype, typename rtype, template <typename> class operation>
        class binary_expression<ltype, rtype, operation, cuda_backend>
        {
        private:
            using ltraits = storage_traits<ltype>;
            using rtraits = storage_traits<rtype>;
            typename ltraits::type m_lstore;
            typename rtraits::type m_rstore;
            typename ltraits::eval_type m_left;
            typename rtraits::eval_type m_right;

        public:
            using self_type = binary_expression<ltype, rtype, operation, cuda_backend>;
            using backend_type = cuda_backend;
            using size_type = typename traits<backend_type>::size_type;
            using op_type = operation<backend_type>;
            using value_type = typename device_type<typename result_type<self_type>::value_type, backend_type>::type;
            using result_type = typename result_type<self_type>::type;
            using eval_type = internal::expression_applicative<result_type, self_type, backend_type>;

            binary_expression(typename ltraits::type l, typename rtraits::type r) : m_lstore(l), m_rstore(r), m_left(ltraits::data(l)), m_right(rtraits::data(r)) {}

            auto left() const -> decltype(m_lstore) { return m_lstore; }
            auto right() const -> decltype(m_rstore) { return m_rstore; }

            template <typename array_type>
            void operator()(array_type &res) const { CALL_AND_HANDLE(eval_type::apply(res, *this), "Failed to evaluate binary expression."); }
            template <typename array_type>
            void addition_assign(array_type &res) const { CALL_AND_HANDLE(eval_type::addition_assign(res, *this), "Failed to evaluate binary expression."); }
            template <typename array_type>
            void subtraction_assign(array_type &res) const { CALL_AND_HANDLE(eval_type::subtraction_assign(res, *this), "Failed to evaluate binary expression."); }
            inline __device__ value_type operator[](size_type i) const { return op_type::apply(m_left, m_right, i); }
        }; // class binary_expression


} // namespace linalg
}
#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_EXPRESSION_CUH_//
