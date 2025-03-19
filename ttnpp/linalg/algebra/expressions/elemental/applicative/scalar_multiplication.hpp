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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_APPLICATIVE_SCALAR_MULTIPLICATION_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_APPLICATIVE_SCALAR_MULTIPLICATION_HPP_

#include "../../../../linalg_forward_decl.hpp"

namespace linalg
{

    namespace expression_templates
    {

        template <>
        class multiplication_op<blas_backend>
        {
        public:
            using size_type = blas_backend::size_type;

        public:
            template <typename T1, typename T2>
            static inline auto apply(T1 a, const T2 *b, size_type i) -> decltype(a * b[i]) { return a * b[i]; }

            template <typename T, typename ltype, typename rtype, template <typename> class op>
            static inline auto apply(T a, binary_expression<ltype, rtype, op, blas_backend> b, size_type i) -> decltype(a * b[i]) { return a * b[i]; }

            template <typename T, typename vtype, template <typename> class op>
            static inline auto apply(T a, unary_expression<vtype, op, blas_backend> b, size_type i) -> decltype(a * b[i]) { return a * b[i]; }
        };

#ifdef PYTTN_BUILD_CUDA
        template <>
        class multiplication_op<cuda_backend>
        {
        public:
            using size_type = cuda_backend::size_type;

        public:
            template <typename T1, typename T2>
            static inline __device__ auto apply(T1 a, const T2 *b, size_type i) -> decltype(a * b[i]) { return a * b[i]; }

            template <typename T, typename ltype, typename rtype, template <typename> class op>
            static inline __device__ auto apply(T a, binary_expression<ltype, rtype, op, cuda_backend> b, size_type i) -> decltype(a * b[i]) { return a * b[i]; }

            template <typename T, typename vtype, template <typename> class op>
            static inline __device__ auto apply(T a, unary_expression<vtype, op, cuda_backend> b, size_type i) -> decltype(a * b[i]) { return a * b[i]; }
        };
#endif

    } // namespace expression_templates
} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_APPLICATIVE_SCALAR_MULTIPLICATION_HPP_