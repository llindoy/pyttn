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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_APPLICATIVE_COMPLEX_CONJUGATION_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_APPLICATIVE_COMPLEX_CONJUGATION_HPP_

#include "../../../../linalg_forward_decl.hpp"

namespace linalg
{
    namespace expression_templates
    {

        template <>
        class conjugation_op<blas_backend>
        {
        public:
            using size_type = blas_backend::size_type;

        public:
            template <typename T>
            static inline T apply(const T *a, size_type i) { return conj(a[i]); }

            template <typename T>
            static inline typename T::value_type apply(const T &a, size_type i) { return conj(a[i]); }
        };

#ifdef PYTTN_BUILD_CUDA

        template <>
        class conjugation_op<cuda_backend>
        {
        public:
            using size_type = cuda_backend::size_type;

        public:
            template <typename T>
            static inline __device__ T apply(const T *a, size_type i) { return conj(a[i]); }

            template <typename T>
            static inline __device__ typename T::value_type apply(const T &a, size_type i) { return conj(a[i]); }
        };

#endif

    } // namespace expression_templates
} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_APPLICATIVE_COMPLEX_CONJUGATION_HPP_
