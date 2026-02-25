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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_STORAGE_TRAITS_HOST_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_STORAGE_TRAITS_HOST_HPP_

#include "../storage_traits.hpp"
#include "../../../../backends/blas/blas_backend.hpp"


namespace linalg
{

    namespace expression_templates
    {
        namespace internal
        {
            template <typename T>
            struct diagonal_matrix_view_storage_type<T, blas_backend>
            {
                using size_type = typename traits<blas_backend>::size_type;
                T *buffer;
                size_type incx;
                T operator[](size_type i) const { return buffer[i * incx]; }
            };
        } // namespace intenral

    } // namespace expression_templates
} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_STORAGE_TRAITS_HPP_//
