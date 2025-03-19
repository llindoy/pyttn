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

#ifndef PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_HPP_
#define PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_HPP_

#include <common/exception_handling.hpp>
#include "singular_value_decomposition_blas.hpp"
#include "singular_value_decomposition_cuda.hpp"

namespace linalg
{
    // eigensolver object for hermitian matrix type that is not mutable
    template <typename matrix_type, bool use_divide_and_conquer>
    class singular_value_decomposition<matrix_type, use_divide_and_conquer, typename std::enable_if<is_dense_matrix<matrix_type>::value, void>::type> : public internal::dense_matrix_singular_value_decomposition<matrix_type, use_divide_and_conquer>
    {
    public:
        using base_type = internal::dense_matrix_singular_value_decomposition<matrix_type, use_divide_and_conquer>;
        template <typename... Args>
        singular_value_decomposition(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (...)
        {
            throw;
        }
    };
} // namespace linalg

#endif // PYTTN_LINALG_DECOMPOSITIONS_SINGULAR_VALUE_DECOMPOSITION_HPP_
