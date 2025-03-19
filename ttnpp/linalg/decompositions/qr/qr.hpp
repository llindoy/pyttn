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

#ifndef PYTTN_LINALG_DECOMPOSITIONS_QR_HPP_
#define PYTTN_LINALG_DECOMPOSITIONS_QR_HPP_

#include "../../utils/exception_handling.hpp"
#include "qr_blas.hpp"

namespace linalg
{
    template <typename matrix_type>
    class qr<matrix_type, typename std::enable_if<
                              is_dense_matrix<matrix_type>::value, void>::type>
        : public internal::dense_matrix_qr<matrix_type>
    {
    public:
        using base_type = internal::dense_matrix_qr<matrix_type>;
        template <typename... Args>
        qr(Args &&...args)
        try : base_type(std::forward<Args>(args)...)
        {
        }
        catch (...)
        {
            throw;
        }
    };

    template <typename matrix_type>
    class lq<matrix_type, typename std::enable_if<
                              is_dense_matrix<matrix_type>::value, void>::type>
        : public internal::dense_matrix_lq<matrix_type>
    {
    public:
        using base_type = internal::dense_matrix_lq<matrix_type>;
        template <typename... Args>
        lq(Args &&...args)
        try : base_type(std::forward<Args>(args)...)
        {
        }
        catch (...)
        {
            throw;
        }
    };
} // namespace linalg

#endif // LINALG_DECOMPOSITIONS_QR_HPP
