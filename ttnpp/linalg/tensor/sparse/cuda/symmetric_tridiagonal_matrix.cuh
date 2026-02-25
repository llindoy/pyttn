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

#ifndef PYTTN_LINALG_TENSOR_SPARSE_SYMMETRIC_TRIDIAGONAL_MATRIX_CUH_
#define PYTTN_LINALG_TENSOR_SPARSE_SYMMETRIC_TRIDIAGONAL_MATRIX_CUH_

#include "../../../backends/cuda/cuda_backend.hpp"
#include "../../../linalg_forward_decl.hpp"
#include "../symmetric_tridiagonal_matrix.hpp"
#include "../../../backends/cuda/host_access.cuh"

namespace linalg
{
    //add host_access support.
    template <typename T>
    class symmetric_tridiagonal_matrix<T, cuda_backend> : public symmetric_tridiagonal_matrix_base<symmetric_tridiagonal_matrix<T, cuda_backend>>
    {
    public:
        using self_type = symmetric_tridiagonal_matrix<T, cuda_backend>;
        using base_type = symmetric_tridiagonal_matrix_base<self_type>;

        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;

    public:
        template <typename... Args>
        symmetric_tridiagonal_matrix(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct symmetric tridiagonal matrix object.");
        }
        template <typename... Args>
        self_type &operator=(Args &&...args)
        {
            CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));
            return *this;
        }

        __host__ __device__ pointer buffer() { return base_type::m_vals; }
        __host__ __device__ const_pointer buffer() const { return base_type::m_vals; }
        __host__ __device__ pointer data() { return base_type::m_vals; }
        __host__ __device__ const_pointer data() const { return base_type::m_vals; }
    }; // symmetric_tridiagonal_matrix<T, cuda_backend>

} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_SPARSE_SYMMETRIC_TRIDIAGONAL_MATRIX_CUH_//
