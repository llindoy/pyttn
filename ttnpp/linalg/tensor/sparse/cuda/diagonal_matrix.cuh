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

#ifndef PYTTN_LINALG_TENSOR_SPARSE_DIAGONAL_MATRIX_CUH_
#define PYTTN_LINALG_TENSOR_SPARSE_DIAGONAL_MATRIX_CUH_

#include "../../../utils/serialisation.cuh"
#include "../../../backends/cuda/cuda_backend.hpp"
#include "../../../utils/memory_helper.cuh"
#include "../diagonal_matrix.hpp"
#include "../../../backends/cuda/host_access.cuh"

namespace linalg
{
    template <typename T>
    class diagonal_matrix<T, cuda_backend> : public diagonal_matrix_base<diagonal_matrix<T, cuda_backend>>, public host_access<T>
    {
    public:
        using self_type = diagonal_matrix<T, cuda_backend>;
        using base_type = diagonal_matrix_base<self_type>;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using const_reference = const T&;
    public:
        template <typename... Args>
        diagonal_matrix(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct diagonal_matrix object.");
        }

        template <typename... Args>
        self_type &operator=(Args &&...args)
        {
            CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));
            return *this;
        }

        void from_host() const
        {
            this->_from_host(base_type::m_vals, base_type::m_nnz);
        }

        inline const_reference at(size_t i) const
        {
            ASSERT(internal::compare_bounds(i, base_type::m_nnz), "Failed to access element of diagonal matrix.  Index out of bounds.");
            if(!this->m_copied_from_host){this->from_host();}
            return this->m_host_buffer[i];
        }

        inline const_reference at(size_t i, size_t j) const
        {
            ASSERT(internal::compare_bounds(i, base_type::m_shape[0]) && internal::compare_bounds(j, base_type::m_shape[1]), "Failed to access element of diagonal matrix.  Index out of bounds.");
            if(!this->m_copied_from_host){this->from_host();}
            return this->m_host_buffer[i];
        }

        inline const_reference operator()(size_t i, size_t /* j */) const
        {
            if(!this->m_copied_from_host){this->from_host();}
            return this->m_host_buffer[i];
        }

        inline const_reference operator[](size_t i) const
        {
            ASSERT(internal::compare_bounds(i, base_type::m_nnz), "Failed to access element of diagonal matrix.  Index out of bounds.");
            if(!this->m_copied_from_host){this->from_host();}
            return this->m_host_buffer[i];
        }

        tensor<T, 2, blas_backend> todense() const
        {
            diagonal_matrix<T, blas_backend> mat(*this);
            return mat.todense();
        }
    }; // diagonal_matrix<T, cuda_backend>

    template <typename T>
    std::ostream &operator<<(std::ostream &out, const diagonal_matrix<T, cuda_backend> &_mat)
    {
        diagonal_matrix<T, blas_backend> mat(_mat);
        out << mat;
        return out;
    }
} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_SPARSE_DIAGONAL_MATRIX_CUH_//
