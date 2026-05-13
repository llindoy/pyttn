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

#ifndef PYTTN_LINALG_TENSOR_SPARSE_DIAGONAL_MATRIX_HPP_
#define PYTTN_LINALG_TENSOR_SPARSE_DIAGONAL_MATRIX_HPP_

#include "../../../backends/blas/blas_backend.hpp"
#include "../diagonal_matrix.hpp"

namespace linalg
{


    template <typename T>
    class diagonal_matrix<T, blas_backend> : public diagonal_matrix_base<diagonal_matrix<T, blas_backend>>
    {
    public:
        using self_type = diagonal_matrix<T, blas_backend>;
        using base_type = diagonal_matrix_base<self_type>;
        using size_type = typename traits<blas_backend>::size_type;

        template <typename U>
        friend std::ostream &operator<<(std::ostream &out, const diagonal_matrix<U, blas_backend> &mat);
        template <typename U>
        friend std::istream &operator>>(std::istream &out, diagonal_matrix<U, blas_backend> &mat);

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

        void from_host() const{}
        void clear_host() const{}

        T &operator[](size_type i) { return base_type::m_vals[i]; }
        const T &operator[](size_type i) const { return base_type::m_vals[i]; }
        T &at(size_type i)
        {
            ASSERT(internal::compare_bounds(i, base_type::m_nnz), "Failed to access element of diagonal matrix.  Index out of bounds.");
            return base_type::m_vals[i];
        }
        const T &at(size_type i) const
        {
            ASSERT(internal::compare_bounds(i, base_type::m_nnz), "Failed to access element of diagonal matrix.  Index out of bounds.");
            return base_type::m_vals[i];
        }

        T &operator()(size_type i, size_type /* j */) { return base_type::m_vals[i]; }
        const T &operator()(size_type i, size_type /* j */) const { return base_type::m_vals[i]; }

        T &at(size_type i, size_type j)
        {
            ASSERT(internal::compare_bounds(i, base_type::m_shape[0]) && internal::compare_bounds(j, base_type::m_shape[1]), "Failed to access element of diagonal matrix.  Index out of bounds.");
            ASSERT(i == j, "Failed to access element of diagonal matrix.  Requested element is not on the diagonal.");
            return base_type::m_vals[i];
        }

        const T &at(size_type i, size_type j) const
        {
            ASSERT(internal::compare_bounds(i, base_type::m_shape[0]) && internal::compare_bounds(j, base_type::m_shape[1]), "Failed to access element of diagonal matrix.  Index out of bounds.");
            ASSERT(i == j, "Failed to access element of diagonal matrix.  Requested element is not on the diagonal.");
            return base_type::m_vals[i];
        }

        tensor<T, 2, blas_backend> todense() const
        {
            tensor<T, 2, blas_backend> mat(base_type::m_shape[0], base_type::m_shape[1]);
            for (size_t i = 0; i < base_type::m_nnz; ++i)
            {
                mat(i, i) = base_type::m_vals[i];
            }
            return mat;
        }
    }; // diagonal_matrix<T, blas_backend>



    template <typename T>
    std::ostream &operator<<(std::ostream &out, const diagonal_matrix<T, blas_backend> &mat)
    {
        using size_type = typename diagonal_matrix<T, blas_backend>::size_type;
        out << "diagonal: " << mat.m_shape[0] << " " << mat.m_shape[1] << std::endl;
        for (size_type i = 0; i < mat.nnz(); ++i)
        {
            out << i << " " << i << " " << mat.m_vals[i] << std::endl;
        }
        return out;
    }
} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_SPARSE_DIAGONAL_MATRIX_HPP_//
