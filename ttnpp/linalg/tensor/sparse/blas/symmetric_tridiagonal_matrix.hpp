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

#ifndef PYTTN_LINALG_TENSOR_SPARSE_SYMMETRIC_TRIDIAGONAL_MATRIX_HPP_
#define PYTTN_LINALG_TENSOR_SPARSE_SYMMETRIC_TRIDIAGONAL_MATRIX_HPP_

#include "../../../backends/blas/blas_backend.hpp"
#include "../symmetric_tridiagonal_matrix.hpp"

namespace linalg
{
    template <typename T>
    class symmetric_tridiagonal_matrix<T, blas_backend> : public symmetric_tridiagonal_matrix_base<symmetric_tridiagonal_matrix<T, blas_backend>>
    {
    public:
        using self_type = symmetric_tridiagonal_matrix<T, blas_backend>;
        using base_type = symmetric_tridiagonal_matrix_base<self_type>;
        using size_type = typename traits<blas_backend>::size_type;

        template <typename U>
        friend std::ostream &operator<<(std::ostream &out, const symmetric_tridiagonal_matrix<U, blas_backend> &mat);
        template <typename U>
        friend std::istream &operator>>(std::istream &in, symmetric_tridiagonal_matrix<U, blas_backend> &mat);

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

        T &operator[](size_type i) { return base_type::m_vals[i]; }
        const T &operator[](size_type i) const { return base_type::m_vals[i]; }
        T &at(size_type i)
        {
            ASSERT(internal::compare_bounds(i, base_type::m_nnz), "Failed to access element of symmetric tridiagonal matrix.  Index out of bounds.");
            return base_type::m_vals[i];
        }
        T at(size_type i) const
        {
            ASSERT(internal::compare_bounds(i, base_type::m_nnz), "Failed to access element of symmetric tridiagonal matrix.  Index out of bounds.");
            return base_type::m_vals[i];
        }

        T &operator()(size_type i, size_type j)
        {
            if (i == j)
            {
                return base_type::m_vals[i];
            }
            else if (i + 1 == j)
            {
                return base_type::m_vals[base_type::m_shape[0] + i];
            }
            else if (i == j + 1)
            {
                return base_type::m_vals[base_type::m_shape[0] + j];
            }
            else
            {
                RAISE_EXCEPTION("Failed to access element of symmetric tridiagonal matrix.  The requested index is not a tridiagonal element.");
            }
        }

        T operator()(size_type i, size_type j) const
        {
            if (i == j)
            {
                return base_type::m_vals[i];
            }
            else if (i + 1 == j)
            {
                return base_type::m_vals[base_type::m_shape[0] + i];
            }
            else if (i == j + 1)
            {
                return conj(base_type::m_vals[base_type::m_shape[0] + j]);
            }
            else
            {
                RAISE_EXCEPTION("Failed to access element of symmetric tridiagonal matrix.  The requested index is not a tridiagonal element.");
            }
        }

        T &at(size_type i, size_type j)
        {
            ASSERT(internal::compare_bounds(i, base_type::m_shape[0]) && internal::compare_bounds(j, base_type::m_shape[1]), "Failed to access element of symmetric tridiagonal matrix.  Index out of bounds.");
            CALL_AND_RETHROW(return this->operator()(i, j));
        }

        T at(size_type i, size_type j) const
        {
            ASSERT(internal::compare_bounds(i, base_type::m_shape[0]) && internal::compare_bounds(j, base_type::m_shape[1]), "Failed to access element of symmetric tridiagonal matrix.  Index out of bounds.");
            CALL_AND_RETHROW(this->operator()(i, j));
        }

    }; // symmetric_tridiagonal_matrix<T, blas_backend>

    template <typename T>
    std::ostream &operator<<(std::ostream &out, const symmetric_tridiagonal_matrix<T, blas_backend> &mat)
    {
        using size_type = typename symmetric_tridiagonal_matrix<T, blas_backend>::size_type;
        out << "symmetric tridiagonal: " << mat.m_shape[0] << " " << mat.m_shape[1] << std::endl;
        for (size_type i = 0; i < mat.nrows(); ++i)
        {
            if (i > 0)
            {
                out << i << " " << i - 1 << " " << mat.m_vals[mat.nrows() + i - 1] << std::endl;
            }
            out << i << " " << i << " " << mat.m_vals[i] << std::endl;
            if (i + 1 < mat.nrows())
            {
                out << i << " " << i + 1 << " " << mat.m_vals[mat.nrows() + i] << std::endl;
            }
        }
        return out;
    }

    // template <typename T, typename be, typename = typename std::enable_if<!is_complex<T>::value, void>::type>
    // using symmetric_tridiagonal_matrix = symmetric_tridiagonal_matrix<T, be>;

} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_SPARSE_SYMMETRIC_TRIDIAGONAL_MATRIX_HPP_//
