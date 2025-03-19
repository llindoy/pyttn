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

#include "special_matrix_base.hpp"

namespace linalg
{

    template <typename impl>
    class symmetric_tridiagonal_matrix_base : public special_matrix_base<symmetric_tridiagonal_matrix_base<impl>>
    {
    public:
        static constexpr size_t rank = 2;
        using self_type = symmetric_tridiagonal_matrix_base<impl>;
        using base_type = special_matrix_base<self_type>;
        using size_type = typename base_type::size_type;
        using shape_type = typename base_type::shape_type;
        using value_type = typename base_type::value_type;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using backend_type = typename base_type::backend_type;
        friend base_type;

    protected:
        using base_type::m_capacity;
        using base_type::m_nnz;
        using base_type::m_shape;
        using base_type::m_vals;

        using allocator = typename base_type::allocator;
        using memfill = typename base_type::memfill;
        template <typename srcbck>
        using memtransfer = memory::transfer<srcbck, backend_type>;

    public:
        template <typename... Args>
        symmetric_tridiagonal_matrix_base(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            std::cerr << ex.what() << std::endl;
            RAISE_EXCEPTION("Failed to construct symmetric_tridiagonal matrix object.");
        }
        template <typename Args>
        self_type &operator=(Args &&args)
        {
            CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)));
            return *this;
        }

        template <typename srcbck>
        symmetric_tridiagonal_matrix_base(const diagonal_matrix<value_type, srcbck> &diag)
        {
            ASSERT(diag.shape(0) == diag.shape(1), "Cannot construct a symmetric tridiagonal matrix from a rectangular diagonal matrix.");
            m_shape = diag.shape();
            m_nnz = 2 * m_shape[0] - 1;
            m_capacity = 2 * m_shape[0] - 1;
            CALL_AND_HANDLE(m_vals = allocator::allocate(m_capacity), "Failed to construct symmetric tridiagonal matrix object.  Value buffer allocation failed.")
            CALL_AND_HANDLE(memtransfer<srcbck>(diag.buffer(), m_shape[0], m_vals), "Failed to construct symmetric tridiagonal matrix object.  Failed to copy diagonal elements from diagonal matrix.");
            CALL_AND_HANDLE(memfill(m_vals + m_shape[0], m_shape[0] - 1, value_type(0.0)), "Failed to construct symmetric tridiagonal matrix object.  Failed to set the off diagonal elements to zero.");
        }

    protected:
        static size_type nnz_from_shape(const shape_type &shape)
        {
            ASSERT(shape[0] == shape[1], "Failed to determine number of non-zero elements in symmetric tridiagonal matrix.  The matrix must be square but is not.");
            return 2 * shape[0] - 1;
        }
        static size_type nrows_from_nnz(const size_type &nnz)
        {
            ASSERT(nnz % 2 == 1, "Failed to determine the number of rows given the number of non-zero elements in the symmetric tridiagonal matrix.  The number of non-zeros must be odd.");
            return (nnz + 1) / 2;
        }

    public:
        inline value_type *D() { return base_type::m_vals; }
        inline const value_type *D() const { return base_type::m_vals; }
        inline value_type *E() { return base_type::m_vals + m_shape[0]; }
        inline const value_type *E() const { return base_type::m_vals + m_shape[0]; }

    }; // class symmetric_tridiagonal_matrix_base

    template <typename T>
    class symmetric_tridiagonal_matrix<T, blas_backend> : public symmetric_tridiagonal_matrix_base<symmetric_tridiagonal_matrix<T, blas_backend>>
    {
    public:
        using self_type = symmetric_tridiagonal_matrix<T, blas_backend>;
        using base_type = symmetric_tridiagonal_matrix_base<self_type>;
        using size_type = typename blas_backend::size_type;

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
            std::cerr << ex.what() << std::endl;
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
                return linalg::conj(base_type::m_vals[base_type::m_shape[0] + j]);
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

#ifdef PYTTN_BUILD_CUDA
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
            std::cerr << ex.what() << std::endl;
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
#endif

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
