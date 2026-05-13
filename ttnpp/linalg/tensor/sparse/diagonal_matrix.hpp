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

#ifndef PYTTN_LINALG_TENSOR_SPARSE_DIAGONAL_MATRIX_BASE_HPP_
#define PYTTN_LINALG_TENSOR_SPARSE_DIAGONAL_MATRIX_BASE_HPP_

#include "special_matrix_base.hpp"

namespace linalg
{
    template <typename impl>
    class diagonal_matrix_base : public special_matrix_base<diagonal_matrix_base<impl>>
    {
    public:
        static constexpr size_t rank = 2;
        using self_type = diagonal_matrix_base<impl>;
        using base_type = special_matrix_base<self_type>;
        using size_type = typename base_type::size_type;
        using shape_type = typename base_type::shape_type;
        using value_type = typename base_type::value_type;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        friend base_type;

    protected:
        using base_type::m_shape;
        using base_type::m_vals;

    public:
        template <typename... Args>
        diagonal_matrix_base(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct diagonal matrix object.");
        }
        template <typename... Args>
        self_type &operator=(Args &&...args)
        {
            CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));
            return *this;
        }

        constexpr size_type incx() const { return 1; }
        constexpr size_type diagonal_stride() const { return 1; }

    protected:
        static size_type nnz_from_shape(const shape_type &shape) { return shape[0] < shape[1] ? shape[0] : shape[1]; }
        static size_type nrows_from_nnz(const size_type &nnz) { return nnz; }

    public:
        inline value_type *D() { return base_type::m_vals; }
        inline const value_type *D() const { return base_type::m_vals; }

    }; // class diagonal_matrix_base

    
    template <typename T, typename backend>
    class diagonal_matrix : public diagonal_matrix_base<diagonal_matrix<T, backend>>
    {
    public:
        using size_type = typename traits<backend>::size_type;

        using self_type = diagonal_matrix<T, backend>;
        using base_type = diagonal_matrix_base<self_type>;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using const_reference = const T&;
        using reference = T&;
    public:
        template <typename... Args>
        diagonal_matrix(Args &&...args);

        template <typename... Args>
        self_type &operator=(Args &&...args);

        void from_host() const;

        reference operator[](size_type i);
        const_reference operator[](size_type i) const ;
        reference at(size_type i);
        const_reference at(size_type i) const;
        reference operator()(size_type i, size_type /* j */) ;
        const_reference operator()(size_type i, size_type /* j */) const;
        reference at(size_type i, size_type j);
        const_reference at(size_type i, size_type j) const;
        inline tensor<T, 2, blas_backend> todense() const;
    }; // diagonal_matrix

    template <typename T, typename backend>
    std::ostream &operator<<(std::ostream &out, const diagonal_matrix<T, backend> &mat);

} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_SPARSE_DIAGONAL_MATRIX_BASE_HPP_//
