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

#ifndef PYTTN_LINALG_TENSOR_SPARSE_SPECIAL_MATRIX_BASE_HPP_
#define PYTTN_LINALG_TENSOR_SPARSE_SPECIAL_MATRIX_BASE_HPP_

#include <vector>
#include "../../linalg_forward_decl.hpp"

namespace linalg
{

    namespace internal
    {
        template <typename T>
        struct special_matrix_type_tags;
        template <typename T>
        struct special_matrix_type_tags<diagonal_matrix_base<T>>
        {
            using type = diagonal_matrix_type;
        };
        template <typename T>
        struct special_matrix_type_tags<symmetric_tridiagonal_matrix_base<T>>
        {
            using type = symmetric_tridiagonal_matrix_type;
        };
    } // namespace internal

    template <typename impl>
    class special_matrix_base : public internal::special_matrix_type_tags<impl>::type
    {
    public:
        using traits_type = traits<impl>;
        using backend_type = typename traits_type::backend_type;
        using value_type = typename traits_type::value_type;
        using size_type = typename backend_type::size_type;

        static constexpr size_type rank = traits_type::rank;

        using pointer = typename std::add_pointer<value_type>::type;
        using const_pointer = typename std::add_pointer<typename std::add_const<value_type>::type>::type;
        using reference = typename std::add_lvalue_reference<value_type>::type;
        using const_reference = typename std::add_lvalue_reference<typename std::add_const<value_type>::type>::type;

        using shape_type = std::array<size_type, rank>;
        using self_type = special_matrix_base<impl>;

        // the classes used for memory allocation and transfer
        using allocator = memory::allocator<value_type, backend_type>;
        using memfill = memory::filler<value_type, backend_type>;
        template <typename srcbck>
        using memtransfer = memory::transfer<srcbck, backend_type>;

    protected:
        pointer m_vals;       ///< The 1-dimensional array used to store the values present in the matrix
        size_type m_nnz;      ///< The total number of non-zero elements in the special matrix
        size_type m_capacity; ///< The maximum number of elements that can be in the special matrix
        shape_type m_shape;   ///< A static array of size 2 that stores the shape of the special_matrix

    public:
        // Default constructor
        special_matrix_base() : m_vals{nullptr}, m_nnz(0), m_capacity{0}, m_shape{{0, 0}} {}
        special_matrix_base(size_type _nrows) : m_vals{nullptr}, m_nnz(0), m_capacity(0), m_shape{{_nrows, _nrows}}
        {
            CALL_AND_HANDLE(m_nnz = impl::nnz_from_shape(m_shape), "Failed to construct special matrix object.  Unable to determine the number of non-zero elements from the specified shape.");
            m_capacity = m_nnz;
            CALL_AND_HANDLE(m_vals = allocator::allocate(m_capacity), "Failed to construct special matrix object.  Value buffer allocation failed.");
        }

        special_matrix_base(size_type _nrows, size_type _ncols) : m_vals{nullptr}, m_nnz(0), m_capacity(0), m_shape{{_nrows, _ncols}}
        {
            CALL_AND_HANDLE(m_nnz = impl::nnz_from_shape(m_shape), "Failed to construct special matrix object.  Unable to determine the number of non-zero elements from the specified shape.");
            m_capacity = m_nnz;
            CALL_AND_HANDLE(m_vals = allocator::allocate(m_capacity), "Failed to construct special matrix object.  Value buffer allocation failed.");
        }

        template <typename Container, typename = other_copy_constructable_type<Container, self_type>>
        special_matrix_base(const Container &src) : special_matrix_base() { CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy construct special matrix object."); }
        special_matrix_base(const self_type &src) : special_matrix_base() { CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy construct special matrix object."); }

        special_matrix_base(self_type &&src) noexcept : m_vals{src.m_vals}, m_nnz(std::move(src.m_nnz)), m_capacity{std::move(src.m_capacity)}, m_shape(std::move(src.m_shape))
        {
            src.m_vals = nullptr;
            src.m_nnz = 0;
            src.m_capacity = 0;
            for (size_type i = 0; i < rank; ++i)
            {
                src.m_shape[i] = 0;
            }
        }
        template <typename Container, typename = other_move_constructable_type<Container, self_type>>
        special_matrix_base(Container &&src) : special_matrix_base() { CALL_AND_HANDLE(move_assign_impl(std::forward<Container>(src)), "Failed to move construct special matrix object from expression."); }

        // construct from three std::vector containing the vals, colind and rowptr arrays and the number of columns.  This assumes that the
        // input csr matrix is a square matrix
        special_matrix_base(const std::vector<value_type> &vals) : special_matrix_base()
        {
            size_type _nrows;
            CALL_AND_HANDLE(_nrows = impl::nrows_from_nnz(vals.size()), "Failed to construct special matrix base object.  Unable to determine the number of rows from val array size.");
            CALL_AND_HANDLE(init(vals, _nrows, _nrows), "Failed to construct special matrix base object.  Error when initialising from vectors.");
        }

        special_matrix_base(const std::vector<value_type> &vals, size_type _nrows) : special_matrix_base()
        {
            CALL_AND_HANDLE(init(vals, _nrows, _nrows), "Failed to construct special matrix base object.  Error when initialising from vectors.");
        }

        special_matrix_base(const std::vector<value_type> &vals, size_type _nrows, size_type _ncols) : special_matrix_base()
        {
            CALL_AND_HANDLE(init(vals, _nrows, _ncols), "Failed to construct special matrix base object.  Error when initialising from vectors.");
        }

        template <typename srcbck>
        special_matrix_base(const tensor<value_type, 1, srcbck> &vals) : special_matrix_base()
        {
            size_type _nrows;
            CALL_AND_HANDLE(_nrows = impl::nrows_from_nnz(vals.size()), "Failed to construct special matrix base object.  Unable to determine the number of rows from val array size.");
            CALL_AND_HANDLE(init(vals, _nrows, _nrows), "Failed to construct special matrix base object.  Error when initialising from vectors.");
        }

        template <typename srcbck>
        special_matrix_base(const tensor<value_type, 1, srcbck> &vals, size_type _nrows) : special_matrix_base()
        {
            CALL_AND_HANDLE(init(vals, _nrows, _nrows), "Failed to construct special matrix base object.  Error when initialising from vectors.");
        }

        template <typename srcbck>
        special_matrix_base(const tensor<value_type, 1, srcbck> &vals, size_type _nrows, size_type _ncols) : special_matrix_base()
        {
            CALL_AND_HANDLE(init(vals, _nrows, _ncols), "Failed to construct special matrix base object.  Error when initialising from vectors.");
        }

        ~special_matrix_base()
        {
            try
            {
                if (m_vals != nullptr)
                {
                    allocator::deallocate(m_vals);
                }
            }
            catch (...)
            {
            }
        }

        self_type &operator=(const self_type &src)
        {
            if (&src != this)
            {
                CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy assign special matrix base type.");
            }
            return *this;
        }
        template <typename Container>
        inline other_copy_assignable_type<Container, self_type> operator=(const Container &src)
        {
            CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy assign special matrix base type.");
            return *this;
        }

        self_type &operator=(self_type &&src)
        {
            if (&src != this)
            {
                if (m_vals != nullptr)
                {
                    deallocate();
                }
                m_vals = src.m_vals;
                src.m_vals = nullptr;
                m_nnz = std::move(src.m_nnz);
                src.m_nnz = 0;
                m_capacity = std::move(src.m_capacity);
                src.m_capacity = 0;
                m_shape = std::move(src.m_shape);
                src.m_shape = {0, 0};
            }
            return *this;
        }
        template <typename Container>
        inline other_move_assignable_type<Container, self_type> operator=(Container &&src) { CALL_AND_RETHROW(return move_assign_impl(std::forward<Container>(src))); }

        inline void deallocate()
        {
            if (m_vals != nullptr)
            {
                CALL_AND_HANDLE(allocator::deallocate(m_vals), "Deallocate failed.  Error when deallocating vals buffer.");
            }
            m_vals = nullptr;
            m_nnz = 0;
            m_capacity = 0;
            for (size_type i = 0; i < rank; ++i)
            {
                m_shape[i] = 0;
            }
        }

        inline void allocate(size_type _capacity)
        {
            ASSERT(m_vals == nullptr, "Unable to allocate special_matrix object.  It has already been allocated.");

            CALL_AND_HANDLE(m_vals = allocator::allocate(_capacity), "Failed to allocate csr matrix object.  Value buffer allocation failed.");
            m_capacity = _capacity;
        }

        inline void reallocate(size_type _capacity)
        {
            CALL_AND_HANDLE(deallocate(), "Failed to reallocate special_matrix object.  Error when deallocating previously allocated special_matrix object.");
            CALL_AND_HANDLE(allocate(_capacity), "Failed to reallocate special_matrix object.  Error when allocate new buffers.");
        }

        inline void clear() { CALL_AND_HANDLE(deallocate(), "Failed to clear special_matrix object."); }

        // reallocate will decrease the size of the array if it is not the correct size
        inline void resize(size_type _nrows, size_type _ncols)
        {
            m_shape[0] = _nrows;
            m_shape[1] = _ncols;
            size_type nnz;
            CALL_AND_HANDLE(nnz = impl::nnz_from_shape(m_shape), "Failed to resize the special matrix object.  Failed to determine the number of non-zero elements from the shape.");
            m_nnz = nnz;
            if (nnz > m_capacity)
            {
                if (m_vals != nullptr)
                {
                    CALL_AND_HANDLE(allocator::deallocate(m_vals), "Failed to resize the special matrix object.  Failed to deallocate the previous buffer object.");
                }
                CALL_AND_HANDLE(m_vals = allocator::allocate(m_nnz), "Failed to resize the special matrix object.  Failed to allocate the new buffer when an increase in size is required.");
                m_capacity = m_nnz;
            }
        }
        inline void resize(size_type _nrows) { CALL_AND_RETHROW(resize(_nrows, _nrows)); }
        inline void resize(const shape_type &_shape) { CALL_AND_RETHROW(resize(_shape[0], _shape[1])); }

        void set_vals(const std::vector<value_type> &vals)
        {
            ASSERT(vals.size() == m_nnz, "Failed to set special matrix vals from vector.  The two buffers are not the same size.");
            CALL_AND_HANDLE(memtransfer<blas_backend>::copy(&vals[0], m_nnz, m_vals), "Failed to set special matrix vals from vector.  Failed when copying the current vals buffer into the new vals buffer.");
        }

        // shrink all of the buffers so that their total capacity is the same as there used size.  This should keep all of the values in the array
        inline void shrink_to_fit()
        {
            if (m_nnz < m_capacity)
            {
                pointer temp;
                CALL_AND_HANDLE(temp = allocator::allocate(m_nnz), "Failed to resize the special matrix object.  Failed to allocate the new buffer when an increase in size is required.");
                CALL_AND_HANDLE(memtransfer<backend_type>::copy(m_vals, m_nnz, temp), "Failed to shrink the special matrix object.  Failed when copying the current vals buffer into the new buffer.");
                CALL_AND_HANDLE(allocator::deallocate(m_vals), "Failed to shrink the special matrix object.  Failed to deallocate the previous buffer object.");
                m_vals = temp;
                temp = nullptr;
                m_capacity = m_nnz;
            }
        }

        // methods for accessing the size of the special_matrix object.
        inline const size_type &capacity() const { return m_capacity; }

        inline const size_type &nnz() const { return m_nnz; }
        inline const size_type &nelems() const { return m_nnz; }
        inline const size_type &size() const { return m_nnz; }
        inline const shape_type &shape() const { return m_shape; }

        inline const size_type &size(size_type i) const
        {
            ASSERT(i < rank, "Unable to access special_matrix size element.  Index out of bounds.");
            return m_shape[i];
        }
        inline const size_type &dims(size_type i) const
        {
            ASSERT(i < rank, "Unable to access special_matrix size element.  Index out of bounds.");
            return m_shape[i];
        }
        inline const size_type &shape(size_type i) const
        {
            ASSERT(i < rank, "Unable to access special_matrix size element.  Index out of bounds.");
            return m_shape[i];
        }

        inline const size_type &nrows() const { return m_shape[0]; }
        inline const size_type &ncols() const { return m_shape[1]; }

        inline bool same_shape(const shape_type &_shape) const { return _shape == m_shape; }
#ifdef CEREAL_LIBRARY_FOUND
    public:
        using buffer_reader_type = internal::buffer_reader_wrapper<value_type, backend_type>;
        using buffer_writer_type = internal::buffer_writer_wrapper<value_type, backend_type>;

        template <typename archive>
        void save(archive &ar) const
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("buffer", buffer_writer_type{m_vals, m_capacity})), "Failed to serialise special sparse matrix object.  Error when serialising the data buffer.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("nnz", m_nnz)), "Failed to serialise special sparse matrix object.  Failed to serialise the number of nonzero elements.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("shape", m_shape)), "Failed to serialise special sparse matrix object.  Failed to serialise the matrix shape.");
        }

        template <typename archive>
        void load(archive &ar)
        {
            CALL_AND_HANDLE(ar(cereal::make_nvp("buffer", buffer_reader_type{&m_vals, &m_capacity})), "Failed to serialise special sparse matrix object.  Error when serialising the data buffer.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("nnz", m_nnz)), "Failed to serialise special sparse matrix object.  Failed to serialise the number of nonzero elements.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("shape", m_shape)), "Failed to serialise special sparse matrix object.  Failed to serialise the matrix shape.");
        }
#endif

    public:
        // Inplace scalar multiplication/division functions
        template <typename Vt>
        inline value_update_type<Vt, impl> operator*=(const Vt &v)
        {
            CALL_AND_HANDLE(backend_type::scal(m_nnz, value_type(v), m_vals, 1), "Failed to perform operator*= on csr matrix object.  scal call failed.");
            return *this;
        }
        template <typename Vt>
        inline value_update_type<Vt, impl> operator/=(const Vt &v)
        {
            CALL_AND_HANDLE(backend_type::scal(m_nnz, value_type(1.0 / v), m_vals, 1), "Failed to perform operator/= on csr matrix object.  scal call failed.");
            return *this;
        }

    public:
        inline pointer buffer() { return m_vals; }
        inline const_pointer buffer() const { return m_vals; }
        inline pointer data() { return m_vals; }
        inline const_pointer data() const { return m_vals; }

        void set_buffer(const value_type *vals, size_type size)
        {
            ASSERT(size == m_nnz, "Failed to set special matrix vals from vector.  The two buffers are not the same size.");
            CALL_AND_HANDLE(memtransfer<blas_backend>::copy(vals, m_nnz, m_vals), "Failed to set special matrix vals from vector.  Failed when copying the current vals buffer into the new vals buffer.");
        }

    public:
        void init(const std::vector<value_type> &vals, size_type _nrows, size_type _ncols)
        {
            size_type _size;
            shape_type tshape{{_nrows, _ncols}};
            CALL_AND_HANDLE(_size = impl::nnz_from_shape(tshape), "Failed to initialise the special matrix object.  Unable to determine the number of non-zero elements from the specified shape.");
            ASSERT(vals.size() == _size, "Failed to initialise the special matrix object.  The input value array is not compatible with the input number of rows or columns (its size must be equal to the smaller of the two).");

            CALL_AND_HANDLE(resize(_nrows, _ncols), "Failed to initialise the special matrix object.  Error when resizing.");
            CALL_AND_HANDLE(memtransfer<blas_backend>::copy(&vals[0], vals.size(), m_vals), "Failed to initialise the special matrix object.  Failed when copying the input vals buffer into the new vals buffer.");
        }

        template <typename srcbck>
        void init(const tensor<value_type, 1, srcbck> &vals, size_type _nrows, size_type _ncols)
        {
            size_type _size;
            shape_type tshape{{_nrows, _ncols}};
            CALL_AND_HANDLE(_size = impl::nnz_from_shape(tshape), "Failed to initialise the special matrix object.  Unable to determine the number of non-zero elements from the specified shape.");
            ASSERT(vals.size() == _size, "Failed to initialise the special matrix object.  The input value array is not compatible with the input number of rows or columns (its size must be equal to the smaller of the two).");

            CALL_AND_HANDLE(resize(_nrows, _ncols), "Failed to initialise the special matrix object.  Error when resizing.");
            CALL_AND_HANDLE(memtransfer<srcbck>::copy(vals.buffer(), vals.size(), m_vals), "Failed to initialise the special matrix object.  Failed when copying the input vals buffer into the new vals buffer.");
        }

    private:
        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container &src)
        {
            // first resize this buffer so that it can store the results of src
            CALL_AND_HANDLE(resize(src.shape()), "Failed to copy assign special_matrix_base object.  Failed to resize buffers so that they could fit the src matrix.");
            static_assert(traits<impl>::is_mutable, "Failed to initialise copy assignment operator for special matrix object.  The specified special matrix is not mutable.");
            using srcbck = typename traits<Container>::backend_type;
            CALL_AND_HANDLE(memtransfer<srcbck>::copy(src.buffer(), src.nnz(), m_vals), "Failed to shrink the special_matrix object.  Failed when copying the src value buffer into m_vals.");
            return *this;
        }
        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container &src)
        {
            static_assert(is_same_backend<Container, impl>::value, "Unable to instantiate copy_assign_impl.");
            static_assert(traits<impl>::is_mutable, "Failed to initialise copy assignment operator for special matrix object.  The specified special matrix is not mutable.");
            // first resize this buffer so that it can store the results of src
            CALL_AND_HANDLE(resize(src.shape()), "Failed to copy assign special_matrix_base object.  Failed to resize buffers so that they could fit the src matrix.");
            CALL_AND_HANDLE(backend_type::copy_real_to_complex(src.buffer(), src.nnz(), m_vals), "Failed to shrink the special_matrix object.  Failed when copying the src value buffer into m_vals.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container &expr)
        {
            static_assert(traits<impl>::is_mutable, "Failed to initialise copy assignment operator for special matrix object.  The specified special matrix is not mutable.");
            bool requires_resize;
            CALL_AND_HANDLE(requires_resize = !(expr.shape() == m_shape), "Copy assignment failed. Failed to determine whether the special matrix requires resizing before evaluation of the input expression.");
            if (requires_resize)
            {
                CALL_AND_HANDLE(resize(expr.shape()), "Copy assignment failed.  Failed to resize the special matrix object.");
            }
            CALL_AND_HANDLE(expr(static_cast<impl &>(*this)), "Copy assignment failed.  Failed to evaluate the expression into the special matrix object.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type move_assign_impl(Container &&expr)
        {
            static_assert(traits<impl>::is_mutable, "Failed to initialise copy assignment operator for special matrix object.  The specified special matrix is not mutable.");
            bool requires_resize;
            CALL_AND_HANDLE(requires_resize = !(expr.shape() == m_shape), "Copy assignment failed. Failed to determine whether the special matrix requires resizing before evaluation of the input expression.");
            if (requires_resize)
            {
                CALL_AND_HANDLE(resize(expr.shape()), "Copy assignment failed.  Failed to resize the special matrix object.");
            }
            CALL_AND_HANDLE(expr(static_cast<impl &>(*this)), "Copy assignment failed.  Failed to evaluate the expression into the special matrix object.");
            return *this;
        }
    }; // class special_matrix_base

} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_SPARSE_SPECIAL_MATRIX_BASE_HPP_//
