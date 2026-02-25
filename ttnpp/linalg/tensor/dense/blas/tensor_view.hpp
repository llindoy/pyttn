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

#ifndef PYTTN_LINALG_TENSOR_DENSE_TENSOR_VIEW_BLAS_HPP_
#define PYTTN_LINALG_TENSOR_DENSE_TENSOR_VIEW_BLAS_HPP_

#include "../../../backends/blas/blas_backend.hpp"
#include "../tensor_view.hpp"

namespace linalg
{
    ///////////////////////////////////////////////////////////////////////////////////////
    // D dimensional implementation of the tensor view object for use with the  //
    //                                    blas backend                                   //
    ///////////////////////////////////////////////////////////////////////////////////////
    template <typename T, size_t D>
    class tensor_view<T, D, blas_backend> : public tensor_view_base<tensor_view<T, D, blas_backend>>
    {
    public:
        using self_type = tensor_view<T, D, blas_backend>;
        using base_type = tensor_view_base<self_type>;
        using size_type = typename base_type::size_type;

        using value_type = T;
        using reference = typename std::add_lvalue_reference<T>::type;
        using const_reference = typename std::add_lvalue_reference<typename std::add_const<T>::type>::type;

        using const_slice_traits = tensor_slice_traits<self_type, typename std::add_const<T>::type, D>;
        using slice_traits = tensor_slice_traits<self_type, T, D>;

    protected:
        using base_type::m_buffer;
        using base_type::m_shape;
        using base_type::m_stride;
        using base_type::m_totsize;

    public:
        template <typename... Args>
        tensor_view(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct tensor view object.");
        }
        template <typename... Args>
        self_type &operator=(Args &&...args)
        {
            CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));
            return *this;
        }

    public:
        // accessor operator[] for returning slices
        inline typename slice_traits::slice_type operator[](size_type i) { return slice_traits::make(this, i); }
        inline typename const_slice_traits::slice_type operator[](size_type i) const { return const_slice_traits::make(this, i); }
        inline typename slice_traits::slice_type slice(size_type i)
        {
            ASSERT(internal::compare_bounds(i, m_shape[0]), "Unable to return slice of array.  Slice index out of bounds.");
            return slice_traits::make(this, i);
        }
        inline typename const_slice_traits::slice_type slice(size_type i) const
        {
            ASSERT(internal::compare_bounds(i, m_shape[0]), "Unable to return slice of array.  Slice index out of bounds.");
            return const_slice_traits::make(this, i);
        }

        // accessor which accesses the tensor as a 1d array
        inline reference operator()(size_type index) { return m_buffer[index]; }
        inline const_reference operator()(size_type index) const { return m_buffer[index]; }
        inline reference at(size_type i)
        {
            ASSERT(internal::compare_bounds(i, m_totsize), "Unable to access tensor element using at.  Index out of bounds.");
            return m_buffer[i];
        }
        inline const_reference at(size_type i) const
        {
            ASSERT(internal::compare_bounds(i, m_totsize), "Unable to access tensor element using at.  Index out of bounds.");
            return m_buffer[i];
        }

        // general accessor functions
        template <typename... Inds>
        inline reference operator()(Inds... indices)
        {
            static_assert(sizeof...(Inds) == D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
            using pack_type = typename internal::check_integral<Inds...>::pack_type;
            return m_buffer[get_index<pack_type>(indices...)];
        }

        template <typename... Inds>
        inline const_reference operator()(Inds... indices) const
        {
            static_assert(sizeof...(Inds) == D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
            using pack_type = typename internal::check_integral<Inds...>::pack_type;
            return m_buffer[get_index<pack_type>(indices...)];
        }

        template <typename... Inds>
        inline reference at(Inds... indices)
        {
            static_assert(sizeof...(Inds) == D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
            using pack_type = typename internal::check_integral<Inds...>::pack_type;
            size_type index;
            CALL_AND_HANDLE(index = get_index_bounds_check<pack_type>(indices...), "Unable to access tensor element.  Failed to determine flattened index.");
            return m_buffer[index];
        }

        template <typename... Inds>
        inline const_reference at(Inds... indices) const
        {
            static_assert(sizeof...(Inds) == D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
            using pack_type = typename internal::check_integral<Inds...>::pack_type;
            size_type index;
            CALL_AND_HANDLE(index = get_index_bounds_check<pack_type>(indices...), "Unable to access tensor element.  Failed to determine flattened index.");
            return m_buffer[index];
        }

    private:
        ///@cond INTERNAL - we might want to move this elsewhere - this should be common to all dense tensor types.
        // get the index in the array corresponding to the parameter pack.
        template <typename IntegerType, typename... Args>
        inline size_type get_index_bounds_check(IntegerType i, Args... args) const
        {
            ASSERT(internal::compare_bounds(i, m_shape[D - sizeof...(args) - 1]), "Unable to get flattened index.  One of the unflattened indices was out of bounds.");
            CALL_AND_HANDLE(return i * m_stride[D - sizeof...(args) - 1] + get_index_bounds_check<IntegerType>(args...), "Unable to get flattened index.  Error on iterated get_index call.");
        }
        template <typename IntegerType>
        inline size_type get_index_bounds_check(IntegerType i) const
        {
            ASSERT(internal::compare_bounds(i, m_shape[D - 1]), "Unable to get flattened index.  Final unflattened index was out of bounds.");
            return i;
        }

        template <typename IntegerType, typename... Args>
        inline size_type get_index(IntegerType i, Args... args) const { return i * m_stride[D - sizeof...(args) - 1] + get_index<IntegerType>(args...); }
        template <typename IntegerType>
        inline size_type get_index(IntegerType i) const { return i; }
        ///@endcond
    };

    ///////////////////////////////////////////////////////////////////////////////////////
    // 1 dimensional implementation of the tensor view object for use with the  //
    //                                    blas backend                                   //
    ///////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    class tensor_view<T, 1, blas_backend> : public tensor_view_base<tensor_view<T, 1, blas_backend>>
    {
    public:
        using self_type = tensor_view<T, 1, blas_backend>;
        using base_type = tensor_view_base<self_type>;
        using size_type = typename base_type::size_type;

        using value_type = T;
        using reference = typename std::add_lvalue_reference<T>::type;
        using const_reference = typename std::add_lvalue_reference<typename std::add_const<T>::type>::type;

    protected:
        using base_type::m_buffer;
        using base_type::m_totsize;

    public:
        template <typename... Args>
        tensor_view(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct tensor view object.");
        }
        template <typename... Args>
        self_type &operator=(Args &&...args)
        {
            CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));
            return *this;
        }

        inline reference operator[](size_type i) { return m_buffer[i]; }
        inline reference operator()(size_type i) { return m_buffer[i]; }
        inline const_reference operator[](size_type i) const { return m_buffer[i]; }
        inline const_reference operator()(size_type i) const { return m_buffer[i]; }

        inline reference slice(size_type i)
        {
            ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access slice. Index out of bounds.");
            return m_buffer[i];
        }
        inline reference at(size_type i)
        {
            ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access slice. Index out of bounds.");
            return m_buffer[i];
        }
        inline const_reference slice(size_type i) const
        {
            ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access slice. Index out of bounds.");
            return m_buffer[i];
        }
        inline const_reference at(size_type i) const
        {
            ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access slice. Index out of bounds.");
            return m_buffer[i];
        }
    };
} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_DENSE_TENSOR_VIEW_HPP_
