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

#ifndef PYTTN_LINALG_TENSOR_DENSE_TENSOR_BLAS_HPP_
#define PYTTN_LINALG_TENSOR_DENSE_TENSOR_BLAS_HPP_

#include "../../../backends/blas/blas_backend.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"

// TODO: Implement stl allocators (and potentially an aligned allocator) to handle memory rather than the hacky approach I have currently taken.
namespace linalg
{
    ///////////////////////////////////////////////////////////////////////////////////////
    // D dimensional implementation of the general tensor object for use with the blas   //
    //                                     backend                                       //
    ///////////////////////////////////////////////////////////////////////////////////////
    template <typename T, size_t D>
    class tensor<T, D, blas_backend> : public tensor_base<tensor<T, D, blas_backend>>
    {
    public:
        using self_type = tensor<T, D, blas_backend>;
        using size_type = typename traits<self_type>::size_type;
        using base_type = tensor_base<self_type>;

        using value_type = T;
        using device_value_type = T;
        using reference = typename std::add_lvalue_reference<T>::type;
        using const_reference = typename std::add_lvalue_reference<typename std::add_const<T>::type>::type;

        using const_slice_traits = tensor_slice_traits<self_type, typename std::add_const<T>::type, D>;
        using slice_traits = tensor_slice_traits<self_type, T, D>;

        using const_slice_type = typename const_slice_traits::slice_type;
        using slice_type = typename slice_traits::slice_type;

        friend class internal::tensor_buffer_swap;

    protected:
        using base_type::m_buffer;
        using base_type::m_shape;
        using base_type::m_stride;
        using base_type::m_totsize;

    public:
        template <typename... Args>
        tensor(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct tensor object.");
        }
        template <typename... Args>
        self_type &operator=(Args &&...args)
        {
            CALL_AND_RETHROW(base_type::operator=(std::forward<Args>(args)...));
            return *this;
        }

        // accessor operator[] for returning slices
        inline slice_type operator[](size_type i) { return slice_traits::make(this, i); }
        inline const_slice_type operator[](size_type i) const { return const_slice_traits::make(this, i); }
        inline slice_type slice(size_type i)
        {
            ASSERT(internal::compare_bounds(i, m_shape[0]), "Unable to return slice of array.  Slice index out of bounds.");
            return slice_traits::make(this, i);
        }
        inline const_slice_type slice(size_type i) const
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
            return m_buffer[NDIndex<D>::flatten(m_stride, indices...)];
        }

        template <typename... Inds>
        inline const_reference operator()(Inds... indices) const
        {
            static_assert(sizeof...(Inds) == D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
            return m_buffer[NDIndex<D>::flatten(m_stride, indices...)];
        }

        template <typename... Inds>
        inline reference at(Inds... indices)
        {
            static_assert(sizeof...(Inds) == D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
            size_type index;
            CALL_AND_HANDLE(index = NDIndex<D>::flatten_check(m_shape, m_stride, indices...), "Unable to access tensor element.  Failed to determine flattened index.");
            return m_buffer[index];
        }

        template <typename... Inds>
        inline const_reference at(Inds... indices) const
        {
            static_assert(sizeof...(Inds) == D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
            size_type index;
            CALL_AND_HANDLE(index = NDIndex<D>::flatten_check(m_shape, m_stride, indices...), "Unable to access tensor element.  Failed to determine flattened index.");
            return m_buffer[index];
        }
    }; // class tensor

    template <typename T>
    class tensor<T, 1, blas_backend> : public tensor_base<tensor<T, 1, blas_backend>>
    {
    public:
        using self_type = tensor<T, 1, blas_backend>;
        using size_type = typename traits<blas_backend>::size_type;
        using base_type = tensor_base<self_type>;

        using value_type = T;
        using device_value_type = T;
        using reference = typename std::add_lvalue_reference<T>::type;
        using const_reference = typename std::add_lvalue_reference<typename std::add_const<T>::type>::type;

        friend class internal::tensor_buffer_swap;

    protected:
        using base_type::m_buffer;
        using base_type::m_totsize;

    public:
        template <typename... Args>
        tensor(Args &&...args)
        try : base_type(std::forward<Args>(args)...) {}
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to construct tensor object.");
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
            ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access element. Index out of bounds.");
            return m_buffer[i];
        }
        inline reference at(size_type i)
        {
            ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access element. Index out of bounds.");
            return m_buffer[i];
        }
        inline const_reference slice(size_type i) const
        {
            ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access element. Index out of bounds.");
            return m_buffer[i];
        }
        inline const_reference at(size_type i) const
        {
            ASSERT(internal::compare_bounds(i, m_totsize), "Failed to access element. Index out of bounds.");
            return m_buffer[i];
        }
    };

} // namespace linalg

#include "tensor_slice.hpp"

namespace linalg
{

    ///////////////////////////////////////////////////////////////////////////////////////
    //            ostream operators for the D dimensional blas tensor objects            //
    ///////////////////////////////////////////////////////////////////////////////////////
    template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
    typename std::enable_if<traits<array_type>::rank == 1 && !is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
    {
        os << "[";
        for (size_t i = 0; i < t.shape(0); ++i)
        {
            os << t(i) << (i + 1 == t.shape(0) ? "]" : ", ");
        }
        return os;
    }

    template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
    typename std::enable_if<traits<array_type>::rank == 2 && !is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
    {
        os << "[";
        for (size_t i = 0; i < t.shape(0); ++i)
        {
            os << "[";
            for (size_t j = 0; j < t.shape(1); ++j)
            {
                os << t(i, j) << (j + 1 == t.shape(1) ? "]" : ", ");
            }
            os << (i + 1 == t.shape(0) ? "]" : ",") << std::endl;
        }
        return os;
    }

    template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
    typename std::enable_if<traits<array_type>::rank == 3 && !is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
    {
        os << "[";
        for (size_t i = 0; i < t.shape(0); ++i)
        {
            os << "[";
            for (size_t j = 0; j < t.shape(1); ++j)
            {
                os << "[";
                for (size_t k = 0; k < t.shape(2); ++k)
                {
                    os << t(i, j, k) << (k + 1 == t.shape(2) ? "]" : ", ");
                }
                os << (j + 1 == t.shape(1) ? "]" : ",");
            }
            os << (i + 1 == t.shape(0) ? "]" : ",");
        }
        return os;
    }

    template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
    typename std::enable_if<(traits<array_type>::rank > 3) && !is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
    {
        os << "shape: [";
        for (size_t i = 0; i < t.rank; ++i)
        {
            os << t.shape(i) << (i + 1 == t.rank ? "]" : ", ");
        }
        os << std::endl
           << "data : [";
        for (size_t i = 0; i < t.size(); ++i)
        {
            os << t(i) << (i + 1 == t.size() ? "]" : ", ");
        }
        return os;
    }

    template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
    typename std::enable_if<traits<array_type>::rank == 1 && is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
    {
        using std::abs;
        os << "[";
        for (size_t i = 0; i < t.shape(0); ++i)
        {
            os << t(i).real() << (t(i).imag() < 0.0 ? "-" : "+") << abs(t(i).imag()) << "i" << (i + 1 == t.shape(0) ? "]" : ", ");
        }
        return os;
    }

    template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
    typename std::enable_if<traits<array_type>::rank == 2 && is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
    {
        using std::abs;
        os << "[";
        for (size_t i = 0; i < t.shape(0); ++i)
        {
            os << "[";
            for (size_t j = 0; j < t.shape(1); ++j)
            {
                os << t(i, j).real() << (t(i, j).imag() < 0.0 ? "-" : "+") << abs(t(i, j).imag()) << "i" << (j + 1 == t.shape(1) ? "]" : ", ");
            }
            os << (i + 1 == t.shape(0) ? "]" : ",") << std::endl;
        }
        return os;
    }

    template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
    typename std::enable_if<traits<array_type>::rank == 3 && is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
    {
        using std::abs;
        os << "[";
        for (size_t i = 0; i < t.shape(0); ++i)
        {
            os << "[";
            for (size_t j = 0; j < t.shape(1); ++j)
            {
                os << "[";
                for (size_t k = 0; k < t.shape(2); ++k)
                {
                    os << t(i, j, k).real() << (t(i, j, k).imag() < 0.0 ? "-" : "+") << abs(t(i, j, k).imag()) << "i" << (k + 1 == t.shape(2) ? "]" : ", ");
                }
                os << (j + 1 == t.shape(1) ? "]" : ",");
            }
            os << (i + 1 == t.shape(0) ? "]" : ",");
        }
        return os;
    }

    template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<blas_backend, typename traits<array_type>::backend_type>::value, void>::type>
    typename std::enable_if<(traits<array_type>::rank > 3) && is_complex<typename traits<array_type>::value_type>::value, std::ostream &>::type operator<<(std::ostream &os, const array_type &t)
    {
        using std::abs;
        os << "shape: [";
        for (size_t i = 0; i < t.rank; ++i)
        {
            os << t.shape(i) << (i + 1 == t.rank ? "]" : ", ");
        }
        os << std::endl
           << "data : [";
        for (size_t i = 0; i < t.size(); ++i)
        {
            os << t(i).real() << (t(i).imag() < 0.0 ? "-" : "+") << abs(t(i).imag()) << "i" << (i + 1 == t.size() ? "]" : ", ");
        }
        return os;
    }
}

#endif // PYTTN_LINALG_TENSOR_DENSE_TENSOR_HPP_//
