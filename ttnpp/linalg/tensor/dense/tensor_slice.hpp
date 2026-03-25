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

#ifndef PYTTN_LINALG_TENSOR_DENSE_TENSOR_SLICE_BASE_HPP_
#define PYTTN_LINALG_TENSOR_DENSE_TENSOR_SLICE_BASE_HPP_

#include "../../linalg_forward_decl.hpp"
#include "tensor_slice_traits.hpp"
#include "tensor_details.hpp"

/**
 * @cond INTERNAL
 */

namespace linalg
{
    //////////////////////////////////////////////////////////////////////////////////////////
    // CRTP base class for the arbitrary rank, reduced dimensional slice of a tensor.  This  //
    // provide the majority of the functionality required for the implementations.           //
    //////////////////////////////////////////////////////////////////////////////////////////
    template <typename tensor_slice_impl>
    class tensor_slice_base : public tensor_details<tensor_slice_impl>, public dense_tensor_type<traits<tensor_slice_impl>::rank>
    {
    public:
        using value_type = typename traits<tensor_slice_impl>::value_type;
        using device_value_type = typename traits<tensor_slice_impl>::device_value_type;
        using backend_type = typename traits<tensor_slice_impl>::backend_type;
        using size_type = typename traits<backend_type>::size_type;
        using pointer = typename std::add_pointer<value_type>::type;
        using const_pointer = typename std::add_pointer<typename std::add_const<device_value_type>::type>::type;
        using device_pointer = typename std::add_pointer<device_value_type>::type;
        using const_device_pointer = typename std::add_pointer<typename std::add_const<device_value_type>::type>::type;

        using container_type = typename traits<tensor_slice_impl>::container_type;
        static constexpr size_type rank = traits<tensor_slice_impl>::rank;
        using slice_traits = tensor_slice_traits<container_type, value_type, rank>;

        static constexpr size_type container_rank = slice_traits::container_rank;
        static constexpr size_type first_index = (container_rank - rank);

        using shape_type = std::array<size_type, rank>;
        using container_pointer = typename slice_traits::container_pointer;
        using self_type = tensor_slice_base<tensor_slice_impl>;
        using detail_type = tensor_details<tensor_slice_impl>;

    public:
        using memfill = memory::filler<device_value_type, backend_type>;
        template <typename srcbck>
        using memtransfer = memory::transfer<srcbck, backend_type>;

    protected:
        container_pointer m_tensor;
        device_pointer m_buffer;
        size_type m_i; // parameters for indexing the

    public:
        tensor_slice_base(container_pointer tensor, device_pointer _buffer, size_type i) : m_tensor(tensor), m_buffer(_buffer + i * tensor->stride(first_index - 1)), m_i(i) {}
        tensor_slice_base(tensor_slice_impl &&impl) : m_tensor(std::move(impl.m_tensor)), m_buffer(std::move(impl.m_buffer)), m_i(std::move(impl.m_i)) {}

        tensor_slice_base(const tensor_slice_base &o) = default;
        tensor_slice_base(tensor_slice_base &&o) = default;
        /**
         *  Value assignment operator.  This sets all elements of the tensor to a specific value.
         *  \param val The value that the tensor will be set to.
         */
        template <typename U, typename = typename std::enable_if<std::is_convertible<U, value_type>::value, void>::type>
        inline self_type &operator=(const U &_val)
        {
            ASSERT(m_buffer != nullptr, "Unable to fill tensor object.  The buffer has not been allocated");
            CALL_AND_HANDLE(fill_impl(_val), "Failed to value assign each element of the array.  Failed to fill the buffer.");
            return *this;
        }

        // copy assignment operator.
        inline self_type &operator=(const self_type &src)
        {
            if (this != &src)
            {
                CALL_AND_RETHROW(copy_assign_impl(src));
            }
            return *this;
        }
        template <typename Container>
        inline other_copy_assignable_type<Container, self_type> operator=(const Container &src) { CALL_AND_RETHROW(return copy_assign_impl(src)); }
        template <typename Container>
        inline other_move_assignable_type<Container, self_type> operator=(Container &&src) { CALL_AND_RETHROW(return move_assign_impl(std::forward<Container>(src))); }

        template <typename Container>
        inline typename std::enable_if<is_buffer_copyable_dense<Container, self_type>::value, self_type &>::type set_buffer(const Container &src)
        {
            ASSERT(size() == src.size(), "Failed to copy buffer from input container.  The two objects do not have the same size.");
            using srcbck = typename Container::backend_type;
            CALL_AND_HANDLE(memtransfer<srcbck>::copy(reinterpret_cast<const_device_pointer>(src.buffer()), src.size(), m_buffer), "Copy assignment operator failed.  Error when copying the buffer.");
            return *this;
        }

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////
        //                   Addition assignment from generic tensor base types                  //
        ///////////////////////////////////////////////////////////////////////////////////////////
        inline self_type &operator+=(const self_type &src)
        {
            if (this != &src)
            {
                CALL_AND_RETHROW(addition_assign_impl(src));
            }
            return *this;
        }
        template <typename Container>
        inline other_copy_assignable_type<Container, self_type> operator+=(const Container &src) { CALL_AND_RETHROW(return addition_assign_impl(src)); }
        template <typename Container>
        inline other_move_assignable_type<Container, self_type> operator+=(Container &&src) { CALL_AND_RETHROW(return addition_assign_impl(std::forward<Container>(src))); }

        ///////////////////////////////////////////////////////////////////////////////////////////
        //                 Subtraction assignment from generic tensor base types                 //
        ///////////////////////////////////////////////////////////////////////////////////////////
        inline self_type &operator-=(const self_type &src)
        {
            if (this != &src)
            {
                CALL_AND_RETHROW(subtraction_assign_impl(src));
            }
            return *this;
        }
        template <typename Container>
        inline other_copy_assignable_type<Container, self_type> operator-=(const Container &src) { CALL_AND_RETHROW(return subtraction_assign_impl(src)); }
        template <typename Container>
        inline other_move_assignable_type<Container, self_type> operator-=(Container &&src) { CALL_AND_RETHROW(return subtraction_assign_impl(std::forward<Container>(src))); }

    private:
        template <typename Container>
        inline typename std::enable_if<!is_expression<Container>::value && is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container &src)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            using srcbck = typename traits<Container>::backend_type;
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == src.shape(i), "Failed to copy assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(memtransfer<srcbck>::copy(reinterpret_cast<const_device_pointer>(src.buffer()), size(), m_buffer), "Failed to copy assign tensor_slice_base object.  Error when copying the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container &src)
        {
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate copy_assign_impl.");
            static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == src.shape(i), "Failed to copy assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(backend_algebra<backend_type>::copy_real_to_complex(src.buffer(), size(), m_buffer), "Copy operator failed.  Error when copying the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container &expr)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            shape_type _shape;
            CALL_AND_HANDLE(_shape = expr.shape(), "Failed to copy assign tensor_slice object.  Failed to get the shape of the expression object.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == _shape[i], "Failed to copy assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(expr(*this), "Failed to copy assign tensor_slice object.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type move_assign_impl(Container &&expr)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise copy assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            shape_type _shape;
            CALL_AND_HANDLE(_shape = expr.shape(), "Failed to copy assign tensor_slice object.  Failed to get the shape of the expression object.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == _shape[i], "Failed to copy assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(expr(*this), "Failed to copy assign tensor_slice object.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

    private:
        template <typename Container>
        inline typename std::enable_if<!is_expression<Container>::value && is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type
        addition_assign_impl(const Container &src)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == src.shape(i), "Failed to addition assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(backend_algebra<backend_type>::addition_assign(reinterpret_cast<const_device_pointer>(src.buffer()), src.size(), m_buffer), "Addition assignment operator failed.  Error when additioning the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type
        addition_assign_impl(const Container &src)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == src.shape(i), "Failed to addition assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(backend_algebra<backend_type>::addition_assign_real_to_complex(src.buffer(), src.size(), m_buffer), "Addition assignment operator failed.  Error when copying the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type
        addition_assign_impl(const Container &expr)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise addition assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            shape_type _shape;
            CALL_AND_HANDLE(_shape = expr.shape(), "Failed to addition assign tensor_slice object.  Failed to get the shape of the expression object.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == _shape[i], "Failed to addition assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(expr.add_assignment(*this), "Addition assignment failed.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type addition_assign_impl(Container &&expr)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise subtract assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            shape_type _shape;
            CALL_AND_HANDLE(_shape = expr.shape(), "Failed to subtract assign tensor_slice object.  Failed to get the shape of the expression object.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == _shape[i], "Failed to subtract assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(expr.add_assignment(*this), "Failed to subtract assign tensor_slice object.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

    private:
        template <typename Container>
        inline typename std::enable_if<!is_expression<Container>::value && is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type
        subtraction_assign_impl(const Container &src)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise subtraction assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == src.shape(i), "Failed to subtraction assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(backend_algebra<backend_type>::subtraction_assign(reinterpret_cast<const_device_pointer>(src.buffer()), src.size(), m_buffer), "Subtraction assignment operator failed.  Error when subtractioning the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type
        subtraction_assign_impl(const Container &src)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise subtraction assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate subtraction_assign_impl.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == src.shape(i), "Failed to subtraction assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(backend_algebra<backend_type>::subtraction_assign_real_to_complex(src.buffer(), src.size(), m_buffer), "Subtraction assignment operator failed.  Error when subtractioning the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type
        subtraction_assign_impl(const Container &expr)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise subtraction assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            shape_type _shape;
            CALL_AND_HANDLE(_shape = expr.shape(), "Failed to subtraction assign tensor_slice object.  Failed to get the shape of the expression object.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == _shape[i], "Failed to subtraction assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(expr.subtract_assignment(*this), "Subtraction assignment failed.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type subtraction_assign_impl(Container &&expr)
        {
            static_assert(traits<self_type>::is_mutable, "Failed to initialise subtract assignment operator for tensor_slice object.  The specified tensor slice is not mutable.");
            shape_type _shape;
            CALL_AND_HANDLE(_shape = expr.shape(), "Failed to subtract assign tensor_slice object.  Failed to get the shape of the expression object.");
            for (size_type i = 0; i < rank; ++i)
            {
                ASSERT(shape(i) == _shape[i], "Failed to subtract assign tensor_slice object.  The two slice object do not have the same shape.");
            }
            CALL_AND_HANDLE(expr.subtract_assignment(*this), "Failed to subtract assign tensor_slice object.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

    public:
        template <typename Container>
        inline typename std::enable_if<is_buffer_copyable_dense<Container, tensor_slice_impl>::value, self_type &>::type copy_buffer(const Container &src)
        {
            ASSERT(size() == src.size(), "Failed to copy buffer from input container.  The two objects do not have the same size.");
            using srcbck = typename Container::backend_type;
            CALL_AND_HANDLE(memtransfer<srcbck>::copy(reinterpret_cast<const_device_pointer>(src.buffer()), src.size(), m_buffer), "Copy assignment operator failed.  Error when copying the buffer.");
            return *this;
        }

    public:
        inline const size_type &capacity() const { return m_tensor->stride(first_index - 1); }
        inline const size_type &size() const { return m_tensor->stride(first_index - 1); }
        inline const size_type &nelems() const { return m_tensor->stride(first_index - 1); }
        inline const shape_type shape() const
        {
            shape_type _shape;
            for (size_type i = first_index; i < container_rank; ++i)
            {
                _shape[i - first_index] = m_tensor->shape(i);
            }
            return _shape;
        }
        inline const shape_type stride() const
        {
            shape_type _shape;
            for (size_type i = first_index; i < container_rank; ++i)
            {
                _shape[i - first_index] = m_tensor->stride(i);
            }
            return _shape;
        }


        inline const size_type &size(size_type i) const { CALL_AND_HANDLE(return m_tensor->shape(i + first_index), "Failed to return size of tensor slice object. Failed when accessing underlying tensor objects sizes."); }
        inline const size_type &dims(size_type i) const { CALL_AND_HANDLE(return m_tensor->shape(i + first_index), "Failed to return size of tensor slice object. Failed when accessing underlying tensor objects sizes."); }
        inline const size_type &shape(size_type i) const { CALL_AND_HANDLE(return m_tensor->shape(i + first_index), "Failed to return size of tensor slice object. Failed when accessing underlying tensor objects sizes."); }
        inline const size_type &slice_index() const { return m_i; }

        inline const size_type &stride(size_type i) const { CALL_AND_HANDLE(return m_tensor->stride(i + first_index), "Failed to return size of tensor slice object. Failed when accessing underlying tensor objects sizes."); }

        inline bool same_shape(const shape_type &_shape) const { return _shape == shape(); }

        inline device_pointer buffer() { return m_buffer; }
        inline const_device_pointer buffer() const { return m_buffer; }
        inline device_pointer data() { return m_buffer; }
        inline const_device_pointer data() const { return m_buffer; }

    public:
        template <typename... Args>
        inline reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args), backend_type> reinterpret_shape(Args &&...args) const
        {
            using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args), backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(size(), m_buffer, std::forward<Args>(args)...), "Failed to reinterpret shape of tensor_slice object.");
        }

        template <typename... Args>
        inline reinterpreted_tensor<value_type, sizeof...(Args), backend_type> reinterpret_shape(Args &&...args)
        {
            using reinterpreted_type = reinterpreted_tensor<value_type, sizeof...(Args), backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(size(), m_buffer, std::forward<Args>(args)...), "Failed to reinterpret shape of tensor_slice object.");
        }

        template <size_t vD>
        inline reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type> reinterpret_shape(const std::array<size_type, vD> &_size) const
        {
            using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(size(), m_buffer, _size), "Failed to reinterpret the shape of the tensor_slice object.");
        }

        template <size_t vD>
        inline reinterpreted_tensor<value_type, vD, backend_type> reinterpret_shape(const std::array<size_type, vD> &_size)
        {
            using reinterpreted_type = reinterpreted_tensor<value_type, vD, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(size(), m_buffer, _size), "Failed to reinterpret the shape of the tensor_slice object.");
        }

    public:
        template <typename U>
        inline typename std::enable_if<std::is_convertible<U, value_type>::value, self_type &>::type fill_value(const U &u)
        {
            CALL_AND_HANDLE(fill_impl(u), "Failed to fill buffer with value.  Error when calling fill impl.");
            return *this;
        }
        inline self_type &fill_zeros()
        {
            CALL_AND_HANDLE(fill_impl(device_value_type(0.0)), "Failed to fill buffer to zero.");
            return *this;
        }
        inline self_type &fill_ones()
        {
            CALL_AND_HANDLE(fill_impl(device_value_type(1.0)), "Failed to fill buffer to one.");
            return *this;
        }

    private:
        template <bool _mutable = traits<self_type>::is_mutable, typename U = value_type>
        typename std::enable_if<_mutable && std::is_convertible<U, value_type>::value, void>::type
        fill_impl(const U &v) { CALL_AND_HANDLE(memfill::fill(m_buffer, size(), device_value_type(v)), "Failed to set buffer to value.  Error when calling the memfill object fill function."); }

    public:
        // Inplace scalar multiplication/division functions
        template <typename Vt>
        inline value_update_type<Vt, self_type> operator*=(const Vt &v)
        {
            CALL_AND_HANDLE(backend_algebra<backend_type>::scal(size(), device_value_type(v), m_buffer, 1), "Failed to perform operator*= on tensor slice object.  scal call failed.");
            return *this;
        }
        template <typename Vt>
        inline value_update_type<Vt, self_type> operator/=(const Vt &v)
        {
            CALL_AND_HANDLE(backend_algebra<backend_type>::scal(size(), device_value_type(1.0 / v), m_buffer, 1), "Failed to perform operator/= on tensor slice object.  scal call failed.");
            return *this;
        }
    }; // class tensor_slice_base

    template <template <typename, size_t, typename> class ArrType, typename T, size_t D1, typename backend, size_t D2, typename pref>
    class tensor_slice<ArrType<T, D1, backend>, pref, D2> : public tensor_slice_base<tensor_slice<ArrType<T, D1, backend>, pref, D2>>
    {
    public:
        using array_type = ArrType<T, D1, backend>;
        static_assert(is_dense_tensor<array_type>::value, "Failed to instantiate tensor_slice object.  The input array type is not a valid dense tensor type.");
        using self_type = tensor_slice<array_type, pref, D2>;
        using slice_base = tensor_slice_base<self_type>;
        using slice_traits = tensor_slice_traits<array_type, pref, D2>;

        using pointer = typename slice_base::pointer;
        using const_pointer = typename slice_base::const_pointer;
        using const_slice_traits = tensor_slice_traits<array_type, typename std::add_const<pref>::type, D2>;
        using size_type = typename slice_base::size_type;
        using const_reference = const T&;
        using reference = T&;


    public:
        template <typename... Args>
        tensor_slice(Args &&...args);
        template <typename... Args>
        self_type &operator=(Args &&...args);
        // accessor operator[] for returning slices
        inline typename slice_traits::slice_type operator[](size_type i);
        inline typename const_slice_traits::slice_type operator[](size_type i) const;
        inline typename slice_traits::slice_type slice(size_type i);
        inline typename const_slice_traits::slice_type slice(size_type i) const;

        // accessor which accesses the tensor as a 1d array
        inline reference operator()(size_type index);
        inline const_reference operator()(size_type index) const;
        inline reference at(size_type i);
        inline const_reference at(size_type i) const;

        // general accessor functions
        template <typename... Inds> inline reference operator()(Inds... indices);
        template <typename... Inds> inline const_reference operator()(Inds... indices) const;
        template <typename... Inds> inline reference at(Inds... indices);
        template <typename... Inds> inline const_reference at(Inds... indices) const;
    };

    template <template <typename, size_t, typename> class ArrType, typename T, size_t D1, typename backend, typename pref>
    class tensor_slice<ArrType<T, D1, backend>, pref, 1> : public tensor_slice_base<tensor_slice<ArrType<T, D1, backend>, pref, 1>>
    {
    public:
        using array_type = ArrType<T, D1, backend>;
        static_assert(is_dense_tensor<array_type>::value, "Failed to instantiate tensor_slice object.  The input array type is not a valid dense tensor type.");
        using self_type = tensor_slice<array_type, pref, 1>;
        using slice_base = tensor_slice_base<self_type>;
        using slice_traits = tensor_slice_traits<array_type, pref, 1>;
        using size_type = typename slice_base::size_type;

        using pointer = typename slice_base::pointer;
        using const_pointer = typename slice_base::const_pointer;
        using const_reference = const T&;
        using reference = T&;

    public:
        template <typename... Args>
        tensor_slice(Args &&...args);
        template <typename... Args>
        self_type &operator=(Args &&...args);

        inline reference operator[](size_type i);
        inline reference operator()(size_type i);
        inline const_reference operator[](size_type i) const;
        inline const_reference operator()(size_type i) const;

        inline reference slice(size_type i);
        inline reference at(size_type i);
        inline const_reference slice(size_type i) const;
        inline const_reference at(size_type i) const; 
    };

} // namespace linalg

///@endcond

#endif // PYTTN_LINALG_TENSOR_DENSE_TENSOR_SLICE_BASE_HPP_//
