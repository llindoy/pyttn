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

#ifndef PYTTN_LINALG_TENSOR_DENSE_TENSOR_BASE_HPP_
#define PYTTN_LINALG_TENSOR_DENSE_TENSOR_BASE_HPP_

#include "../../linalg_forward_decl.hpp"
#include "../../linalg_type_traits.hpp"

#include "tensor_view.hpp"
#include "tensor_details.hpp"
#include "tensor_slice_traits.hpp"

#include "../../utils/serialisation.hpp"
#include "../../utils/memory_helper.hpp"


// TODO: Implement stl allocators (and potentially an aligned allocator) to handle memory rather than the hacky approach I have currently taken.
namespace linalg
{

    namespace internal
    {
        class tensor_buffer_swap
        {
        public:
            template <typename T1, typename T2>
            static void swap_buffers(T1 &t1, T2 &t2)
            {
                static_assert(is_dense_tensor<T1>::value && is_dense_tensor<T2>::value, "Unable to swap buffers of objects.  The two input types are not dense tensors.");
                static_assert(is_same_value<T1, T2>::value, "Unable to swap buffers of tensor objects.  They do not have the same value type.");
                static_assert(is_same_backend<T1, T2>::value, "Unable to swap buffers of tensor objects.  They do not have the same backend type.");

                using value_type = typename traits<T1>::device_value_type;
                using pointer = typename std::add_pointer<value_type>::type;
                using size_type = typename T2::size_type;
                ASSERT(t1.size() <= t2.capacity() && t2.size() <= t1.capacity(), "Unable to swap the two buffers they do not both have sizes less than the capacity of the other.");

                pointer temp = t1.m_buffer;
                t1.m_buffer = t2.m_buffer;
                t2.m_buffer = temp;
                size_type cap = t1.capacity();
                t1.m_totcapacity = t2.capacity();
                t2.m_totcapacity = cap;
            }
        };
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // CRTP base class for the arbitrary rank, dynamically sized tensor object.  This provide//
    // the majority of the functionality required for the implementations.                   //
    //////////////////////////////////////////////////////////////////////////////////////////
    template <typename tensor_impl>
    class tensor_base : public tensor_details<tensor_impl>, public dense_tensor_type<traits<tensor_impl>::rank>
    {
    public:
        using backend_type = typename traits<tensor_impl>::backend_type;
        using value_type = typename traits<tensor_impl>::value_type;
        using device_value_type = typename traits<tensor_impl>::device_value_type;
        using size_type = typename traits<backend_type>::size_type;
        static constexpr size_type rank = traits<tensor_impl>::rank;

        using pointer = typename std::add_pointer<value_type>::type;
        using const_pointer = typename std::add_pointer<typename std::add_const<value_type>::type>::type;
        using device_pointer = typename std::add_pointer<device_value_type>::type;
        using const_device_pointer = typename std::add_pointer<typename std::add_const<device_value_type>::type>::type;

        using shape_type = std::array<size_type, rank>;
        using self_type = tensor_base<tensor_impl>;

        // the classes used for memory allocation and transfer
        using allocator = memory::allocator<device_value_type, backend_type>;
        using memfill = memory::filler<device_value_type, backend_type>;
        template <typename srcbck>
        using memtransfer = memory::transfer<srcbck, backend_type>;

        using detail_type = tensor_details<tensor_impl>;

        friend class internal::tensor_buffer_swap;

    protected:
        device_pointer m_buffer; ///< The 1-dimensional array used to store the tensor
        shape_type m_shape;      ///< A static array of size D that stores the shape of the tensor
        shape_type m_stride;     ///< A D-dimensional static array storing
        size_type m_totsize;     ///< The total size of the array
        size_type m_totcapacity; ///< The total capacity of the array

    public:
        tensor_base() : m_buffer(nullptr), m_shape{0}, m_stride{0}, m_totsize{0}, m_totcapacity{0} {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        //                Constructors setting tensor size and optionally values                 //
        ///////////////////////////////////////////////////////////////////////////////////////////
        template <typename... Args>
        tensor_base(size_type i, Args &&...args) : tensor_base() { CALL_AND_HANDLE(init<0>(i, std::forward<Args>(args)...), "Failed to construct tensor object.  Array buffer initialisation failed."); }

        tensor_base(const shape_type &_size) : m_buffer(nullptr), m_shape(_size), m_stride{}, m_totsize{1}, m_totcapacity{0}
        {
            for (size_type i = 0; i < rank; ++i)
            {
                m_totsize *= _size[i];
            }
            m_totcapacity = m_totsize;
            init_stride();
            CALL_AND_HANDLE(m_buffer = allocator::allocate(m_totsize), "Failed to construct tensor object.  Buffer allocation failed.");
        }

        template <typename int_type>
        tensor_base(const strict_array<int_type, rank> &_size) : m_buffer(nullptr), m_shape(_size), m_stride{}, m_totsize{1}, m_totcapacity{0}
        {
            for (size_type i = 0; i < rank; ++i)
            {
                m_totsize *= _size[i];
            }
            m_totcapacity = m_totsize;
            init_stride();
            CALL_AND_HANDLE(m_buffer = allocator::allocate(m_totsize), "Failed to construct tensor object.  Buffer allocation failed.");
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        //                   Copy constructor from generic tensor base types                     //
        ///////////////////////////////////////////////////////////////////////////////////////////
        template <typename Container, typename = other_copy_constructable_type<Container, self_type>>
        tensor_base(const Container &src) : tensor_base() { CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy construct tensor object."); }
        tensor_base(const self_type &src) : tensor_base() { CALL_AND_HANDLE(copy_assign_impl(src), "Failed to copy construct tensor object."); }

        tensor_base(self_type &&src) noexcept : m_buffer{src.m_buffer}, m_shape(std::move(src.m_shape)), m_stride(std::move(src.m_stride)), m_totsize{src.m_totsize}, m_totcapacity{src.m_totcapacity}
        {
            src.m_buffer = nullptr;
            src.m_totsize = 0;
            src.m_totcapacity = 0;
            for (size_type i = 0; i < rank; ++i)
            {
                src.m_shape[i] = 0;
                src.m_stride[i] = 0;
            }
        }

        template <typename Container, typename = other_move_constructable_type<Container, self_type>>
        tensor_base(Container &&src) : tensor_base() { CALL_AND_HANDLE(move_assign_impl(std::forward<Container>(src)), "Failed to move construct tensor object from expression."); }
        ~tensor_base()
        {
            try
            {
                if (m_buffer != nullptr)
                {
                    allocator::deallocate(m_buffer);
                }
                m_buffer = nullptr;
            }
            catch (...)
            {
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        //            Copy assignment from valid value types.  This fills the buffer             //
        ///////////////////////////////////////////////////////////////////////////////////////////
        template <typename U>
        inline typename std::enable_if<std::is_convertible<U, device_value_type>::value, self_type &>::type operator=(const U &_val)
        {
            ASSERT(m_buffer != nullptr, "Unable to fill tensor object.  The buffer has not been allocated");
            CALL_AND_HANDLE(fill_impl(_val), "Failed to value assign each element of the array.  Failed to fill the buffer.");
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        //                   Copy assignment from generic tensor base types                      //
        ///////////////////////////////////////////////////////////////////////////////////////////
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

        // move assignment
        inline self_type &operator=(self_type &&src)
        {
            if (this != &src)
            {
                if (m_buffer != nullptr)
                {
                    CALL_AND_HANDLE(allocator::deallocate(m_buffer), "Move assignment operator failed.  Error when deallocating the current arrays buffer.");
                }
                m_buffer = src.m_buffer;
                src.m_buffer = nullptr;
                m_totsize = src.m_totsize;
                src.m_totsize = 0;
                m_totcapacity = src.m_totcapacity;
                src.m_totcapacity = 0;

                m_shape = std::move(src.m_shape);
                m_stride = std::move(src.m_stride);
                CALL_AND_HANDLE(std::fill_n(&src.m_shape[0], rank, 0), "Move assignment operations failed.  Error when zeroing input tensors size array.");
                CALL_AND_HANDLE(std::fill_n(&src.m_stride[0], rank, 0), "Move assignment operations failed.  Error when zeroing input tensors stride array.");
            }
            return *this;
        }

        template <typename Container>
        inline other_move_assignable_type<Container, self_type> operator=(Container &&src) { CALL_AND_RETHROW(return move_assign_impl(std::forward<Container>(src))); }

        template <typename Container>
        inline typename std::enable_if<is_buffer_copyable_dense<Container, self_type>::value, self_type &>::type set_buffer(const Container &src)
        {
            ASSERT(m_totsize == src.size(), "Failed to copy buffer from input container.  The two objects do not have the same size.");
            using srcbck = typename Container::backend_type;
            CALL_AND_HANDLE(memtransfer<srcbck>::copy(reinterpret_cast<const_device_pointer>(src.buffer()), src.size(), m_buffer), "Copy assignment operator failed.  Error when copying the buffer.");
            return *this;
        }

        template <typename U>
        inline self_type &set_buffer(const U *src, size_type size)
        {
            ASSERT(m_totsize == size, "Failed to copy buffer from input buffer.  The two objects do not have the same size.");
            CALL_AND_HANDLE(memtransfer<blas_backend>::copy(reinterpret_cast<const_device_pointer>(src), size, m_buffer), "Copy assignment operator failed.  Error when copying the buffer.");
            return *this;
        }

        template <typename U>
        inline self_type &set_buffer(const U *src, const std::array<size_type, rank> &size, const std::array<size_type, rank> &strides)
        {
            ASSERT(m_shape == size, "Failed to copy buffer from input buffer.  The two objects do not have the same shape.");
            bool strides_equal = true;
            for (size_t i = 0; i < rank; ++i)
            {
                if (strides[i] != m_stride[i])
                {
                    strides_equal = false;
                    break;
                }
            }
            if (strides_equal)
            {
                CALL_AND_HANDLE(memtransfer<blas_backend>::copy(reinterpret_cast<const_device_pointer>(src), m_totsize, m_buffer), "Copy assignment operator failed.  Error when copying the buffer.");
            }
            else
            {
                memtransfer<blas_backend>::template copy_noncontiguous<device_value_type, rank>(reinterpret_cast<const_device_pointer>(src), m_shape, strides, m_buffer, m_stride);
            }
            return *this;
        }

        inline self_type &swap(self_type &src) noexcept
        {
            using std::swap;
            if (this != &src)
            {
                swap(m_totsize, src.m_totsize);
                swap(m_totcapacity, src.m_totcapacity);
                swap(m_shape, src.m_shape);
                swap(m_stride, src.m_stride);
                device_pointer temp = src.m_buffer;
                src.m_buffer = m_buffer;
                m_buffer = temp;
            }
            return *this;
        }

        template <size_t DD>
        inline self_type &swap_buffer(tensor<value_type, DD, backend_type> &src)
        {
            CALL_AND_RETHROW(internal::tensor_buffer_swap::swap_buffers(*this, src));
            return *this;
        }

        // deallocate the memory allocated and zero the array
        inline void deallocate()
        {
            if (m_buffer != nullptr)
            {
                CALL_AND_HANDLE(allocator::deallocate(m_buffer), "Deallocate failed.  Error when deallocating buffer object.");
            }
            m_buffer = nullptr;
            m_totsize = 0;
            m_totcapacity = 0;
            CALL_AND_HANDLE(std::fill_n(&m_shape[0], rank, 0), "Deallocate failed.  Error when zeroing input tensors size array.");
            CALL_AND_HANDLE(std::fill_n(&m_stride[0], rank, 0), "Deallocate failed.  Error when zeroing input tensors size array.");
        }

        template <typename... Args>
        inline void allocate(Args &&...args)
        {
            static_assert(sizeof...(args) == rank || sizeof...(args) == 1, "Incorrect number of arguments passed to tensor allocator");
            ASSERT(m_buffer == nullptr, "Unable to allocate tensor object.  It has already been allocated.");
            m_totsize = 0;
            m_totcapacity = 1;
            CALL_AND_HANDLE(init_alloc<0>(args...), "Failed to allocate tensor object.  Error when initialising the buffer.");
        }

        inline void allocate_buffer(size_type capacity)
        {
            ASSERT(m_buffer == nullptr, "Unable to allocate tensor object.  It has already been allocated.");
            m_totcapacity = capacity;
            CALL_AND_HANDLE(m_buffer = allocator::allocate(capacity), "Allocate buffer failed.");
        }

        template <typename... Args>
        inline void reallocate(Args &&...args)
        {
            static_assert(sizeof...(args) == rank || sizeof...(args) == 1, "Incorrect number of arguments passed to tensor allocator");
            CALL_AND_HANDLE(deallocate(), "Failed to reallocate tensor object.  Error when deallocating previously allocated tensor object.");
            CALL_AND_HANDLE(allocate(std::forward<Args>(args)...), "Failed to reallocate tensor object.  Error when allocate new buffers.");
        }

        inline void clear() { CALL_AND_HANDLE(deallocate(), "Failed to clear tensor object."); }

        // reallocate will decrease the size of the array if it is not the correct size
        template <typename... Args>
        inline void resize(const Args &...args)
        {
            static_assert(sizeof...(args) == rank, "The reallocate function requires all sizes to be specified");

            // we only reallocate if the size of the system will be changed.
            size_type size_prod = expand_sizes(args...);
            expand_size_to_array(m_shape, args...);
            init_stride();

            if (m_totcapacity < size_prod)
            {
                if (m_buffer != nullptr)
                {
                    CALL_AND_HANDLE(allocator::deallocate(m_buffer), "Failed to reallocate buffer object.  Error when deallocating previous buffer.");
                }
                CALL_AND_HANDLE(m_buffer = allocator::allocate(size_prod), "Failed to reallocate tensor object.  Error when allocating the buffer.");
                m_totcapacity = size_prod;
            }
            m_totsize = size_prod;
        }

        inline void resize(const shape_type &_size)
        {
            if (m_shape != _size)
            {
                m_shape = _size;
                size_type mtotsize = 1;
                for (size_type i = 0; i < rank; ++i)
                {
                    mtotsize *= _size[i];
                }
                init_stride();
                if (m_totcapacity < mtotsize)
                {
                    if (m_buffer != nullptr)
                    {
                        CALL_AND_HANDLE(allocator::deallocate(m_buffer), "Failed to reallocate buffer object.  Error when deallocating previous buffer.");
                    }
                    CALL_AND_HANDLE(m_buffer = allocator::allocate(mtotsize), "Failed to reallocate tensor object.  Error when allocating the buffer.");
                    m_totcapacity = mtotsize;
                }
                m_totsize = mtotsize;
            }
        }

        // shrink the array so that its total capacity is the same as its total size.  This additionally should keep all values presently in the array
        inline void shrink_to_fit()
        {
            if (m_totsize < m_totcapacity)
            {
                pointer temp;
                CALL_AND_HANDLE(temp = allocator::allocate(m_totsize), "Failed to shrink the buffer object.  Error when allocating the new storage buffer.");
                CALL_AND_HANDLE(memtransfer<backend_type>::copy(m_buffer, m_totsize, temp), "Failed to shrink the buffer object.  Failed when copying the current buffer into the new storage buffer.");
                CALL_AND_HANDLE(allocator::deallocate(m_buffer), "Failed to shrink the buffer object.  Failed to deallocate the previous buffer object.");
                m_buffer = temp;
                temp = nullptr;
                m_totcapacity = m_totsize;
            }
        }

        // methods for accessing the size of the tensor object.
        inline const size_type &capacity() const { return m_totcapacity; }
        inline const size_type &max_size() const { return m_totcapacity; }
        inline const size_type &size() const { return m_totsize; }
        inline const size_type &nelems() const { return m_totsize; }
        inline const shape_type &shape() const { return m_shape; }
        inline const shape_type &stride() const { return m_stride; }

        inline const size_type &size(size_type i) const
        {
            ASSERT(i < rank, "Unable to access tensor size element.  Index out of bounds.");
            return m_shape[i];
        }
        inline const size_type &dims(size_type i) const
        {
            ASSERT(i < rank, "Unable to access tensor size element.  Index out of bounds.");
            return m_shape[i];
        }
        inline const size_type &shape(size_type i) const
        {
            ASSERT(i < rank, "Unable to access tensor size element.  Index out of bounds.");
            return m_shape[i];
        }
        inline const size_type &stride(size_type i) const
        {
            ASSERT(i < rank, "Unable to access tensor size element.  Index out of bounds.");
            return m_stride[i];
        }

        inline bool same_shape(const shape_type &_shape) const { return m_shape == _shape; }
#ifdef CEREAL_LIBRARY_FOUND
    public:
        using buffer_reader_type = internal::buffer_reader_wrapper<device_value_type, backend_type>;
        using buffer_writer_type = internal::buffer_writer_wrapper<device_value_type, backend_type>;

        template <typename archive>
        void save(archive &ar) const
        {
            auto bwriter = buffer_writer_type{m_buffer, m_totsize, m_totcapacity};
            CALL_AND_HANDLE(ar(cereal::make_nvp("buffer", bwriter)), "Failed to serialise tensor object.  Error when serialising the data buffer.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("shape", m_shape)), "Failed to serialise tensor object.  Failed to serialise the tensor shape.");
        }

        template <typename archive>
        void load(archive &ar)
        {
            size_type totsize;
            auto breader = buffer_reader_type{&m_buffer, &totsize, &m_totcapacity};
            CALL_AND_HANDLE(ar(cereal::make_nvp("buffer", breader)), "Failed to deserialise tensor object.  Error when deserialising the data buffer.");
            CALL_AND_HANDLE(ar(cereal::make_nvp("shape", m_shape)), "Failed to deserialise tensor object.  Failed to deserialise the tensor shape.");
            m_totsize = 1;
            for (size_type i = 0; i < rank; ++i)
            {
                m_totsize *= m_shape[i];
            }
            ASSERT(m_totsize <= m_totcapacity, "Failed to deserialise tensor object. The total size of the tensor is incompatible with the storage buffer size.");
            ASSERT(m_totsize == totsize, "Failed to deserialise tensor object. The total size of the tensor is incompatible with the shape.");
            init_stride();
        }
#endif

    public:
        template <typename Itype, typename... Args>
        inline typename std::enable_if<std::is_integral<Itype>::value, reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args) + 1, backend_type>>::type reinterpret_shape(const Itype &i, Args &&...args) const
        {
            using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args) + 1, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(m_totsize, true, m_buffer, i, std::forward<Args>(args)...), "Failed to reinterpret the shape of the tensor object.");
        }

        template <typename Itype, typename... Args>
        inline typename std::enable_if<std::is_integral<Itype>::value, reinterpreted_tensor<value_type, sizeof...(Args) + 1, backend_type>>::type reinterpret_shape(const Itype &i, Args &&...args)
        {
            using reinterpreted_type = reinterpreted_tensor<value_type, sizeof...(Args) + 1, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(m_totsize, true, m_buffer, i, std::forward<Args>(args)...), "Failed to reinterpret the shape of the tensor object.");
        }

        template <size_t vD>
        inline reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type> reinterpret_shape(const std::array<size_type, vD> &size) const
        {
            using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(m_totsize, true, m_buffer, size), "Failed to reinterpret the shape of the tensor object.");
        }

        template <size_t vD>
        inline reinterpreted_tensor<value_type, vD, backend_type> reinterpret_shape(const std::array<size_type, vD> &size)
        {
            using reinterpreted_type = reinterpreted_tensor<value_type, vD, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(m_totsize, true, m_buffer, size), "Failed to reinterpret the shape of the tensor object.");
        }

        // functions for reinterpret the capacity of the tensor object
        template <typename Itype, typename... Args>
        inline typename std::enable_if<std::is_integral<Itype>::value, reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args) + 1, backend_type>>::type reinterpret_capacity(const Itype &i, Args &&...args) const
        {
            using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, sizeof...(Args) + 1, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(m_totcapacity, false, m_buffer, i, std::forward<Args>(args)...), "Failed to reinterpret the capacity of the tensor object.");
        }

        template <typename Itype, typename... Args>
        inline typename std::enable_if<std::is_integral<Itype>::value, reinterpreted_tensor<value_type, sizeof...(Args) + 1, backend_type>>::type reinterpret_capacity(const Itype &i, Args &&...args)
        {
            using reinterpreted_type = reinterpreted_tensor<value_type, sizeof...(Args) + 1, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(m_totcapacity, false, m_buffer, i, std::forward<Args>(args)...), "Failed to reinterpret the capacity of the tensor object.");
        }

        template <size_t vD>
        inline reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type> reinterpret_capacity(const std::array<size_type, vD> &capacity) const
        {
            using reinterpreted_type = reinterpreted_tensor<typename std::add_const<value_type>::type, vD, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(m_totcapacity, false, m_buffer, capacity), "Failed to reinterpret the capacity of the tensor object.");
        }

        template <size_t vD>
        inline reinterpreted_tensor<value_type, vD, backend_type> reinterpret_capacity(const std::array<size_type, vD> &capacity)
        {
            using reinterpreted_type = reinterpreted_tensor<value_type, vD, backend_type>;
            CALL_AND_HANDLE(return reinterpreted_type(m_totcapacity, false, m_buffer, capacity), "Failed to reinterpret the capacity of the tensor object.");
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
            CALL_AND_HANDLE(backend_algebra<backend_type>::scal(m_totsize, device_value_type(v), m_buffer, 1), "Failed to perform operator*= on tensor object.  scal call failed.");
            return *this;
        }
        template <typename Vt>
        inline value_update_type<Vt, self_type> operator/=(const Vt &v)
        {
            CALL_AND_HANDLE(backend_algebra<backend_type>::scal(m_totsize, device_value_type(1.0 / v), m_buffer, 1), "Failed to perform operator/= on tensor object.  scal call failed.");
            return *this;
        }

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

    public:
        inline device_pointer buffer() { return m_buffer; }
        inline const_device_pointer buffer() const { return m_buffer; }
        inline device_pointer data() { return m_buffer; }
        inline const_device_pointer data() const { return m_buffer; }

    protected:
        /** These two functions can and should probably be changed to ensure no additional runtime overhead*/
        // initialise the remaining dimensions of the array if required.
        template <size_type d>
        inline void init_size()
        {
            m_totsize = 1;
            if (d != rank)
            {
                for (size_type i = d; i < rank; ++i)
                {
                    m_shape[i] = m_shape[d - 1];
                }
            }
            for (size_type i = 0; i < rank; ++i)
            {
                m_totsize *= m_shape[i];
            }
        }

        // initialize the stride array storing how far we need to skip through the array
        inline void init_stride()
        {
            m_stride[rank - 1] = 1;
            for (size_type i = 1; i < rank; ++i)
            {
                m_stride[rank - i - 1] = m_shape[rank - i] * m_stride[rank - i];
            }
        }

    private:
        // functions for copy assigning object
        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container &src)
        {
            if (m_totcapacity < src.size())
            {
                if (m_buffer != nullptr)
                {
                    CALL_AND_HANDLE(allocator::deallocate(m_buffer), "Copy assignment failed.  Error when deallocating the buffer.");
                }
                CALL_AND_HANDLE(m_buffer = allocator::allocate(src.size()), "Copy assignment failed.  Error when allocating the buffer.")
                m_totcapacity = src.size();
            }
            using srcbck = typename traits<Container>::backend_type;
            CALL_AND_HANDLE(memtransfer<srcbck>::copy(reinterpret_cast<const_device_pointer>(src.buffer()), src.size(), m_buffer), "Copy operator failed.  Error when copying the buffer.");
            m_shape = src.shape();
            m_totsize = src.size();
            init_stride();
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container &src)
        {
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate copy_assign_impl.");
            if (m_totcapacity < src.size())
            {
                if (m_buffer != nullptr)
                {
                    CALL_AND_HANDLE(allocator::deallocate(m_buffer), "Copy assignment failed.  Error when deallocating the buffer.");
                }
                CALL_AND_HANDLE(m_buffer = allocator::allocate(src.size()), "Copy assignment failed.  Error when allocating the buffer.")
                m_totcapacity = src.size();
            }
            CALL_AND_HANDLE(backend_algebra<backend_type>::copy_real_to_complex(src.buffer(), src.size(), m_buffer), "Copy operator failed.  Error when copying the buffer.");
            m_shape = src.shape();
            m_totsize = src.size();
            init_stride();
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type copy_assign_impl(const Container &expr)
        {
            bool requires_resize;
            CALL_AND_HANDLE(requires_resize = !(expr.shape() == m_shape), "Copy assignment failed. Failed to determine whether the tensor requires resizing before evaluation of the input expression.");
            if (requires_resize)
            {
                CALL_AND_HANDLE(resize(expr.shape()), "Copy assignment failed.  Failed to resize the tensor object.");
            }
            CALL_AND_HANDLE(expr(*this), "Copy assignment failed.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type move_assign_impl(Container &&expr)
        {
            bool requires_resize;
            CALL_AND_HANDLE(requires_resize = !(expr.shape() == m_shape), "Copy assignment failed. Failed to determine whether the tensor requires resizing before evaluation of the input expression.");
            if (requires_resize)
            {
                CALL_AND_HANDLE(resize(expr.shape()), "Copy assignment failed.  Failed to resize the tensor object.");
            }
            CALL_AND_HANDLE(expr(*this), "Copy assignment failed.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

    private:
        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type
        addition_assign_impl(const Container &src)
        {
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
            ASSERT(m_shape == src.shape(), "Addition assignment failed.  The two objects are not the same size");
            CALL_AND_HANDLE(backend_algebra<backend_type>::addition_assign(reinterpret_cast<const_device_pointer>(src.buffer()), src.size(), m_buffer), "Addition assignment operator failed.  Error when copying the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type
        addition_assign_impl(const Container &src)
        {
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
            ASSERT(m_shape == src.shape(), "Addition assignment failed.  The two objects are not the same size");
            CALL_AND_HANDLE(backend_algebra<backend_type>::addition_assign_real_to_complex(src.buffer(), src.size(), m_buffer), "Addition assignment operator failed.  Error when copying the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type
        addition_assign_impl(const Container &expr)
        {
            ASSERT(m_shape == expr.shape(), "Addition assignment failed.  The two objects are not the same size");
            CALL_AND_HANDLE(expr.add_assignment(*this), "Addition assignment failed.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type addition_assign_impl(Container &&expr)
        {
            ASSERT(m_shape == expr.shape(), "Addition assignment failed.  The two objects are not the same size");
            CALL_AND_HANDLE(expr.add_assignment(*this), "Addition assignment failed.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

    private:
        template <typename Container>
        inline typename std::enable_if<!is_expression<Container>::value && is_tensor<Container>::value && is_same_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type
        subtraction_assign_impl(const Container &src)
        {
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate addition_assign_impl.");
            ASSERT(m_shape == src.shape(), "Addition assignment failed.  The two objects are not the same size");
            CALL_AND_HANDLE(backend_algebra<backend_type>::subtraction_assign(reinterpret_cast<const_device_pointer>(src.buffer()), src.size(), m_buffer), "Addition assignment operator failed.  Error when copying the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_tensor<Container>::value && is_real_to_complex_value<Container, self_type>::value, copy_assignable_type<Container, self_type>>::type
        subtraction_assign_impl(const Container &src)
        {
            static_assert(is_same_backend<Container, self_type>::value, "Unable to instantiate subtraction_assign_impl.");
            ASSERT(m_shape == src.shape(), "Subtraction assignment failed.  The two objects are not the same size");
            CALL_AND_HANDLE(backend_algebra<backend_type>::subtraction_assign_real_to_complex(src.buffer(), src.size(), m_buffer), "Subtraction assignment operator failed.  Error when copying the buffer.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, copy_assignable_type<Container, self_type>>::type
        subtraction_assign_impl(const Container &expr)
        {
            ASSERT(m_shape == expr.shape(), "Subtraction assignment failed.  The two objects are not the same size");
            CALL_AND_HANDLE(expr.subtract_assignment(*this), "Subtraction assignment failed.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

        template <typename Container>
        inline typename std::enable_if<is_expression<Container>::value, move_assignable_type<Container, self_type>>::type subtraction_assign_impl(Container &&expr)
        {
            ASSERT(m_shape == expr.shape(), "Subtraction assignment failed.  The two objects are not the same size");
            CALL_AND_HANDLE(expr.subtract_assignment(*this), "Subtraction assignment failed.  Failed to evaluate the expression into the tensor object.");
            return *this;
        }

    private:
        // Function for initialising the array
        template <size_type d, typename U, typename... Args>
                typename std::enable_if < std::is_integral<U>::value and
            !std::is_pointer<U>::value and d<rank, void>::type init(U i, Args &&...args)
        {
            m_shape[d] = i;
            CALL_AND_RETHROW(init<d + 1>(std::forward<Args>(args)...));
        }

        // if we get a value_type in the initialiser we set the value to contain buffer size.
        template <size_type d>
        void init()
        {
            init_size<d>();
            m_totcapacity = m_totsize;
            init_stride();
            CALL_AND_HANDLE(m_buffer = allocator::allocate(m_totcapacity), "Failed to initialise tensor object.  Error when allocating the buffer.");
        }

        // if we get a value_type in the initialiser we set the value to contain buffer size.
        template <size_type d>
        typename std::enable_if<!std::is_integral<value_type>::value || d == rank, void>::type init(const value_type &v)
        {
            init_size<d>();
            m_totcapacity = m_totsize;
            init_stride();
            CALL_AND_HANDLE(m_buffer = allocator::allocate(m_totcapacity), "Failed to initialise tensor object.  Error when allocating the buffer.");
            CALL_AND_HANDLE(memfill::fill(m_buffer, m_totsize, device_value_type(v)), "Failed to initialise tensor object.  Error when filling buffer.");
        }

        // initialise the array using a functor that defines the elements.
        template <int d, typename U, typename... Args>
        typename std::enable_if<!std::is_pointer<U>::value and !std::is_integral<U>::value and !std::is_convertible<U, value_type>::value, void>::type init(U t, Args &&...args)
        {
            init_size<d>();
            m_totcapacity = m_totsize;
            init_stride();
            CALL_AND_HANDLE(m_buffer = allocator::allocate(m_totcapacity), "Failed to initialise tensor object.  Error when allocating the buffer.");
            CALL_AND_HANDLE(detail_type::fill(std::forward<U>(t), std::forward<Args>(args)...), "Failed to initialise tensor object.  Error when filling elements from a functor.");
        }

        template <size_type d, typename U, typename... Args>
                typename std::enable_if < std::is_integral<U>::value and
            !std::is_pointer<U>::value and d<rank, void>::type init_alloc(U i, Args &&...args)
        {
            m_totcapacity *= i;
            CALL_AND_RETHROW(init_alloc<d + 1>(std::forward<Args>(args)...))
        }

        template <size_type d>
        void init_alloc() { CALL_AND_HANDLE(m_buffer = allocator::allocate(m_totcapacity), "Failed to initialise tensor object.  Error when allocating the buffer."); }

        ///////////////////////////////////////////////////////////////////////////////////////////////////
        //   Functions for determining whether we need to reallocate the array to get the correct size   //
        ///////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename A, typename... Args>
        size_type expand_sizes(const A &a, Args &&...args) { return a * expand_sizes(std::forward<Args>(args)...); }
        size_type expand_sizes() { return 1; }
        template <typename A, typename... Args>
        void expand_size_to_array(shape_type &_size, const A &a, Args &&...args)
        {
            _size[rank - (sizeof...(args) + 1)] = a;
            expand_size_to_array(_size, std::forward<Args>(args)...);
        }
        template <typename A>
        void expand_size_to_array(shape_type &_size, const A &a) { _size[rank - 1] = a; }

        // initialise the array from a pointer. (I might hide this later)
        template <int d, typename U, typename... Args>
        typename std::enable_if<std::is_pointer<U>::value, void>::type requires_reallocate(bool &_reallocate, U t)
        {
            m_buffer = t;
            _reallocate = false;
        }

        // initialise the array using a functor that defines the elements.
        template <int d, typename U, typename... Args>
        typename std::enable_if<!std::is_pointer<U>::value and !std::is_integral<U>::value, void>::type requires_reallocate(bool &_reallocate, U &&t, Args &&...args)
        {
            CALL_AND_HANDLE(detail_type::fill(std::forward<U>(t), std::forward<Args>(args)...), "Failed to reallocate tensor object.  Error when filling elements from a functor.");
            _reallocate = false;
        }
    }; // class tensor


    ///////////////////////////////////////////////////////////////////////////////////////
    // D dimensional interfaces of the general tensor object for use with the general    //
    //                                     backend                                       //
    ///////////////////////////////////////////////////////////////////////////////////////
    template <typename T, size_t D, typename backend>
    class tensor : public tensor_base<tensor<T, D, backend>>
    {
    public:
        using self_type = tensor<T, D, backend>;
        using value_type = typename traits<self_type>::value_type;
        using device_value_type = typename traits<self_type>::device_value_type;
        using size_type = typename traits<self_type>::size_type;
        using base_type = tensor_base<self_type>;
        using const_slice_traits = tensor_slice_traits<self_type, const T, D>;
        using slice_traits = tensor_slice_traits<self_type, T, D>;

        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using reference = T&;
        using const_reference = const value_type&;

    public:
        template <typename... Args>
        tensor(Args &&...args);

        template <typename... Args>
        self_type &operator=(Args &&...args);
        template <typename... Inds> const_reference operator()(Inds... indices) const;
        template <typename... Inds> reference operator()(Inds... indices);
        template <typename... Inds> const_reference at(Inds... indices) const;
        template <typename... Inds> reference at(Inds... indices);

        // slice accessor operator[]
        typename slice_traits::slice_type operator[](size_type i);
        typename const_slice_traits::slice_type operator[](size_type i) const;

        typename slice_traits::slice_type slice(size_type i);
        typename const_slice_traits::slice_type slice(size_type i) const;
    }; // class tensor

    ///////////////////////////////////////////////////////////////////////////////////////
    // 1 dimensional interface of the general tensor object for use with the general     //
    //                                     backend                                       //
    ///////////////////////////////////////////////////////////////////////////////////////
    template <typename T, typename backend>
    class tensor<T, 1, backend> : public tensor_base<tensor<T, 1, backend>>
    {
    public:
        using self_type = tensor<T, 1, backend>;
        using size_type = typename traits<backend>::size_type;
        using base_type = tensor_base<self_type>;
        using reference = T&;
        using const_reference = const T&;

        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        friend class internal::tensor_buffer_swap;

    public:
        template <typename... Args>
        tensor(Args &&...args);
        template <typename... Args>
        self_type &operator=(Args &&...args);
    
        void from_host() const;
        template <typename... Inds> inline const_reference operator()(Inds... indices) const;
        template <typename... Inds> inline reference operator()(Inds... indices);
        template <typename... Inds> inline const_reference at(Inds... indices) const;
        template <typename... Inds> inline reference at(Inds... indices);        
        inline const_reference operator[](size_t i) const;
        inline reference operator[](size_t i);
    };

    ///////////////////////////////////////////////////////
    //            ostream operators interface            //
    ///////////////////////////////////////////////////////
    //template <typename T, size_t D, typename backend>
    //std::ostream& operator<<(std::ostream &os, const tensor<T,D,backend> &t);
}

#endif // PYTTN_LINALG_TENSOR_DENSE_TENSOR_HPP_//
