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

#ifndef PYTTN_LINALG_TENSOR_DENSE_TENSOR_CUH_
#define PYTTN_LINALG_TENSOR_DENSE_TENSOR_CUH_

#include "../../../linalg_forward_decl.hpp"
#include "../tensor.hpp"
#include "../../../backends/cuda/cuda_backend.hpp"
#include "../../../backends/cuda/host_access.cuh"

// TODO: implement cuda storing the cuda device types internally but exposing the host type externally.  
//       this is likely easiest done by adding a value_type and internal_type for each container type
//       and letting the backend set these.  A key aspect of this is that the unary expressions should
//       probably work with the internal_type as they need to be applicable to devices code.  
// TODO: Implement stl allocators (and potentially an aligned allocator) to handle memory rather than the hacky approach I have currently taken.
namespace linalg
{
    ///////////////////////////////////////////////////////////////////////////////////////
    // D dimensional implementation of the general tensor object for use with the cuda   //
    //                                     backend                                       //
    ///////////////////////////////////////////////////////////////////////////////////////
    template <typename T, size_t D>
    class tensor<T, D, cuda_backend> : public tensor_base<tensor<T, D, cuda_backend>>, public host_access<T>
    {
    public:
        using self_type = tensor<T, D, cuda_backend>;
        using value_type = typename traits<self_type>::value_type;
        using size_type = typename traits<self_type>::size_type;
        using base_type = tensor_base<self_type>;
        using const_slice_traits = tensor_slice_traits<self_type, const T, D>;
        using slice_traits = tensor_slice_traits<self_type, T, D>;

        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using const_reference = const value_type&;
        friend class internal::tensor_buffer_swap;

    protected:
        using base_type::m_buffer;
        using base_type::m_shape;
        using base_type::m_stride;
        using base_type::m_totsize;


    public:
        template <typename... Args>
        tensor(Args &&...args)
        try : base_type(std::forward<Args>(args)...){}
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

        void from_host() const
        {
            this->_from_host(this->m_buffer, this->m_totsize);
        }

        template <typename... Inds>
        inline const_reference operator()(Inds... indices) const
        {
            if(!this->m_copied_from_host){this->from_host();}
            static_assert(sizeof...(Inds) == D, "Failed to access element of tensor object.  The input index list does not have the correct size.");
            using pack_type = typename internal::check_integral<Inds...>::pack_type;
            return this->m_host_buffer[get_index<pack_type>(indices...)];
        }

        // slice accessor operator[]
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

    public:
        pointer buffer() { return m_buffer; }
        const_pointer buffer() const { return m_buffer; }
        pointer data() { return m_buffer; }
        const_pointer data() const { return m_buffer; }

    }; // class tensor

    template <typename T>
    class tensor<T, 1, cuda_backend> : public tensor_base<tensor<T, 1, cuda_backend>>, public host_access<T>
    {
    public:
        using self_type = tensor<T, 1, cuda_backend>;
        using size_type = typename traits<cuda_backend>::size_type;
        using base_type = tensor_base<self_type>;
        using const_reference = const T&;

        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
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
    
        void from_host() const
        {
            this->_from_host(this->m_buffer, this->m_totsize);
        }

        inline const_reference operator()(size_t i) const
        {
            if(!this->m_copied_from_host){this->from_host();}
            return this->m_host_buffer[i];
        }

        inline const_reference operator[](size_t i) const
        {
            if(!this->m_copied_from_host){this->from_host();}
            return this->m_host_buffer[i];
        }

    public:
        pointer buffer() { return m_buffer; }
        const_pointer buffer() const { return m_buffer; }
        pointer data() { return m_buffer; }
        const_pointer data() const { return m_buffer; }
    };
} // namespace linalg

#include "tensor_slice.cuh"

namespace linalg
{
    template <typename array_type, typename = typename std::enable_if<is_dense_tensor<array_type>::value && std::is_same<cuda_backend, typename traits<array_type>::backend_type>::value, void>::type>
    std::ostream &operator<<(std::ostream &os, const array_type &t)
    {
        tensor<typename array_type::value_type, traits<array_type>::rank, blas_backend> _t(t);
        os << _t;
        return os;
        // os << "shape: ["; for(size_t i=0; i < t.rank; ++i){os << t.shape(i) << (i+1 == t.rank ? "]": ", ");}
        // os << "cuda buffer" << std::endl;
        // return os;
    }
}

#endif // PYTTN_LINALG_TENSOR_DENSE_TENSOR_CUH_//
