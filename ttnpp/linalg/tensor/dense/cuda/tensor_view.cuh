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

#ifndef PYTTN_LINALG_TENSOR_DENSE_TENSOR_VIEW_CUH_
#define PYTTN_LINALG_TENSOR_DENSE_TENSOR_VIEW_CUH_

#include "../../../linalg_forward_decl.hpp"
#include "../tensor_view.hpp"
#include "../../../backends/cuda/cuda_backend.hpp"
#include "../../../backends/cuda/host_access.cuh"

namespace linalg
{
    //TODO Add host access

    ///////////////////////////////////////////////////////////////////////////////////////
    // D dimensional implementation of the tensor view object for use with the  //
    //                                    cuda backend                                   //
    ///////////////////////////////////////////////////////////////////////////////////////
    template <typename T, size_t D>
    class tensor_view<T, D, cuda_backend> : public tensor_view_base<tensor_view<T, D, cuda_backend>>
    {
    public:
        using self_type = tensor_view<T, D, cuda_backend>;
        using value_type = T;
        using base_type = tensor_view_base<self_type>;
        using size_type = typename base_type::size_type;
        using const_slice_traits = tensor_slice_traits<self_type, typename std::add_const<T>::type, D>;
        using slice_traits = tensor_slice_traits<self_type, T, D>;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;

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

        pointer buffer() { return m_buffer; }
        const_pointer buffer() const { return m_buffer; }
        pointer data() { return m_buffer; }
        const_pointer data() const { return m_buffer; }
    };

    ///////////////////////////////////////////////////////////////////////////////////////
    //       1 dimensional implementation of the tensor view object for use with the     //
    //                                    cuda backend                                   //
    ///////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    class tensor_view<T, 1, cuda_backend> : public tensor_view_base<tensor_view<T, 1, cuda_backend>>
    {
    public:
        using self_type = tensor_view<T, 1, cuda_backend>;
        using base_type = tensor_view_base<self_type>;
        using size_type = typename base_type::size_type;
        using value_type = T;

        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;

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

        __host__ __device__ pointer buffer() { return m_buffer; }
        __host__ __device__ const_pointer buffer() const { return m_buffer; }
        __host__ __device__ pointer data() { return m_buffer; }
        __host__ __device__ const_pointer data() const { return m_buffer; }
    };
} // namespace linalg

#endif // PYTTN_LINALG_TENSOR_DENSE_TENSOR_VIEW_CUH_
