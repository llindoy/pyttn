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

#ifndef PYTTN_BACKENDS_CUDA_HOST_ACCESS_CUH_
#define PYTTN_BACKENDS_CUDA_HOST_ACCESS_CUH_

#include "../../utils/memory_helper.cuh"

// TODO: Implement stl allocators (and potentially an aligned allocator) to handle memory rather than the hacky approach I have currently taken.
namespace linalg
{
    template <typename T>
    struct host_access
    {
    protected:
        using host_allocator = memory::allocator<T, blas_backend>;
        using host_memtransfer = memory::transfer<cuda_backend, blas_backend>;

        using host_pointer = T*;
        using device_value_type = typename device_type<T, cuda_backend>::type;
        using device_pointer = device_value_type*;

        mutable T* m_host_buffer;
        mutable size_t m_host_buffer_size;
        mutable bool m_copied_from_host;

    protected:
        void allocate_host(size_t dim) const
        {
            if(m_host_buffer == nullptr)
            {
                CALL_AND_HANDLE(m_host_buffer = host_allocator::allocate(dim), "Failed to copy from host.  Buffer allocation failed.");
                m_host_buffer_size = dim;
            }
        }

    public:
        host_access() : m_host_buffer(nullptr), m_host_buffer_size(0), m_copied_from_host(false){}
        ~host_access(){clear_host();}

        void clear_host() const
        {
            if (m_host_buffer != nullptr)
            {
                host_allocator::deallocate(m_host_buffer);
            }
            m_host_buffer = nullptr;  
            m_host_buffer_size = 0;
        }

        void _from_host(device_pointer buffer, size_t totsize) const
        {
            if(m_host_buffer == nullptr)
            {
                allocate_host(totsize);
            }
            else
            {
                if(m_host_buffer_size < totsize)
                {
                    clear_host();
                    allocate_host(totsize);
                }
            }

            CALL_AND_HANDLE(host_memtransfer::copy(buffer, totsize, m_host_buffer), "Retrieving data from host failed.  Error when copying the buffer.");
            m_copied_from_host=true;
        }
    };
}

#endif // PYTTN_BACKENDS_CUDA_HOST_ACCESS_CUH_//
