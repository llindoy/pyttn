/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2026 NPL Management Limited
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

#include <cstdint>
#include <cstddef>
#include "allocator.hpp"
#include <common/exception_handling.hpp>

#include <linalg/backends/cuda/cuda_backend.hpp>
#include <linalg/backends/cuda/cuda_backend.cuh>
#include <linalg/backends/cuda/cuda_utils.cuh>
#include <linalg/domain_manager/cuda/allocator.hpp>


namespace linalg
{
    namespace memory
    {

        //standard Cuda Device allocator
        CudaDeviceAllocator::CudaDeviceAllocator(const ExecDomain<cuda_backend>& domain)
        {
            CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(this->m_domain->gpu_id())), "Failed to set cuda device id.");

            void* res;
            CALL_AND_HANDLE(cuda_safe_call(cudaMalloc(&res, n)), "Failed to allocate memory buffer.");
            return res;
        }

        void* CudaDeviceAllocator::allocate_bytes(std::size_t n, std::size_t /*alignment*/)
        {
            CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(this->m_domain->gpu_id())), "Failed to set cuda device id.");

            void* res;
            CALL_AND_HANDLE(cuda_safe_call(cudaMalloc(&res, n)), "Failed to allocate memory buffer.");
            return res;
        }

        void CudaDeviceAllocator::deallocate_bytes(void* ptr, std::size_t /*alignment*/)
        {
            CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(this->m_domain->gpu_id())), "Failed to set cuda device id.");

            if (ptr != nullptr)
            {
                CALL_AND_HANDLE(cuda_safe_call(cudaFree(ptr)), "Failed to deallocate memory buffer.");
            }
        }


        //Pinned Host memory allocator
        PinnedHostAllocator::PinnedHostAllocator(const ExecDomain<blas_backend>& domain) : Allocator<blas_backend>(domain){}

        void* PinnedHostAllocator::allocate_bytes(std::size_t n, std::size_t /*alignment*/)
        {
            void* res;
            CALL_AND_HANDLE(cuda_safe_call(cudaMallocHost(&res, n)), "Failed to allocate memory buffer.");
            return res;
        }

        void PinnedHostAllocator::deallocate_bytes(void* ptr, std::size_t /*alignment*/) override
        {
            if (ptr != nullptr)
            {
                CALL_AND_HANDLE(cuda_safe_call(cudaFreeHost(ptr)), "Failed to deallocate memory buffer.");
            }
        }

        //Rotating Cuda Buffer 
        static std::size_t CudaCircularBufferAllocator::align_up(std::size_t value, std::size_t alignment)
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        explicit CudaCircularBufferAllocator::CudaCircularBufferAllocator(const ExecDomain<cuda_backend>& domain, std::size_t capacity_bytes) : Allocator<cuda_backend>(domain), m_capacity(capacity_bytes), m_offset(0), m_buffer(nullptr)
        {
            CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(this->m_domain->gpu_id())), "Failed to set cuda device id.");
            CALL_AND_HANDLE(cuda_safe_call(cudaMalloc(&m_buffer, capacity_bytes)), "Failed to allocate memory buffer.");
        }

        CudaCircularBufferAllocator::~CudaCircularBufferAllocator()
        {
            if(m_buffer)
            {
                CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(this->m_domain->gpu_id())), "Failed to set cuda device id.");
                CALL_AND_HANDLE(cuda_safe_call(cudaFree(m_buffer)), "Failed to allocate memory buffer.");
            }
            m_buffer = nullptr;
        }

        void* CudaCircularBufferAllocator::allocate_bytes(std::size_t nbytes, std::size_t alignment) override
        {
            std::size_t aligned_offset = align_up(m_offset, alignment);
            // If the aligned offset plus size is larger than the capacity we will wrap 
            if (aligned_offset + nbytes > m_capacity) 
            {
                aligned_offset = 0;
                // If after wrapping it doesn't fit then we cannot allocate the object using this buffer and we will exit. 
                if (nbytes > m_bytes) {throw std::bad_alloc();
                }
            }
            //make it so that the alignment is adding bytes
            void* ptr = static_cast<char*>(m_buffer) + aligned_offset;
            m_offset = aligned_offset + nbytes;
            return ptr;
        }

        void CudaCircularBufferAllocator::deallocate_bytes(void* /*ptr*/, std::size_t /*alignment*/) override
        {
            //deallocate does nothing
        }

        void CudaCircularBufferAllocator::reset()
        {
            m_offset = 0;
        }

        void CudaCircularBufferAllocator::expand(std::size_t new_capacity) 
        {
            if(new_capacity < m_capacity)
            {
                return;
            }
            CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(this->m_domain->gpu_id())), "Failed to set cuda device id.");
            CALL_AND_HANDLE(cuda_safe_call(cudaFree(m_buffer)), "Failed to free existing buffer.");
            CALL_AND_HANDLE(cuda_safe_call(cudaMalloc(&m_buffer, new_capacity)), "Failed to allocate new buffer.");
            m_capacity = new_capacity;
            m_offset = 0;
        }

        void CudaCircularBufferAllocator::expand_to_fit(std::size_t required_bytes) 
        {
            if (required_bytes > m_capacity) {
                expand(std::max(required_bytes, 2 * m_capacity));
            }
        }
    }
}

