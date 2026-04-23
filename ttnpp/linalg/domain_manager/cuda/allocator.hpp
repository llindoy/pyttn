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

#ifndef PYTTN_LINALG_UTILS_ALLOCATOR_CUH_
#define PYTTN_LINALG_UTILS_ALLOCATOR_CUH_

#include <cstdint>
#include <cstddef>
#include "allocator.hpp"
#include <common/exception_handling.hpp>
#include "../../backends/cuda/cuda_backend.hpp"
#include "execDomain.hpp"
#include "../allocator.hpp"


namespace linalg
{
    namespace memory
    {
        //standard Cuda Device allocator
        class CudaDeviceAllocator : public Allocator<cuda_backend>
        {
        public:
            explicit CudaDeviceAllocator(const ExecDomain<cuda_backend>& domain);
            void* allocate_bytes(std::size_t n, std::size_t /*alignment*/) override;
            void deallocate_bytes(void* ptr, std::size_t /*alignment*/) override;
        };

        //Pinned Host memory allocator
        class PinnedHostAllocator : public Allocator<blas_backend>
        {
        public:
            explicit PinnedHostAllocator(const ExecDomain<blas_backend>& domain);
            void* allocate_bytes(std::size_t n, std::size_t /*alignment*/) override;
            void deallocate_bytes(void* ptr, std::size_t /*alignment*/) override;
        };

        //Rotating Cuda Buffer 
        //standard Cuda Device allocator
        class CudaCircularBufferAllocator : public Allocator<cuda_backend>
        {
        protected:
            std::size_t m_capacity
            std::size_t m_offset;
            void * m_buffer;

        protected:
            static std::size_t align_up(std::size_t value, std::size_t alignment);
        public:
            explicit CudaCircularBufferAllocator(const ExecDomain<cuda_backend>& domain, std::size_t capacity_bytes) ;
            ~CudaCircularBufferAllocator();

            void* allocate_bytes(std::size_t nbytes, std::size_t alignment) override;
            void deallocate_bytes(void* /*ptr*/, std::size_t /*alignment*/) override;
            void reset();
            void expand(std::size_t new_capacity);
            void expand_to_fit(std::size_t required_bytes);
        };
    }
}

#endif //PYTTN_LINALG_UTILS_ALLOCATOR_CUH_
