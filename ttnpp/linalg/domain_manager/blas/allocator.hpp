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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_BLAS_ALLOCATOR_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_BLAS_ALLOCATOR_HPP_

#include <cstdint>
#include <cstddef>


#if __cplusplus >= 201703L
#include <new>
#else
#if defined(_MSC_VER)
#include <malloc.h>
#else
#include <cstdlib>
#endif
#endif

#include "execProps.hpp"
#include "../allocator.hpp"
#include "../../backends/blas/blas_backend.hpp"

namespace linalg
{
    namespace memory
    {
        //implementation of the aligned host allocator
        class HostAllocator : public Allocator<blas_backend>
        {
        public:
            HostAllocator(const ExecDomain<blas_backend>& domain) : Allocator<blas_backend>(domain){}

            void* allocate_bytes(std::size_t nbytes, std::size_t alignment) override
            {

#if __cplusplus >= 201703L
                return ::operator new(nbytes, std::align_val_t(alignment));
#else

#if defined(_MSC_VER)
                void* ptr = _aligned_malloc(bytes, alignment);
                if (!ptr) {
                    throw std::bad_alloc();
                }
                return ptr;
#else
                void* ptr = nullptr;
                if (posix_memalign(&ptr, alignment, bytes) != 0) {
                    throw std::bad_alloc();
                }
                return ptr;
#endif
#endif
            }

            void deallocate_bytes(void* ptr, std::size_t alignment) override 
            {
                if (!ptr) return;
#if __cplusplus >= 201703L
                ::operator delete(ptr, std::align_val_t(alignment));
#else
#if defined(_MSC_VER)
                _aligned_free(ptr);
#else
                free(ptr);
#endif
#endif
            }
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_BLAS_ALLOCATOR_HPP_
