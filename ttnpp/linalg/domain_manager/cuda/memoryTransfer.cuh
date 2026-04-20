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
 
#ifndef PYTTN_LINALG_DOMAIN_MANAGER_CUDA_MEMORY_TRANSFER_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_CUDA_MEMORY_TRANSFER_HPP_

#include <cstdint>
#include <cstddef>
#include <memory>
#include <common/exception_handling.hpp>

#include "../../backends/cuda/cuda_backend.cuh"
#include "../../backends/cuda/cuda_utils.cuh"
#include "../memoryTransfer.hpp"
#include "../execProps.hpp"

namespace linalg
{
    namespace memory
    {
        template <>
        class MemoryTransfer<cuda_backend, cuda_backend>
        {
        public:
            template <typename T>
            static void copy(const T* src, T* dst, std::size_t n)
            {
                try
                {
                    auto& ctx = ExecContextScope<cuda_backend>::current();
                    cudaStream_t s = ctx.stream();                
                    CALL_AND_HANDLE(cuda_safe_call(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyDeviceToDevice, s)), "cudaMemcpy call failed.");
                }
                catch(const std::exception& e)
                {
                    logging::error(e.what());
                    RAISE_EXCEPTION("Failed to transfer memory between cuda devices.");
                }
            }
        };

        template <>
        class MemoryTransfer<blas_backend, cuda_backend>
        {
        public:
            template <typename T>
            static void copy(const T* src, T* dst, std::size_t n)
            {
                try
                {
                    auto& ctx = ExecContextScope<cuda_backend>::current();
                    cudaStream_t s = ctx.stream();                
                    CALL_AND_HANDLE(cuda_safe_call(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyHostToDevice, s)), "cudaMemcpy call failed.");
                }
                catch(const std::exception& e)
                {
                    logging::error(e.what());
                    RAISE_EXCEPTION("Failed to transfer from host to cuda device.");
                }
            }
        };

        template <>
        class MemoryTransfer<cuda_backend, blas_backend>
        {
        public:
            template <typename T>
            static void copy(const T* src, T* dst, std::size_t n)
            {
                try
                {
                    auto& ctx = ExecContextScope<cuda_backend>::current();
                    cudaStream_t s = ctx.stream();                
                    CALL_AND_HANDLE(cuda_safe_call(cudaMemcpyAsync(dst, src, n*sizeof(T), cudaMemcpyDeviceToHost, s)), "cudaMemcpy call failed.");
                }
                catch(const std::exception& e)
                {
                    logging::error(e.what());
                    RAISE_EXCEPTION("Failed to transfer memory from cuda device to host.");
                }
            }
        };

    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_CUDA_MEMORY_TRANSFER_HPP_
