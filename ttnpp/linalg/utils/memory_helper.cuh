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

#ifndef PYTTN_LINALG_UTILS_MEMORY_HELPER_CUH_
#define PYTTN_LINALG_UTILS_MEMORY_HELPER_CUH_

#include <common/exception_handling.hpp>
#include "memory_helper.hpp"
#include "../backends/cuda/cuda_backend.hpp"
#include "../backends/cuda/cuda_utils.cuh"

#include "../linalg_forward_decl.hpp"

namespace linalg
{
    namespace memory
    {
        template <typename T>
        class allocator<T, cuda_backend>
        {
            using size_type = typename traits<cuda_backend>::size_type;

        public:
            static inline T *allocate(size_type n)
            {
                T *res;
                CALL_AND_HANDLE(cuda_safe_call(cudaMalloc(&res, n * sizeof(T))), "Failed to allocate memory buffer.");
                return res;
            }

            static inline void deallocate(T *&v)
            {
                if (v != nullptr)
                {
                    CALL_AND_HANDLE(cuda_safe_call(cudaFree(v)), "Failed to deallocate memory buffer.");
                }
                v = nullptr;
            }
        };

        namespace internal
        {
            class cuda_transfer
            {
            public:
                using size_type = typename traits<cuda_backend>::size_type;
                template <typename T, typename size_type>
                static inline void copy(const T *const src, size_type n, T *const dest, cudaMemcpyKind type)
                {
                    CALL_AND_HANDLE(cuda_safe_call(cudaMemcpy(dest, src, n * sizeof(T), type)), "cudaMemcpy call failed.");
                }

                template <typename T, typename size_type>
                static inline void copy(const T *const src, size_type n, T *const dest, cudaMemcpyKind type, cudaStream_t stream)
                {
                    CALL_AND_HANDLE(cuda_safe_call(cudaMemcpy(dest, src, n * sizeof(T), type, stream)), "cudaMemcpy call failed.");
                }

                template <typename T, size_t D>
                static inline void copy_noncontiguous(const T *const src, const std::array<size_type, D> &size, const std::array<size_type, D> &s_strides, T *dest, const std::array<size_type, D> &d_strides, cudaMemcpyKind type)
                {
                    RAISE_EXCEPTION("Noncontiguous copy not supported by cuda transfer.");
                }

                template <typename T, size_t D>
                static inline void copy_noncontiguous(const T *const src, const std::array<size_type, D> &size, const std::array<size_type, D> &s_strides, T *dest, const std::array<size_type, D> &d_strides, cudaMemcpyKind type, cudaStream_t stream)
                {
                    RAISE_EXCEPTION("Noncontiguous copy not supported by cuda transfer.");
                }
            };

        } // namespace internal

        // blas to cuda memory transfer
        template <>
        class transfer<blas_backend, cuda_backend>
        {
            using size_type = typename traits<cuda_backend>::size_type;

        public:
            template <typename T>
            static inline void copy(const T *const src, size_type n, T *const dest) { CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyHostToDevice), "Failed to perform host to device memory transfer."); }
            template <typename T>
            static inline void copy(const T *const src, size_type n, T *const dest, cudaStream_t stream) { CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyHostToDevice, stream), "Failed to perform asynchronous host to device memory transfer."); }

            template <typename T, size_t D>
            static inline void copy_noncontiguous(const T *const src, const std::array<size_t, D> &strides, size_type n, T *const dest)
            {
                CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, strides, n, dest, cudaMemcpyHostToDevice));
            }

            template <typename T, size_t D>
            static inline void copy_noncontiguous(const T *const src, const std::array<size_type, D> &size, const std::array<size_type, D> &s_strides, T *dest, const std::array<size_type, D> &d_strides)
            {
                CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, size, s_strides, dest, d_strides, cudaMemcpyHostToDevice));
            }

            template <typename T, size_t D>
            static inline void copy_noncontiguous(const T *const src, const std::array<size_type, D> &size, const std::array<size_type, D> &s_strides, T *dest, const std::array<size_type, D> &d_strides, cudaStream_t stream)
            {
                CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, size, s_strides, dest, d_strides, cudaMemcpyHostToDevice, stream));
            }
        };

        // cuda to blas memory transfer
        template <>
        class transfer<cuda_backend, blas_backend>
        {
            using size_type = typename traits<cuda_backend>::size_type;

        public:
            template <typename T>
            static inline void copy(const T *const src, size_type n, T *const dest) { CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyDeviceToHost), "Failed to perform device to host memory transfer."); }
            template <typename T>
            static inline void copy(const T *const src, size_type n, T *const dest, cudaStream_t stream) { CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyDeviceToHost, stream), "Failed to perform asynchronous device to host memory transfer."); }
        };

        // cuda to cuda memory transfer
        template <>
        class transfer<cuda_backend, cuda_backend>
        {
            using size_type = typename traits<cuda_backend>::size_type;

        public:
            template <typename T>
            static inline void copy(const T *const src, size_type n, T *const dest) { CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyDeviceToDevice), "Failed to perform device to device memory transfer."); }
            template <typename T>
            static inline void copy(const T *const src, size_type n, T *const dest, cudaStream_t stream) { CALL_AND_HANDLE(internal::cuda_transfer::copy(src, n, dest, cudaMemcpyDeviceToDevice, stream), "Failed to perform asynchronous device to device memory transfer."); }

            template <typename T, size_t D>
            static inline void copy_noncontiguous(const T *const src, const std::array<size_type, D> &size, const std::array<size_type, D> &s_strides, T *dest, const std::array<size_type, D> &d_strides)
            {
                CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, size, s_strides, dest, d_strides, cudaMemcpyDeviceToDevice));
            }

            template <typename T, size_t D>
            static inline void copy_noncontiguous(const T *const src, const std::array<size_type, D> &size, const std::array<size_type, D> &s_strides, T *dest, const std::array<size_type, D> &d_strides, cudaStream_t stream)
            {
                CALL_AND_RETHROW(internal::cuda_transfer::copy_noncontiguous(src, size, s_strides, dest, d_strides, cudaMemcpyDeviceToDevice, stream));
            }
        };

        template <typename T>
        class filler<T, cuda_backend>
        {
            using size_type = typename traits<cuda_backend>::size_type;

        public:
            static inline void fill(T *dest, size_type n, const T &val) { cuda_backend::fill_n(dest, n, val); }
        }; // class filler<cuda_backend>
    } // namespace memory
} // namespace linalg

#endif // PYTTN_LINALG_UTILS_MEMORY_HELPER_CUH_//
