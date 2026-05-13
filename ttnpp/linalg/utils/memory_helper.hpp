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

#ifndef PYTTN_LINALG_UTILS_MEMORY_HELPER_HPP_
#define PYTTN_LINALG_UTILS_MEMORY_HELPER_HPP_

#include <common/exception_handling.hpp>
#include "../linalg_forward_decl.hpp"
#include "../backends/blas/blas_backend.hpp"


namespace linalg
{
    namespace memory
    {
        template <typename T, typename backend>
        class allocator
        {
            using size_type = typename traits<backend>::size_type;
        public:
            static T *allocate(size_type n);
            static void deallocate(T *&v);

        };

        template <typename T, typename backend>
        class filler
        {
            using size_type = typename traits<blas_backend>::size_type;

        public:
            static void fill(T *dest, size_type n, const T &val);
        }; // class filler<blas_backend>


                // blas to blas memory transfer
        template <typename B1, typename B2>
        class transfer
        {
            using size_type = typename traits<B1>::size_type;

        public:
            template <typename T>
            static void copy(const T *const src, size_type n, T *dest);
            template <typename T, size_t D>
            static void copy_noncontiguous(const T *const src, const std::array<size_type, D> &size, const std::array<size_type, D> &s_strides, T *dest, const std::array<size_type, D> &d_strides);
        };


        template <typename T>
        class allocator<T, blas_backend>
        {
            using size_type = typename traits<blas_backend>::size_type;

        public:
            static inline T *allocate(size_type n)
            {
                T *res;
                CALL_AND_RETHROW(res = new T[n]);
                return res;
            }
            static inline void deallocate(T *&v)
            {
                if (v != nullptr)
                {
                    delete[] v;
                }
                v = nullptr;
            }
        };

        namespace internal
        {

            template <typename T>
            static inline void copy_buffers(const T *const src, typename traits<blas_backend>::size_type n, T *dest)
            {
                if (dest == src)
                {
                    return;
                }

                std::ptrdiff_t overlap = std::abs(src - dest) < static_cast<std::ptrdiff_t>(n) ? src - dest : 0;

#ifdef LINALG_ALLOW_BUFFER_OVERLAP
                using size_type = typename traits<blas_backend>::size_type;
                // if the two buffers don't overlap then we can just use copy_n
                if (overlap == 0)
                {
                    CALL_AND_HANDLE(std::copy_n(src, n, dest), "Failed to copy blas buffers.   Error when calling copy_n.");
                }
                // if src < dest then copying we need to copy src to dest backwards so that we don't overwrite the elements of src that overlap with the elements of dest
                else if (overlap < 0)
                {
                    for (size_type i = 0; i < n; ++i)
                    {
                        dest[n - (i + 1)] = src[n - (i + 1)];
                    }
                }
                // if src > dest then copying we need to copy src to dest forwards so that we don't overwrite the elements of src that overlap with the elements of dest
                else
                {
                    for (size_type i = 0; i < n; ++i)
                    {
                        dest[i] = src[i];
                    }
                }
#else
                ASSERT(overlap == 0, "Failed to copy buffers. The underlying buffers overlap which has not been allowed.");
                if (overlap == 0)
                {
                    CALL_AND_HANDLE(std::copy_n(src, n, dest), "Failed to copy blas buffers.   Error when calling copy_n.");
                }
#endif
            }
        } // namespace internal

        // blas to blas memory transfer
        template <>
        class transfer<blas_backend, blas_backend>
        {
            using size_type = typename traits<blas_backend>::size_type;

        public:
            template <typename T>
            static inline void copy(const T *const src, size_type n, T *dest) { CALL_AND_HANDLE(internal::copy_buffers(src, n, dest), "Failed to copy buffers."); }

            template <typename T, size_t D>
            static inline void copy_noncontiguous(const T *const src, const std::array<size_type, D> &size, const std::array<size_type, D> &s_strides, T *dest, const std::array<size_type, D> &d_strides)
            {
                size_type dim = 1;
                for (size_type d = 0; d < D; ++d)
                {
                    dim *= size[d];
                }
                for (size_type i = 0; i < dim; ++i)
                {
                    size_type d2 = 0;
                    size_type itemp = i;
                    for (size_type d = 0; d < D; ++d)
                    {
                        size_type iv = (itemp / d_strides[i]);
                        d2 += iv * s_strides[i];
                        itemp -= iv * d_strides[i];
                    }
                    dest[i] = src[d2];
                }
            }
        };


        template <typename T>
        class filler<T, blas_backend>
        {
            using size_type = typename traits<blas_backend>::size_type;

        public:
            static inline void fill(T *dest, size_type n, const T &val) { CALL_AND_HANDLE(std::fill_n(dest, n, val), "Failed to fill buffer.  Call to std::fill_n failed."); }

        }; // class filler<blas_backend>
    } // namespace memory
} // namespace linalg

#endif // PYTTN_LINALG_UTILS_MEMORY_HELPER_HPP_//
