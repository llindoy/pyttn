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

#ifndef PYTTN_LINALG_BACKENDS_CUDA_BACKEND_ALGEBRA_HPP_
#define PYTTN_LINALG_BACKENDS_CUDA_BACKEND_ALGEBRA_HPP_

#include "../../linalg_forward_decl.hpp"
#include "../../utils/linalg_utils.hpp"
#include "../../../common/exception_handling.hpp"
#include "../backend.hpp"

#include "cuda_backend.hpp"
#include "cuda_backend.cuh"


#include <array>
#include <vector>

#include <tuple>
#include <utility>
#include <type_traits>

#include <cub/device/device_reduce.cuh>


namespace linalg
{
    template <>
    class backend_algebra<cuda_backend>
    {
    public:
        using size_type = typename traits<cuda_backend>::size_type;
        using index_type = typename traits<cuda_backend>::index_type;
        using int_type = typename traits<cuda_backend>::int_type;
        using transform_type = cuda_transform_type;

    public:
        template <typename F, typename... Args>
        static inline void async_for(size_type start, size_type end, F &&f, Args &&...args)
        {
            try
            {
                ASSERT(cuda_backend::environment().is_initialised(), "Failed to perform async_for calculation on cuda_backend.  The backend has not been initialised.");
                ASSERT(start <= end, "Unable to perform async_for.  The final index must be less than the starting index");
                for (size_type i = start; i < end; ++i)
                {
                    f(i, std::forward<Args>(args)...);
                    cuda_backend::environment().increment_stream_id();
                }
                cuda_backend::environment().reset_stream_id();
                // now we sync all of the streams
                cuda_backend::synchronise();
            }
            catch (const common::invalid_value &ex)
            {
                logging::error(ex.what());
                RAISE_NUMERIC("evaluating for loop using asynchronous execution.");
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to evaluate for loop using asynchronous execution.");
            }
        }

public: 
    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmv_kernel_selector(bool conjA, bool conjB, size_type m, size_type n, T1 alpha, const T2 *A, int_type inca, const T3 *X, int_type incx, T4 beta, T5 *Y, int_type incy)
    {
        size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
        dim3 dg((m + nthreads - 1) / nthreads);
        dim3 db(nthreads);

        auto ecopt2 = [] __device__(const T2 &a){ return cuda_conj<T2>::eval(a); };
        auto ecopt3 = [] __device__(const T3 &a){ return cuda_conj<T3>::eval(a); };
        auto enopt2 = [] __device__(const T2 &a){ return a; };
        auto enopt3 = [] __device__(const T3 &a){ return a; };
        if (inca == 1 && incx == 1 && incy == 1)
        {
            if (conjA)
            {
                if (conjB)
                {
                    cuda_kernels::elementwise_multiplication<<<dg, db, 0, hCuda()>>>(m, n, alpha, A, X, beta, Y, ecopt2, ecopt3);
                }
                else
                {
                    cuda_kernels::elementwise_multiplication<<<dg, db, 0, hCuda()>>>(m, n, alpha, A, X, beta, Y, ecopt2, enopt3);
                }
            }
            else
            {
                if (conjB)
                {
                    cuda_kernels::elementwise_multiplication<<<dg, db, 0, hCuda()>>>(m, n, alpha, A, X, beta, Y, enopt2, ecopt3);
                }
                else
                {
                    cuda_kernels::elementwise_multiplication<<<dg, db, 0, hCuda()>>>(m, n, alpha, A, X, beta, Y, enopt2, enopt3);
                }
            }
        }
        else
        {
            if (conjA)
            {
                if (conjB)
                {
                    cuda_kernels::elementwise_multiplication_strided<<<dg, db, 0, hCuda()>>>(m, n, alpha, A, inca, X, incx, beta, Y, incy, ecopt2, ecopt3);
                }
                else
                {
                    cuda_kernels::elementwise_multiplication_strided<<<dg, db, 0, hCuda()>>>(m, n, alpha, A, inca, X, incx, beta, Y, incy, ecopt2, enopt3);
                }
            }
            else
            {
                if (conjB)
                {
                    cuda_kernels::elementwise_multiplication_strided<<<dg, db, 0, hCuda()>>>(m, n, alpha, A, inca, X, incx, beta, Y, incy, enopt2, ecopt3);
                }
                else
                {
                    cuda_kernels::elementwise_multiplication_strided<<<dg, db, 0, hCuda()>>>(m, n, alpha, A, inca, X, incx, beta, Y, incy, enopt2, enopt3);
                }
            }
        }
    }

    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmm_kernel_parameter_selector(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2 *A, size_type inca, const T3 *B, size_type ldb, T4 beta, T5 *C, size_type ldc)
    {
        // determine the TILE_DIM and BLOCK_ROWS parameters used for the dgmm operation.  This also checks whether this will result in to much shared memory and if it will then it
        // uses a kernel with a smaller TILE_DIM.  We will want to optimise this in the future.
        if (sizeof(T3) <= sizeof(float))
        {
            dgmm_kernel_launcher<32, 16>(sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
        }
        else
        {
            dgmm_kernel_launcher<16, 8>(sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
        }
    }

    template <size_t TILE_DIM, size_t BLOCK_ROWS, typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void dgmm_kernel_launcher(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2 *A, size_type inca, const T3 *B, size_type ldb, T4 beta, T5 *C, size_type ldc)
    {
        bool conjA = false;
        bool conjB = false;
        bool transDense = false;

        dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

        auto ecopt2 = [] __device__(const T2 &a)
        { return cuda_conj<T2>::eval(a); };
        auto ecopt3 = [] __device__(const T3 &a)
        { return cuda_conj<T3>::eval(a); };
        auto enopt2 = [] __device__(const T2 &a)
        { return a; };
        auto enopt3 = [] __device__(const T3 &a)
        { return a; };
        auto stream = hCuda();
        // we will be calling the dgm_dm_m? kernels so we should determine what parameters we need
        if (sparse_left)
        {
            if (opA == transform_type::c || opA == transform_type::h)
            {
                conjA = true;
            }
            if (opB == transform_type::c || opB == transform_type::h)
            {
                conjB = true;
            }
            if (opB == transform_type::t || opB == transform_type::h)
            {
                transDense = true;
            }
            size_type max_km = m > k ? m : k;

            if (!transDense)
            {
                constexpr size_t smem = sizeof(T5) * TILE_DIM * (TILE_DIM + 1);
                dim3 dimGrid((n + TILE_DIM - 1) / TILE_DIM, (max_km + TILE_DIM - 1) / TILE_DIM, 1);
                if (conjA && conjB)
                {
                    cuda_kernels::dgm_dm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, ecopt2, ecopt3);
                }
                else if (conjA && !conjB)
                {
                    cuda_kernels::dgm_dm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, ecopt2, enopt3);
                }
                else if (!conjA && conjB)
                {
                    cuda_kernels::dgm_dm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, enopt2, ecopt3);
                }
                else
                {
                    cuda_kernels::dgm_dm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, enopt2, enopt3);
                }
            }
            else
            {
                constexpr size_t smem = sizeof(T5) * TILE_DIM * (TILE_DIM + 2);
                dim3 dimGrid((max_km + TILE_DIM - 1) / TILE_DIM, (n + TILE_DIM - 1) / TILE_DIM, 1);
                if (conjA && conjB)
                {
                    cuda_kernels::dgm_dm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, ecopt2, ecopt3);
                }
                else if (conjA && !conjB)
                {
                    cuda_kernels::dgm_dm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, ecopt2, enopt3);
                }
                else if (!conjA && conjB)
                {
                    cuda_kernels::dgm_dm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, enopt2, ecopt3);
                }
                else
                {
                    cuda_kernels::dgm_dm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, A, inca, B, ldb, beta, C, ldc, enopt2, enopt3);
                }
            }
        }
        // if the sparse matrix is on the right then we need to call the dm_dgm_m? kernels
        else
        {
            if (opB == transform_type::c || opB == transform_type::h)
            {
                conjA = true;
            }
            if (opA == transform_type::c || opA == transform_type::h)
            {
                conjB = true;
            }
            if (opA == transform_type::t || opA == transform_type::h)
            {
                transDense = true;
            }
            size_type max_kn = n > k ? n : k;

            if (!transDense)
            {
                constexpr size_t smem = sizeof(T5) * TILE_DIM * (TILE_DIM + 1);
                dim3 dimGrid((max_kn + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM, 1);

                if (conjA && conjB)
                {
                    cuda_kernels::dm_dgm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, ecopt3, ecopt2);
                }
                else if (conjA && !conjB)
                {
                    cuda_kernels::dm_dgm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, ecopt3, enopt2);
                }
                else if (!conjA && conjB)
                {
                    cuda_kernels::dm_dgm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, enopt3, ecopt2);
                }
                else
                {
                    cuda_kernels::dm_dgm_m<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, enopt3, enopt2);
                }
            }
            else
            {
                constexpr size_t smem = sizeof(T5) * TILE_DIM * (TILE_DIM + 2);
                dim3 dimGrid((m + TILE_DIM - 1) / TILE_DIM, (max_kn + TILE_DIM - 1) / TILE_DIM, 1);

                if (conjA && conjB)
                {
                    cuda_kernels::dm_dgm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, ecopt3, ecopt2);
                }
                else if (conjA && !conjB)
                {
                    cuda_kernels::dm_dgm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, ecopt3, enopt2);
                }
                else if (!conjA && conjB)
                {
                    cuda_kernels::dm_dgm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, enopt3, ecopt2);
                }
                else
                {
                    cuda_kernels::dm_dgm_mt<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock, smem, stream>>>(m, n, k, alpha, B, ldb, A, inca, beta, C, ldc, enopt3, enopt2);
                }
            }
        }
    }

    public:
        template <typename T>
        static inline bool is_equal(const T *a, const T *b, size_type N)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend is_equal call failed.  The cuda environment has not yet been initialised.");
            
            bool* d_compare;
            bool* d_result;
            cuda_safe_call(cudaMalloc(&d_compare, N*sizeof(bool)));
            cuda_safe_call(cudaMalloc(&d_result, sizeof(bool)));

            cuda_kernels::compare<<<(N+255)/256, 256>>>(a, b, N, d_compare);

            void *d_temp = nullptr;
            size_t temp_bytes = 0;

            //get temporary storage size
            cub::DeviceReduce::Reduce(nullptr, temp_bytes, d_compare, d_result, N, cuda_kernels::andOp(), true);
            cuda_safe_call(cudaMalloc(&d_temp, temp_bytes));
            cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_compare, d_result, N, cuda_kernels::andOp(), true);

            bool h_result;
            cuda_safe_call(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));

            cuda_safe_call(cudaFree(d_compare));
            cuda_safe_call(cudaFree(d_result));
            cuda_safe_call(cudaFree(d_temp));

            return h_result;
        }

        template <typename T>
        static inline void vector_scalar_product(int_type N, T A, const T *X, int_type INCX, T *Y, int_type INCY)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend vector_scalar_product call failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((N + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::vector_scalar_product<<<dg, db, 0, hCuda()>>>(N, A, X, INCX, Y, INCY);
        }
        
        template <typename T>
        static inline T trace(int_type N, const T *X, int_type INCX)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend vector_scalar_product call failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((N + nthreads - 1) / nthreads);
            dim3 db(nthreads);

            T* d_result;
            cuda_safe_call(cudaMalloc(&d_result, sizeof(T)));        
            size_t shmem = nthreads*sizeof(T);

            cuda_kernels::trace_grid_atomic<T><<<dg, db, shmem, hCuda()>>>(X, N, INCX, d_result);
            T h_result;
            cuda_safe_call(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
            cuda_safe_call(cudaFree(d_result));
            return h_result;
        }
        
        // valid-1/2/3 via cublas routines
        template <typename T>
        static inline void axpy(int_type N, T A, const T *X, int_type INCX, T *Y, int_type INCY)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend axpy call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_backend::set_cublas_stream(), "axpy call Failed.");
            CALL_AND_HANDLE(cublas::axpy(hCublas(), N, &A, X, INCX, Y, INCY), "axpy call failed using cuda_backend.");
        }

        template <typename T>
        static inline void scal(int_type N, T A, T *X, int_type INCX)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend Scal call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_backend::set_cublas_stream(), "scal call Failed.");
            CALL_AND_HANDLE(cublas::scal(hCublas(), N, &A, X, INCX), "Scal call failed using cuda_backend.");
        }

        template <typename T>
        static inline T dot(bool conj, int_type N, const T *X, int_type INCX, const T *Y, int_type INCY)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend dot call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_backend::set_cublas_stream(), "dot call Failed.");
            CALL_AND_HANDLE(return cublas::dot(hCublas(), conj, N, X, INCX, Y, INCY), "Failed to compute dot product using cuda backend.");
        }

        template <typename T>
        static inline void
        gemm(transform_type TRANSA, transform_type TRANSB, int_type M, int_type N, int_type K, T ALPHA, const T *A, int_type LDA, const T *B, int_type LDB, T BETA, T *C, int_type LDC)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend gemm call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_backend::set_cublas_stream(), "gemm call Failed.");
            CALL_AND_HANDLE(cublas::gemm(hCublas(), map_op(TRANSA), map_op(TRANSB), M, N, K, &ALPHA, A, LDA, B, LDB, &BETA, C, LDC), "Failed to compute general matrix matrix product using cuda backend.");
        }

        template <typename T>
        static inline void
        gemv(transform_type trans, int_type m, int_type n, T alpha, const T *A, int_type lda, const T *x, int_type incx, T beta, T *y, int_type incy)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend gemv call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_backend::set_cublas_stream(), "gemv call Failed.");
            CALL_AND_HANDLE(cublas::gemv(hCublas(), map_op(trans), m, n, &alpha, A, lda, x, incx, &beta, y, incy), "Failed to compute general matrix vector product using cuda backend.");
        }


        // batched_gemm
        template <typename T>
        static inline void
        batched_gemm(transform_type opA, transform_type opB, int_type m, int_type n, int_type k, T alpha, const T *A, int_type lda, long long int strideA, const T *B, int_type ldb, long long int strideB, T beta, T *C, int_type ldc, long long int strideC, int_type batchCount)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend batched_gemm call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_backend::set_cublas_stream(), "batched_gemm call Failed.");
            CALL_AND_HANDLE(cublas::batched_gemm(hCublas(), map_op(opA), map_op(opB), m, n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount), "Failed to cublas strided batched gemm");
        }

        // outer contraction of rank 3 tensors
        template <typename T>
        static inline void outer_contract(transform_type opA, transform_type opB, int_type m, int_type n, int_type k, T alpha, const T *A,
                                          int_type lda, long long int strideA, const T *B, int_type ldb, long long int strideB, T beta, T *C, int_type ldc,
                                          long long int strideC, int_type batchCount, T *res)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend outer_contract call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_backend::set_cublas_stream(), "outer_contract call Failed.");
            T b(0.0);
            // first we go ahead and perform the many parallel small matrix products required
            CALL_AND_HANDLE(batched_gemm(opA, opB, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, b, C, ldc, strideC, batchCount), "Failed to compute contraction of outer indices of two rank three tensors.  Failed to compute the contraction over the last index.");

            // first we check that the backend has a large enough vector of ones to perform the final contraction using gemv.  If it isn't then we will throw an exception - as we don't want these routines to break thread safety.
            CALL_AND_HANDLE(cuda_internals::allocate_ones<T>(batchCount), "Failed to allocate ones array.");
            auto &_ones = std::get<cuda_internals::ones_indexer<T>::index()>(cuda_internals::cuda_ones::ones());

            // now we set up the gemv call to contract over k.  To do this we do C_{ij} = \sum_k A_{kij} v_k where v_k = 1.  This can be
            // performed using a gemv call.
            T a(1.0);
            transform_type op = transform_type::n;
            int_type incx = 1;
            int_type incy = 1;
            int_type mv = m * n;
            int_type nv = batchCount;
            int_type ldav = mv;
            CALL_AND_HANDLE(gemv(op, mv, nv, a, C, ldav, std::get<0>(_ones), incx, beta, res, incy), "Failed to compute contraction of outer indices of two rank three tensor.  Failed to compute the contraction over the first index.");
        }

    public:
        // sparse matrix vector operations
        template <typename T1, typename T2, typename T3, typename T4, typename T5>
        static inline void dgmv(bool conjA, bool conjB, int_type m, int_type n, T1 alpha, const T2 *A, int_type inca, const T3 *X, int_type incx, T4 beta, T5 *Y, int_type incy)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend dgmv call failed.  The cuda environment has not yet been initialised.");
            dgmv_kernel_selector(conjA, conjB, m, n, alpha, A, inca, X, incx, beta, Y, incy);
        }

        // sparse matrix matrix operations
        template <typename T1, typename T2, typename T3, typename T4, typename T5>
        static inline void dgmm(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2 *A, size_type inca, const T3 *B, size_type ldb, T4 beta, T5 *C, size_type ldc)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend dgmm call failed.  The cuda environment has not yet been initialised.");
            dgmm_kernel_parameter_selector(sparse_left, opA, opB, m, n, k, alpha, A, inca, B, ldb, beta, C, ldc);
        }

    public:
        template <typename T>
        static inline void csrmv(transform_type opA, bool /*conjB*/, int_type m, int_type n, size_type nnz, T alpha, const T *A, const int *rowptr, const int *colind, const T *X, int_type incx, T beta, T *Y, int_type incy)
        {
            ASSERT(incx == 1 && incy == 1, "Failed to compute csrmv.  cuda spmv only supports contiguous vectors.");
            auto _opA = map_sp_op(opA);

            CALL_AND_HANDLE(cusparse::spmv(hCusparse(), _opA, m, n, nnz, alpha, A, rowptr, colind, X, beta, Y), "Failed to compute csr matrix - matrix multiplication.");
        }

        template <typename T>
        static inline void csrmv(transform_type opA, bool conjB, int_type m, int_type n, size_type nnz, cuda::std::complex<T> alpha, const cuda::std::complex<T> *A, const int *rowptr, const int *colind, const cuda::std::complex<T> *X, int_type incx, cuda::std::complex<T> beta, cuda::std::complex<T> *Y, int_type incy)
        {
            ASSERT(!conjB, "Failed to compute csrmv.  cuda spmv does not support conjugated vectors.");
            ASSERT(incx == 1 && incy == 1, "Failed to compute csrmv.  cuda spmv only supports contiguous vectors.");
            auto _opA = map_sp_op(opA);

            CALL_AND_HANDLE(cusparse::spmv(hCusparse(), _opA, m, n, nnz, alpha, A, rowptr, colind, X, beta, Y), "Failed to compute csr matrix - matrix multiplication.");
        }
    public:
        template <typename T>
        static inline void csrmm(bool opres, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, size_type nnz, T alpha, const T *A, const int *rowptr, const int *colind, const T *B, size_type ldb, T beta, T *C, size_type ldc)
        {
            ASSERT(!opres, "cuda backend does not support csrmm with transposed results.");

            auto _opA = map_sp_op(opA);
            auto _opB = map_sp_op(opB);

            CALL_AND_HANDLE(cusparse::spmm(hCusparse(), _opA, _opB, m, n, k, nnz, alpha, A, rowptr, colind, B, ldb, beta, C, ldc), "Failed to compute csr matrix - matrix multiplication.");
        }

    public:
        template <typename T>
        static inline void complex_conjugate(size_type /*size*/, const T *const /*X*/, T *const /*Y*/){}

        template <typename T>
        static inline void complex_conjugate(size_type size, const cuda::std::complex<T> *const X, cuda::std::complex<T> *const Y)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend complex_conjugate call failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((size + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::complex_conjugate<<<dg, db, 0, hCuda()>>>(X, size, Y);
        }

        template <typename T, typename expr>
        static inline void evaluate_expression_tree(T *res, size_type n, const expr &e)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::eval_expression_kernel<<<dg, db, 0, hCuda()>>>(res, n, e);
        }

        template <typename T, typename expr>
        static inline void evaluate_add_expression_tree(T *res, size_type n, const expr &e)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::eval_add_expression_kernel<<<dg, db, 0, hCuda()>>>(res, n, e);
        }

        template <typename T, typename expr>
        static inline void evaluate_sub_expression_tree(T *res, size_type n, const expr &e)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::eval_sub_expression_kernel<<<dg, db, 0, hCuda()>>>(res, n, e);
        }

        template <typename T, typename expr>
        static inline void evaluate_expression_tree_strided(T *res, size_type n, size_type stride, const expr &e)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::eval_expression_strided_kernel<<<dg, db, 0, hCuda()>>>(res, n, stride, e);
        }

        template <typename T, typename expr>
        static inline void evaluate_add_expression_tree_strided(T *res, size_type n, size_type stride, const expr &e)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::eval_add_expression_strided_kernel<<<dg, db, 0, hCuda()>>>(res, n, stride, e);
        }

        template <typename T, typename expr>
        static inline void evaluate_sub_expression_tree_strided(T *res, size_type n, size_type stride, const expr &e)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend evaluate expression tree call failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::eval_sub_expression_strided_kernel<<<dg, db, 0, hCuda()>>>(res, n, stride, e);
        }

    public:
        template <typename T>
        static inline void transpose(bool conj, int_type m, int_type n, const T &alpha, const T *in, const T &beta, T *out)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend transpose call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_backend::set_cublas_stream(), "transpose call Failed.");
            ASSERT(in != out, "Failed to evaluate cuda_backend::transpose.  The input and output buffers must not be the same.");
            transform_type op = conj ? transform_type::h : transform_type::t;
            CALL_AND_HANDLE(cublas::geam(hCublas(), map_op(op), map_op(transform_type::n), m, n, &alpha, in, n, &beta, in, m, out, m), "Failed to evaluate cuda_backend::transpose.  Error when calling geam.");
        }

        // we might want to modify this in the future so that it uses a specialised kernel for performing the batched transpose operation.  This is likely to lead to improved performance
        // for problems in which the overhead of the kernel launches dominates the time to run the operation.
        template <typename T>
        static inline void batched_transpose(bool conj, size_type m, size_type n, const T &alpha, const T *in, const T &beta, T *out, size_type batchCount)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend batched_transpose call failed.  The cuda environment has not yet been initialised.");
            ASSERT(in != out, "Failed to evaluate cuda_backend::batched_transpose.  The input and output buffers must not be the same.");
            transform_type op = conj ? transform_type::h : transform_type::t;

            size_type skip = m * n;
            for (size_type i = 0; i < batchCount; ++i)
            {
                size_type bskip = skip * i;
                CALL_AND_HANDLE(cuda_backend::set_cublas_stream(), "batched transpose call Failed.  Failed to set cublas stream to parallelise transpose calls.");
                CALL_AND_HANDLE(cublas::geam(hCublas(), map_op(op), map_op(transform_type::n), m, n, &alpha, in, n, &beta, in + bskip, m, out + bskip, m), "Failed to evaluate cuda_backend::transpose.  Error when calling geam.");
                cuda_backend::environment().increment_stream_id();
            }
            cuda_backend::environment().reset_stream_id();
            cudaDeviceSynchronize();
        }

        // function for copying between two buffers
        template <typename T>
        static inline void copy(const T *src, size_type n, T *dest)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend copy operation failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_safe_call((cudaMemcpy(dest, src, n * sizeof(T), cudaMemcpyDeviceToDevice))), "Failed to copy memory buffer from one buffer to another.  cudaMemcpy call failed.");
        }

        template <typename T>
        static inline void rank_3_strided_copy(const T *src, size_type n1, size_type n2, size_type n3, T *dest, size_type n4)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
            size_t Bx = 32;
            size_t By = 4;
            size_t Bz = 1;
            dim3 dg((n3+Bx-1)/Bx, (n2+By-1)/By, (n1+Bz-1)/Bz);
            dim3 db(Bx, By,  Bz);
            cuda_kernels::copy_tensor3_strided<<<dg, db, 0, hCuda()>>>(src, dest, n1, n2, n3, n4);
        }

        template <typename T>
        static inline void rank_3_strided_append(const T *src, size_type n1, size_type n2, size_type n3, size_type iadd, T *dest, size_type n4)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
            size_t Bx = 32;
            size_t By = 4;
            size_t Bz = 1;
            dim3 dg((n3+Bx-1)/Bx, (iadd+By-1)/By, (n1+Bz-1)/Bz);
            dim3 db(Bx, By,  Bz);
            cuda_kernels::copy_tensor3_append<<<dg, db, 0, hCuda()>>>(src, dest, n1, n2, n3, iadd, n4);
        }

        template <typename T>
        static inline void assign(const T *src, size_type n, T *dest, T beta = T(0))
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::addition_assign_array<<<dg, db, 0, hCuda()>>>(src, n, dest);
        }

        template <typename T>
        static inline void addition_assign(const T *src, size_type n, T *dest)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::addition_assign_array<<<dg, db, 0, hCuda()>>>(src, n, dest);
        }


        template <typename T>
        static inline void subtraction_assign(const T *src, size_type n, T *dest)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::subtraction_assign_array<<<dg, db, 0, hCuda()>>>(src, n, dest);
        }

        template <typename T>
        static inline void copy_real_to_complex(const T *src, size_type n, cuda::std::complex<T> *dest)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend copy_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::copy_real_to_complex_array<<<dg, db, 0, hCuda()>>>(src, n, dest);
        }

        template <typename T>
        static inline void addition_assign_real_to_complex(const T *src, size_type n, cuda::std::complex<T> *dest)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::addition_assign_real_to_complex_array<<<dg, db, 0, hCuda()>>>(src, n, dest);
        }

        template <typename T>
        static inline void subtraction_assign_real_to_complex(const T *src, size_type n, cuda::std::complex<T> *dest)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend addition_assign_real_to_complex operation failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::addition_assign_real_to_complex_array<<<dg, db, 0, hCuda()>>>(src, n, dest);
        }

        template <typename T>
        static inline void copy_matrix_subblock(size_type m1, size_type n1, const T *src, size_type lda, T *dest, size_type ldb)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend copy_matrix_subblock operation failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cuda_safe_call((cudaMemcpy2D(dest, lda * sizeof(T), src, ldb * sizeof(T), n1, m1, cudaMemcpyDeviceToDevice))), "Failed to copy memory buffer from one buffer to another.  cudaMemcpy2D call failed.");
        }

        // function for filling a buffer with a value
        template <typename T>
        static inline void fill_n(T *dest, size_type n, const T &val)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend fill_n operation failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::fill_n<<<dg, db, 0, hCuda()>>>(dest, n, val);
        }

        template <typename T, typename Func, typename... Args>
        static inline void func_fill_1(T *res, size_type m, Func &&f, Args &&...args)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend func_fill_1 operation failed.  The cuda environment has not yet been initialised.");
            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((m + nthreads - 1) / nthreads);
            dim3 db(nthreads);
            cuda_kernels::fill_func_1<<<dg, db, 0, hCuda()>>>(res, m, std::forward<Func>(f), std::forward<Args>(args)...);
        }

        template <typename T, typename Func, typename... Args>
        static inline void func_fill_2(T *res, size_type m, size_type n, Func &&f, Args &&...args)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend func_fill_2 operation failed.  The cuda environment has not yet been initialised.");
            dim3 dg, db;
            size_type mnt = cuda_backend::environment().maximum_threads_per_block();
            size_type mnx = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            size_type mny = cuda_backend::environment().maximum_dimensions_threads_per_block()[1];
            if (n >= mnt)
            {
                db.y = mnt < mny ? mnt : mny;
                db.x = mnt / db.y;
            }
            else
            {
                if (n >= 16 && (n & (n - 1)) == 0)
                {
                    db.y = n < mny ? n : mny;
                    db.x = mnt / db.y;
                } // if m >= 16 and m is a power of 2 then we use m as the value for db.x
                else
                {
                    db.y = 16 < mny ? 16 : mny;
                    db.x = mnt / db.y;
                } // otherwise we will just use 16 for db.x
            }
            db.x = db.x < mnx ? db.x : mnx;
            dg.x = (m + db.x - 1) / db.x;
            dg.y = (n + db.y - 1) / db.y;

            cuda_kernels::fill_func_2<<<dg, db, 0, hCuda()>>>(res, m, n, std::forward<Func>(f), std::forward<Args>(args)...);
        }

        template <typename T, typename Func, typename... Args>
        static inline void func_fill_3(T *res, size_type m, size_type n, size_type o, Func &&f, Args &&...args)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend func_fill_3 operation failed.  The cuda environment has not yet been initialised.");
            dim3 dg, db;
            size_type mnt = cuda_backend::environment().maximum_threads_per_block();
            size_type mnx = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            size_type mny = cuda_backend::environment().maximum_dimensions_threads_per_block()[1];
            size_type mnz = cuda_backend::environment().maximum_dimensions_threads_per_block()[2];

            if (o >= mnt)
            {
                db.z = mnt < mnz ? mnt : mnz;

                size_type ncomp = mnt / db.z;
                if (n >= ncomp)
                {
                    db.y = ncomp < mny ? ncomp : mny;
                    db.x = ncomp / db.y;
                }
                else
                {
                    size_type maxy = mny < ncomp ? mny : ncomp;
                    if (n >= 8 && (n & (n - 1)) == 0)
                    {
                        db.y = n < maxy ? n : maxy;
                        db.x = ncomp / db.y;
                    } // if m >= 16 and m is a power of 2 then we use m as the value for db.x
                    else
                    {
                        db.y = 8 < maxy ? 8 : maxy;
                        db.x = ncomp / db.y;
                    } // otherwise we will just use 16 for db.x
                }
            }
            else
            {
                if (o >= 16 && (o & (o - 1)) == 0)
                {
                    db.z = o;
                }
                else
                {
                    db.z = 16;
                }
                db.z = db.z < mnz ? db.z : mnz;

                size_type ncomp = mnt / db.z;
                db.y = ncomp < mny ? ncomp : mny;
                db.x = ncomp / db.y;
                while (db.y > n)
                {
                    db.y = db.y >> 1;
                    db.x = db.x << 1;
                }
            }
            db.x = db.x < mnx ? db.x : mnx;

            dg.x = (m + db.x - 1) / db.x;
            dg.y = (n + db.y - 1) / db.y;
            dg.z = (o + db.z - 1) / db.z;
            cuda_kernels::fill_func_3<<<dg, db, 0, hCuda()>>>(res, m, n, o, std::forward<Func>(f), std::forward<Args>(args)...);
        }

        template <typename T>
        static inline void fill_matrix_block(const T *src, size_type m, size_type n, T *dest, size_type m2, size_type n2)
        {
            ASSERT(n <= n2 && m <= m2, "fill_block call failed.  The subblock is larger than the full matrix.");
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend func_fill_2 operation failed.  The cuda environment has not yet been initialised.");
            dim3 dg, db;
            size_type mnt = cuda_backend::environment().maximum_threads_per_block();
            size_type mnx = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            size_type mny = cuda_backend::environment().maximum_dimensions_threads_per_block()[1];
            if (n >= mnt)
            {
                db.y = mnt < mny ? mnt : mny;
                db.x = mnt / db.y;
            }
            else
            {
                if (n >= 16 && (n & (n - 1)) == 0)
                {
                    db.y = n < mny ? n : mny;
                    db.x = mnt / db.y;
                } // if m >= 16 and m is a power of 2 then we use m as the value for db.x
                else
                {
                    db.y = 16 < mny ? 16 : mny;
                    db.x = mnt / db.y;
                } // otherwise we will just use 16 for db.x
            }
            db.x = db.x < mnx ? db.x : mnx;
            dg.x = (m + db.x - 1) / db.x;
            dg.y = (n + db.y - 1) / db.y;

            cuda_kernels::fill_matrix_block<<<dg, db, 0, hCuda()>>>(src, m, n, dest, m2, n2);
        }

        template <typename T, size_t D>
        static inline void set_tensor_block(const T* src, const std::array<size_type, D>& src_dims, T* dest, const std::array<size_type, D>& dest_dims, const std::array<size_type, D>& skip)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend fill_n operation failed.  The cuda environment has not yet been initialised.");

            std::array<size_type, D> dest_stride;
            std::array<size_type, D> src_stride;

            std::fill(dest_stride.begin(), dest_stride.end(), 1);
            std::fill(src_stride.begin(), src_stride.end(), 1);

            for(size_type k = 1; k < D; ++k)
            {
                size_type kd = D-(k+1);
                dest_stride[kd] = dest_stride[kd+1]*dest_dims[kd+1];
                src_stride[kd] = src_stride[kd+1]*src_dims[kd+1];
            }
            size_t n = src_stride[0]*src_dims[0];

            size_type nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
            dim3 dg((n + nthreads - 1) / nthreads);
            dim3 db(nthreads);

            cuda_kernels::copy_tensor_subblock_nd<<<dg, db, 0, hCuda()>>>(src, &src_stride[0], &src_dims[0], dest, &dest_stride[0], &skip[0], D, n);    
        }


        template <typename T, typename U>
        static inline void transfer_coo_tuple_to_csr(const std::vector<std::tuple<index_type, index_type, T>> &coo, U *vals, index_type *colinds)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend transfer_coo_tuple_to_csr operation failed.  The cuda environment has not yet been initialised.");
            T *h_vals;
            index_type *h_colinds;
            size_type n = coo.size();
            CALL_AND_HANDLE(h_vals = new T[n], "Failed to transfer coo tuple to csr object.  Error when allocating device vector for storing values.");
            CALL_AND_HANDLE(h_colinds = new index_type[n], "Failed to transfer coo tuple to csr object.  Error when allocating device vector for storing colinds.")

            // now we copy the vector buffer to set colinds
            for (size_type i = 0; i < n; ++i)
            {
                h_colinds[i] = std::get<1>(coo[i]);
                h_vals[i] = std::get<2>(coo[i]);
            }

            // and now we copy the results
            CALL_AND_HANDLE(cuda_safe_call((cudaMemcpy(reinterpret_cast<U*>(h_vals), vals, n * sizeof(U), cudaMemcpyHostToDevice))), "Failed to transfer coo tuple to csr object.  cudaMemcpy call failed.");
            CALL_AND_HANDLE(cuda_safe_call((cudaMemcpy(h_colinds, colinds, n * sizeof(index_type), cudaMemcpyHostToDevice))), "Failed to transfer coo tuple to csr object.  cudaMemcpy call failed.");

            CALL_AND_HANDLE(delete[] h_vals, "Failed to transfer coo tuple to csr object.  Error when deallocating device vector for storing values.");
            CALL_AND_HANDLE(delete[] h_colinds, "Failed to transfer coo tuple to csr object.  Error when deallocating device vector for storing colinds.")
        }

    public:
        template <typename T>
        static inline void heev(eig_mode jobz, fill_mode uplo, int_type n, T *A, int_type lda, typename get_real_type<T>::type *W, T *work, int_type lwork, int *devinfo)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cusolver::heev(hCusolver(), map_eig_mode(jobz), map_fill(uplo), n, A, lda, W, work, lwork, devinfo), "cuda backend heev call failed.  Error when calling heev.");
        }
        template <typename T>
        static inline void heev_buffersize(eig_mode jobz, fill_mode uplo, int_type n, T *A, int_type lda, typename get_real_type<T>::type *W, int *lwork)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cusolver::heev_buffersize(hCusolver(), map_eig_mode(jobz), map_fill(uplo), n, A, lda, W, lwork), "cuda backend heev_buffersize call failed.  Error when calling determining the workspace buffer size.");
        }

    public:
        template <typename T>
        static inline void getrf(int_type m, int_type n, T *A, int_type lda, T *work, int *ipiv, int *devinfo)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend getrf call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cusolver::getrf(hCusolver(), m, n, A, lda, work, ipiv, devinfo), "cuda backend getrf call failed.  Error when calling getrf.");
        }

        template <typename T>
        static inline void getrf_buffersize(int_type m, int_type n, T *A, int_type lda, int *lwork)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend getrf call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cusolver::getrf_buffersize(hCusolver(), m, n, A, lda, lwork), "cuda backend getrf_buffersize call failed.  Error when determining the workspace buffer size.");
        }

    public:
        template <typename T>
        static inline void gesvd(const char jobu, const char jobv, const int_type m, const int_type n, T *A, const int_type lda, typename get_real_type<T>::type *S, T *U, const int_type ldu, T *VT, const int_type ldvt, T *work, const int_type lwork, typename get_real_type<T>::type *rwork, int *devinfo)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cusolver::gesvd(hCusolver(), jobu, jobv, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devinfo), "cuda backend gesvd call failed.  Error when calling gesvd.");
        }

        template <typename T>
        struct gesvd_buffersize
        {
            static inline void eval(int_type m, int_type n, int &lwork)
            {
                ASSERT(cuda_backend::environment().is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
                CALL_AND_HANDLE(cusolver::gesvd_params<T>::buffersize(hCusolver(), m, n, lwork), "cuda backend gesvd_buffersize call failed.  Error when calling determining the workspace buffer size.");
            }
        };

        template <typename T>
        static inline void gesvdj_buffersize(eig_mode jobz, const int_type econ, const int_type m, const int_type n, T *A, const int_type lda, typename get_real_type<T>::type *S, T *U, const int_type ldu, T *VT, const int_type ldvt, int &lwork, void* params)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cusolver::gesvdj_buffersize(hCusolver(), map_eig_mode(jobz), econ, m, n, A, lda, S, U, ldu, VT, ldvt, lwork, static_cast<gesvdjInfo_t>(params)), "cuda backend gesvd_buffersize call failed.  Error when calling determining the workspace buffer size.");
        }
        

        template <typename T>
        static inline void gesvdj(eig_mode jobz, const int_type econ, const int_type m, const int_type n, T *A, const int_type lda, typename get_real_type<T>::type *S, T *U, const int_type ldu, T *VT, const int_type ldvt, T *work, const int_type lwork, int *devinfo, void* params)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend heev call failed.  The cuda environment has not yet been initialised.");
            CALL_AND_HANDLE(cusolver::gesvdj(hCusolver(), map_eig_mode(jobz), econ, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, devinfo, static_cast<gesvdjInfo_t>(params)), "cuda backend gesvd call failed.  Error when calling gesvd.");
        }
    public:
        template <typename T, typename arr1, typename arr2, typename arr3, typename arr4>
        static inline void tensor_transpose(const T *in, const arr1 &dimsA, const arr2 &_strideA, T *out, const arr3 &dimsB, const arr4 &_strideB, const std::vector<size_type> &inds)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend tensor transpose call failed.  The cuda environment has not yet been initialised.");

            std::vector<int64_t> extentA(dimsA.size());
            std::vector<int64_t> extentB(dimsA.size());
            std::vector<int64_t> strideA(dimsA.size());
            std::vector<int64_t> strideB(dimsA.size());

            for (size_t i = 0; i < dimsA.size(); ++i)
            {
                extentA[i] = dimsA[i];
                strideA[i] = _strideA[i];
                extentB[i] = dimsB[i];
                strideB[i] = _strideB[i];
            }
            CALL_AND_HANDLE(cutensor::transpose<T>(hCutensor(), hCuda(), in, extentA, strideA, out, extentB, strideB, inds), "Failed to compute tensor transpose.")
        }

        template <typename T>
        T determinant_reduction(T* red, size_t N)
        {
            ASSERT(cuda_backend::environment().is_initialised(), "cuda backend is_equal call failed.  The cuda environment has not yet been initialised.");
            
            T* d_result;
            cuda_safe_call(cudaMalloc(&d_result, sizeof(T)));

            void *d_temp = nullptr;
            size_t temp_bytes = 0;

            //get temporary storage size
            cub::DeviceReduce::Reduce(nullptr, temp_bytes, red, d_result, N, cuda_kernels::mulOp<T>(), T(1.0));
            cuda_safe_call(cudaMalloc(&d_temp, temp_bytes));
            cub::DeviceReduce::Reduce(d_temp, temp_bytes, red, d_result, N, cuda_kernels::mulOp<T>(), T(1.0));

            T h_result;
            cuda_safe_call(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

            cuda_safe_call(cudaFree(d_result));
            cuda_safe_call(cudaFree(d_temp));

            return h_result;
        }
    }; // backend_algebra<cuda_backend>

} // namespace linalg

#endif // PYTTN_LINALG_BACKENDS_CUDA_BACKEND_HPP_//
