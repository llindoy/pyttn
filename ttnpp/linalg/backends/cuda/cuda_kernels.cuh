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

#ifndef PYTTN_LINALG_BACKENDS_CUDA_KERNELS_CUH_
#define PYTTN_LINALG_BACKENDS_CUDA_KERNELS_CUH_

#include "../../utils/linalg_utils.cuh"
#include <cuda_runtime.h>
#include <cuda/std/complex>


// a file containing the kernels used for the linear algebra routines.
// TODO: add kernel for setting a block of a tensor

namespace linalg
{
    namespace cuda_kernels
    {
        struct andOp
        {
            __device__ bool operator()(const bool& a, const bool& b) const;
        };

        template <typename T>
        struct mulOp
        {
            __device__ T operator()(const T& a, const T& b) const;
        };

        template <typename T>
        __global__ void trace_grid_atomic(const T* __restrict__ A, size_t n, size_t incx, T* __restrict__ out);

        template <typename T>
        __global__ void copy_tensor3_strided(const T* __restrict__ src, T* __restrict dst, size_t D, size_t H, size_t W, size_t Dout);

        template <typename T>
        __global__ void copy_tensor3_append(const T* __restrict__ src, T* __restrict dst, size_t D, size_t H, size_t W, size_t iadd, size_t Dout);


        template <typename T, size_t MAX_D=32>
        __global__ void copy_tensor_subblock_nd(const T* __restrict__ src, const size_t* __restrict__ src_strides, const size_t* __restrict__ src_dims, 
            T* __restrict dst, const size_t* __restrict__ dst_strides, const size_t* __restrict__ dst_offset, size_t D, size_t nelem);

        template <typename T>
        __global__ void compare(const T *a, const T* b, const size_t n, bool *out);

        template <typename T>
        __global__ void addition_assign_array(const T *in, const size_t n, T *out);

        template <typename T>
        __global__ void subtraction_assign_array(const T *in, const size_t n, T *out);
    
        template <typename T>
        __global__ void copy_real_to_complex_array(const T *in, const size_t n, cuda::std::complex<T> *out);

        template <typename T>
        static __global__ void addition_assign_real_to_complex_array(const T *in, const size_t n, cuda::std::complex<T> *out);

        template <typename T>
        static __global__ void subtraction_assign_real_to_complex_array(const T *in, const size_t n, cuda::std::complex<T> *out);

        template <typename T>
        static __global__ void fill_matrix_block(const T *src, size_t m1, size_t n1, T *dest, size_t m2, size_t n2);
        template <typename T>

        static __global__ void complex_conjugate(const cuda::std::complex<T> *const in, const size_t n, cuda::std::complex<T> *const out);
        template <typename T>
        
        static __global__ void vector_scalar_product(int N, T A, const T *X, int INCX, T *Y, int INCY);

        template <typename T>
        static __global__ void axpy_conj(const cuda::std::complex<T> *in, const size_t n, cuda::std::complex<T> *out);

        template <typename T, typename expr>
        static __global__ void eval_expression_strided_kernel(T *res, size_t n, size_t resstride, expr op);

        template <typename T, typename expr>
        static __global__ void eval_add_expression_strided_kernel(T *res, size_t n, size_t resstride, expr op);

        template <typename T, typename expr>
        static __global__ void eval_sub_expression_strided_kernel(T *res, size_t n, size_t resstride, expr op);

        template <typename T, typename expr>
        static __global__ void eval_expression_kernel(T *res, size_t n, expr op);

        template <typename T, typename expr>
        static __global__ void eval_add_expression_kernel(T *res, size_t n, expr op);

        template <typename T, typename expr>
        static __global__ void eval_sub_expression_kernel(T *res, size_t n, expr op);

        // kernel for filling an array with a value
        template <typename T>
        __global__ void fill_n(T *devPtr, const size_t m, const T val);

        template <typename T>
        __global__ void fill_n_strided(T *devPtr, const size_t m, const size_t inc, const T val);

        template <typename T, typename Func, typename... Args>
        __global__ void fill_func_1(T *devPtr, const size_t m, Func f, Args... args);

        template <typename T, typename Func, typename... Args>
        __global__ void fill_func_2(T *devPtr, const size_t m, const size_t n, Func f, Args... args);

        template <typename T, typename Func, typename... Args>
        __global__ void fill_func_3(T *devPtr, const size_t m, const size_t n, const size_t o, Func f, Args... args);

        template <typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void elementwise_multiplication(const size_t m, const size_t n, T1 alpha, const T2 *A, const T3 *B, T4 beta, T5 *C, OPA opa, OPB opb);

        template <typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void elementwise_multiplication_strided(const size_t m, const size_t n, T1 alpha, const T2 *A, const size_t inca, const T3 *B, const size_t incb, T4 beta, T5 *C, const size_t incc, OPA opa, OPB opb);
        
        // diagonal matrix * dense matrix
        template <size_t TILE_DIM, size_t BLOCK_ROWS, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void dgm_dm_m(const size_t m, const size_t n, const size_t k, T1 alpha, const T2 *  __restrict__ A, const size_t inca, const T3 *  __restrict__ B, const size_t ldb, T4 beta, T5 *  __restrict__ C, const size_t ldc, OPA opa, OPB opb);
        
        // diagonal matrix * transpose(dense matrix)
        template <size_t TILE_DIM, size_t BLOCK_ROWS, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void dgm_dm_mt(const size_t m, const size_t n, const size_t k, T1 alpha, const T2 *  __restrict__ A, const size_t inca, const T3 *  __restrict__ B, const size_t ldb, T4 beta, T5 *  __restrict__ C, const size_t ldc, OPA opa, OPB opb);

        
        // dense matrix * diagonal matrix
        template <size_t TILE_DIM, size_t BLOCK_ROWS, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void dm_dgm_m(const size_t m, const size_t n, const size_t k, T1 alpha, const T2 *  __restrict__ A, const size_t lda, const T3 *  __restrict__ B, const size_t incb, T4 beta, T5 *  __restrict__ C, const size_t ldc, OPA opa, OPB opb);

        // transpose(dense matrix) * diagonal matrix
        template <size_t TILE_DIM, size_t BLOCK_ROWS, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void dm_dgm_mt(const size_t m, const size_t n, const size_t k, T1 alpha, const T2 *  __restrict__ A, const size_t lda, const T3 *  __restrict__ B, const size_t incb, T4 beta, T5 *  __restrict__ C, const size_t ldc, OPA opa, OPB opb);        

    } // namespace cuda_kernels

} // namespace linalg

#endif // PYTTN_LINALG_BACKENDS_CUDA_KERNELS_CUH_//
