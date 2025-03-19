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

#ifndef PYTTN_LINALG_BACKENDS_CUDA_KERNELS_HPP_
#define PYTTN_LINALG_BACKENDS_CUDA_KERNELS_HPP_

// a file containing the kernels used for the linear algebra routines.

#ifdef PYTTN_BUILD_CUDA
namespace linalg
{
    namespace cuda_kernels
    {
        template <typename T>
        static __global__ void addition_assign_array(const T *in, const size_t n, T *out)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                out[tidx] += in[tidx];
            }
        }

        template <typename T>
        static __global__ void subtraction_assign_array(const T *in, const size_t n, T *out)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                out[tidx] -= in[tidx];
            }
        }

        template <typename T>
        static __global__ void copy_real_to_complex_array(const T *in, const size_t n, complex<T> *out)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                out[tidx] = complex<T>(in[tidx], 0.0);
            }
        }

        template <typename T>
        static __global__ void addition_assign_real_to_complex_array(const T *in, const size_t n, complex<T> *out)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                out[tidx] += complex<T>(in[tidx], 0.0);
            }
        }

        template <typename T>
        static __global__ void subtraction_assign_real_to_complex_array(const T *in, const size_t n, complex<T> *out)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                out[tidx] -= complex<T>(in[tidx], 0.0);
            }
        }

        template <typename T>
        static __global__ void fill_matrix_block(const T *src, size_t m1, size_t n1, T *dest, size_t m2, size_t n2)
        {
            unsigned int tidx, tidy;
            unsigned int stridex = blockDim.x * gridDim.x;
            unsigned int stridey = blockDim.y * gridDim.y;

#pragma unroll
            for (tidx = threadIdx.x + blockDim.x * blockIdx.x; tidx < m2; tidx += stridex)
            {
                if (tidx < m1)
                {
                    for (tidy = threadIdx.y + blockDim.y * blockIdx.y; tidy < n2; tidy += stridey)
                    {
                        if (tidy < n1)
                        {
                            dest[tidx * n2 + tidy] = src[tidx * n1 + tidy];
                        }
                        else
                        {
                            dest[tidx * n2 + tidy] = T(0);
                        }
                    }
                }
                else
                {
                    for (tidy = threadIdx.y + blockDim.y * blockIdx.y; tidy < n2; tidy += stridey)
                    {
                        dest[tidx * n2 + tidy] = T(0);
                    }
                }
            }
        }

        template <typename T>
        static __global__ void complex_conjugate(const complex<T> *const in, const size_t n, complex<T> *const out)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                out[tidx] = conj(in[tidx]);
            }
        }

        template <typename T>
        static __global__ void vector_scalar_product(int N, T A, const T *X, int INCX, T *Y, int INCY)
        {

            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

            if (A == T(1.0))
            {
#pragma unroll
                for (; tidx < N; tidx += stride)
                {
                    Y[tidx * INCY] = X[tidx * INCX];
                }
            }
            else
            {
#pragma unroll
                for (; tidx < N; tidx += stride)
                {
                    Y[tidx * INCY] = A * X[tidx * INCX];
                }
            }
        }

        template <typename T>
        static __global__ void axpy_conj(const complex<T> *in, const size_t n, complex<T> *out)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                out[tidx] = conj(in[tidx]);
            }
        }

        template <typename T, typename expr>
        static __global__ void eval_expression_strided_kernel(T *res, size_t n, size_t resstride, expr op)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                res[tidx * resstride] = op[tidx];
            }
        }

        template <typename T, typename expr>
        static __global__ void eval_add_expression_strided_kernel(T *res, size_t n, size_t resstride, expr op)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                res[tidx * resstride] -= op[tidx];
            }
        }

        template <typename T, typename expr>
        static __global__ void eval_sub_expression_strided_kernel(T *res, size_t n, size_t resstride, expr op)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                res[tidx * resstride] += op[tidx];
            }
        }

        template <typename T, typename expr>
        static __global__ void eval_expression_kernel(T *res, size_t n, expr op)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                res[tidx] = op[tidx];
            }
        }

        template <typename T, typename expr>
        static __global__ void eval_add_expression_kernel(T *res, size_t n, expr op)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                res[tidx] += op[tidx];
            }
        }

        template <typename T, typename expr>
        static __global__ void eval_sub_expression_kernel(T *res, size_t n, expr op)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < n; tidx += stride)
            {
                res[tidx] -= op[tidx];
            }
        }

        // kernel for filling an array with a value
        template <typename T>
        __global__ void fill_n(T *devPtr, const size_t m, const T val)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < m; tidx += stride)
            {
                devPtr[tidx] = val;
            }
        }

        template <typename T>
        __global__ void fill_n_strided(T *devPtr, const size_t m, const size_t inc, const T val)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < m; tidx += stride)
            {
                devPtr[tidx * inc] = val;
            }
        }

        template <typename T, typename Func, typename... Args>
        __global__ void fill_func_1(T *devPtr, const size_t m, Func f, Args... args)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

#pragma unroll
            for (; tidx < m; tidx += stride)
            {
                devPtr[tidx] = f(tidx, args...);
            }
        }

        template <typename T, typename Func, typename... Args>
        __global__ void fill_func_2(T *devPtr, const size_t m, const size_t n, Func f, Args... args)
        {
            unsigned int tidx, tidy;
            unsigned int stridex = blockDim.x * gridDim.x;
            unsigned int stridey = blockDim.y * gridDim.y;

#pragma unroll
            for (tidx = threadIdx.x + blockDim.x * blockIdx.x; tidx < m; tidx += stridex)
            {
                for (tidy = threadIdx.y + blockDim.y * blockIdx.y; tidy < n; tidy += stridey)
                {
                    devPtr[tidx * n + tidy] = f(tidx, tidy, args...);
                }
            }
        }

        template <typename T, typename Func, typename... Args>
        __global__ void fill_func_3(T *devPtr, const size_t m, const size_t n, const size_t o, Func f, Args... args)
        {
            unsigned int tidx, tidy, tidz;
            unsigned int stridex = blockDim.x * gridDim.x;
            unsigned int stridey = blockDim.y * gridDim.y;
            unsigned int stridez = blockDim.z * gridDim.z;

#pragma unroll
            for (tidx = threadIdx.x + blockDim.x * blockIdx.x; tidx < m; tidx += stridex)
            {
                for (tidy = threadIdx.y + blockDim.y * blockIdx.y; tidy < n; tidy += stridey)
                {
                    for (tidz = threadIdx.z + blockDim.z * blockIdx.z; tidz < o; tidz += stridez)
                    {
                        devPtr[(tidx * n + tidy) * o + tidz] = f(tidx, tidy, tidz, args...);
                    }
                }
            }
        }

        template <typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void elementwise_multiplication(const size_t m, const size_t n, T1 alpha, const T2 *A, const T3 *B, T4 beta, T5 *C, OPA opa, OPB opb)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

            if (beta != T4(0.0))
            {
                if (alpha != T1(1.0))
                {
#pragma unroll
                    for (; tidx < n; tidx += stride)
                    {
                        if (tidx < m)
                        {
                            C[tidx] = alpha * opa(A[tidx]) * opb(B[tidx]) + beta * C[tidx];
                        }
                        else
                        {
                            C[tidx] = T5(0.0);
                        }
                    }
                }
                else
                {
#pragma unroll
                    for (; tidx < n; tidx += stride)
                    {
                        if (tidx < m)
                        {
                            C[tidx] = opa(A[tidx]) * opb(B[tidx]) + beta * C[tidx];
                        }
                        else
                        {
                            C[tidx] = T5(0.0);
                        }
                    }
                }
            }
            else
            {
                if (alpha != T1(1.0))
                {
#pragma unroll
                    for (; tidx < n; tidx += stride)
                    {
                        if (tidx < m)
                        {
                            C[tidx] = alpha * opa(A[tidx]) * opb(B[tidx]);
                        }
                        else
                        {
                            C[tidx] = T5(0.0);
                        }
                    }
                }
                else
                {
#pragma unroll
                    for (; tidx < n; tidx += stride)
                    {
                        if (tidx < m)
                        {
                            C[tidx] = opa(A[tidx]) * opb(B[tidx]);
                        }
                        else
                        {
                            C[tidx] = T5(0.0);
                        }
                    }
                }
            }
        }

        template <typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void elementwise_multiplication_strided(const size_t m, const size_t n, T1 alpha, const T2 *A, const size_t inca, const T3 *B, const size_t incb, T4 beta, T5 *C, const size_t incc, OPA opa, OPB opb)
        {
            unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
            unsigned int stride = blockDim.x * gridDim.x;

            if (beta != T4(0.0))
            {
                if (alpha != T1(1.0))
                {
#pragma unroll
                    for (; tidx < n; tidx += stride)
                    {
                        if (tidx < m)
                        {
                            C[tidx * incc] = alpha * opa(A[tidx * inca]) * opb(B[tidx * incb]) + beta * C[tidx * incc];
                        }
                        else
                        {
                            C[tidx * incc] = T5(0.0);
                        }
                    }
                }
                else
                {
#pragma unroll
                    for (; tidx < n; tidx += stride)
                    {
                        if (tidx < m)
                        {
                            C[tidx * incc] = opa(A[tidx * inca]) * opb(B[tidx * incb]) + beta * C[tidx * incc];
                        }
                        else
                        {
                            C[tidx * incc] = T5(0.0);
                        }
                    }
                }
            }
            else
            {
                if (alpha != T1(1.0))
                {
#pragma unroll
                    for (; tidx < n; tidx += stride)
                    {
                        if (tidx < m)
                        {
                            C[tidx * incc] = alpha * opa(A[tidx * inca]) * opb(B[tidx * incb]);
                        }
                        else
                        {
                            C[tidx * incc] = T5(0.0);
                        }
                    }
                }
                else
                {
#pragma unroll
                    for (; tidx < n; tidx += stride)
                    {
                        if (tidx < m)
                        {
                            C[tidx * incc] = opa(A[tidx * inca]) * opb(B[tidx * incb]);
                        }
                        else
                        {
                            C[tidx * incc] = T5(0.0);
                        }
                    }
                }
            }
        }

        ////
        // Functions implementing diagonal matrix - dense matrix multiplication
        // at a later date it might be worth implementing these kernels using rectangular tiles to allow better handling of rectangular matrices (as are often present
        // in the ttns integrator).
        ////
        // diagonal matrix * dense matrix
        template <size_t TILE_DIM, size_t BLOCK_ROWS, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void dgm_dm_m(const size_t m, const size_t n, const size_t k, T1 alpha, const T2 *A, const size_t inca, const T3 *B, const size_t ldb, T4 beta, T5 *C, const size_t ldc, OPA opa, OPB opb)
        {
            size_t min_km = m > k ? k : m;
            size_t x = blockIdx.x * TILE_DIM + threadIdx.x; // col
            size_t y = blockIdx.y * TILE_DIM + threadIdx.y; // row

            size_t sstr = TILE_DIM;
            size_t bskip = TILE_DIM * sstr;
            extern __shared__ T5 local_buffer[];
            T3 *blocal = reinterpret_cast<T3 *>(local_buffer);
            decltype(T1() * T2()) *alocal = reinterpret_cast<decltype(T1() * T2()) *>(local_buffer + bskip);

//__shared__ T2 alocal[TILE_DIM];
//__shared__ T3 blocal[TILE_DIM*TILE_DIM];

// read the B array to shared memory
#pragma unroll
            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            {
                if (y + j < min_km && x < n)
                {
                    blocal[(threadIdx.y + j) * sstr + threadIdx.x] = B[(y + j) * ldb + x];
                }
            }
            // read the A array to shared memory
            if (threadIdx.x == 0)
            {
                if (alpha != T1(1.0))
                {
                    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                    {
                        if (y + j < min_km)
                        {
                            alocal[threadIdx.y + j] = alpha * opa(A[(y + j) * inca]);
                        }
                    }
                }
                else
                {
                    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                    {
                        if (y + j < min_km)
                        {
                            alocal[threadIdx.y + j] = opa(A[(y + j) * inca]);
                        }
                    }
                }
            }

            __syncthreads();

            if (beta != T4(0.0))
            {
                if (x < n)
                {
#pragma unroll
                    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                    {
                        if (y + j < min_km)
                        {
                            C[(y + j) * ldc + x] = alocal[threadIdx.y + j] * opb(blocal[(threadIdx.y + j) * sstr + threadIdx.x]) + beta * C[(y + j) * ldc + x];
                        }
                        else if (y + j < m)
                        {
                            C[(y + j) * ldc + x] *= beta;
                        }
                    }
                }
            }
            else
            {
                if (x < n)
                {
#pragma unroll
                    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                    {
                        if (y + j < min_km)
                        {
                            C[(y + j) * ldc + x] = alocal[threadIdx.y + j] * opb(blocal[(threadIdx.y + j) * sstr + threadIdx.x]);
                        }
                        else if (y + j < m)
                        {
                            C[(y + j) * ldc + x] = T5(0.0);
                        }
                    }
                }
            }
        }

        // diagonal matrix * transpose(dense matrix)
        template <size_t TILE_DIM, size_t BLOCK_ROWS, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void dgm_dm_mt(const size_t m, const size_t n, const size_t k, T1 alpha, const T2 *A, const size_t inca, const T3 *B, const size_t ldb, T4 beta, T5 *C, const size_t ldc, OPA opa, OPB opb)
        {
            size_t min_km = m > k ? k : m;

            size_t x = blockIdx.x * TILE_DIM + threadIdx.x; // col
            size_t y = blockIdx.y * TILE_DIM + threadIdx.y; // row

            size_t sstr = TILE_DIM + 1;
            size_t bskip = TILE_DIM * sstr;
            extern __shared__ T5 local_buffer[];
            T3 *blocal = reinterpret_cast<T3 *>(local_buffer);
            decltype(T1() * T2()) *alocal = reinterpret_cast<decltype(T1() * T2()) *>(local_buffer + bskip);

//__shared__ T2 alocal[TILE_DIM];
//__shared__ T3 blocal[TILE_DIM*(TILE_DIM+1)];

// read the B array to shared memory
#pragma unroll
            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            {
                if (y + j < n && x < k)
                {
                    blocal[(threadIdx.y + j) * sstr + threadIdx.x] = B[(y + j) * ldb + x];
                }
            }
            // read the A array to shared memory
            if (threadIdx.y == 0)
            {
                if (alpha != T1(1.0))
                {
                    if (x < min_km)
                    {
                        alocal[threadIdx.x] = alpha * opa(A[x * inca]);
                    }
                }
                else
                {
                    if (x < min_km)
                    {
                        alocal[threadIdx.x] = opa(A[x * inca]);
                    }
                }
            }

            __syncthreads();
            x = blockIdx.y * TILE_DIM + threadIdx.x; // col
            y = blockIdx.x * TILE_DIM + threadIdx.y; // row
            if (beta != T4(0.0))
            {
                if (x < n)
                {
#pragma unroll
                    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                    {
                        if (y + j < min_km)
                        {
                            C[(y + j) * ldc + x] = alocal[threadIdx.y + j] * opb(blocal[threadIdx.x * sstr + threadIdx.y + j]) + beta * C[(y + j) * ldc + x];
                        }
                        else if (y + j < m)
                        {
                            C[(y + j) * ldc + x] *= beta;
                        }
                    }
                }
            }
            else
            {
                if (x < n)
                {
#pragma unroll
                    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                    {
                        if (y + j < min_km)
                        {
                            C[(y + j) * ldc + x] = alocal[threadIdx.y + j] * opb(blocal[threadIdx.x * sstr + threadIdx.y + j]);
                        }
                        else if (y + j < m)
                        {
                            C[(y + j) * ldc + x] = T5(0.0);
                        }
                    }
                }
            }
        }

        // dense matrix * diagonal matrix
        template <size_t TILE_DIM, size_t BLOCK_ROWS, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void dm_dgm_m(const size_t m, const size_t n, const size_t k, T1 alpha, const T2 *A, const size_t lda, const T3 *B, const size_t incb, T4 beta, T5 *C, const size_t ldc, OPA opa, OPB opb)
        {
            size_t min_kn = n > k ? k : n;
            size_t x = blockIdx.x * TILE_DIM + threadIdx.x; // col
            size_t y = blockIdx.y * TILE_DIM + threadIdx.y; // row

            size_t sstr = TILE_DIM;
            size_t askip = TILE_DIM * sstr;
            extern __shared__ T5 local_buffer[];
            T2 *alocal = reinterpret_cast<T2 *>(local_buffer);
            decltype(T3() * T1()) *blocal = reinterpret_cast<decltype(T3() * T1()) *>(local_buffer + askip);

//__shared__ T2 alocal[TILE_DIM*TILE_DIM];
//__shared__ T3 blocal[TILE_DIM];

// read the A array to shared memory
#pragma unroll
            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            {
                if (y + j < m && x < min_kn)
                {
                    alocal[(threadIdx.y + j) * sstr + threadIdx.x] = A[(y + j) * lda + x];
                }
            }

            // read the B array to shared memory
            if (threadIdx.y == 0)
            {
                if (alpha != T1(1.0))
                {
                    if (x < min_kn)
                    {
                        blocal[threadIdx.x] = alpha * opb(B[x * incb]);
                    }
                }
                else
                {
                    if (x < min_kn)
                    {
                        blocal[threadIdx.x] = opb(B[x * incb]);
                    }
                }
            }

            __syncthreads();

            if (beta != T4(0.0))
            {
#pragma unroll
                for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                {
                    if (y + j < m)
                    {
                        if (x < min_kn)
                        {
                            C[(y + j) * ldc + x] = blocal[threadIdx.x] * opa(alocal[(threadIdx.y + j) * sstr + threadIdx.x]) + beta * C[(y + j) * ldc + x];
                        }
                        else if (x < n)
                        {
                            C[(y + j) * ldc + x] *= beta;
                        }
                    }
                }
            }
            else
            {
#pragma unroll
                for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                {
                    if (y + j < m)
                    {
                        if (x < min_kn)
                        {
                            C[(y + j) * ldc + x] = blocal[threadIdx.x] * opa(alocal[(threadIdx.y + j) * sstr + threadIdx.x]);
                        }
                        else if (x < n)
                        {
                            C[(y + j) * ldc + x] = T5(0.0);
                        }
                    }
                }
            }
        }

        // transpose(dense matrix) * diagonal matrix
        template <size_t TILE_DIM, size_t BLOCK_ROWS, typename OPA, typename OPB, typename T1, typename T2, typename T3, typename T4, typename T5>
        __global__ void dm_dgm_mt(const size_t m, const size_t n, const size_t k, T1 alpha, const T2 *A, const size_t lda, const T3 *B, const size_t incb, T4 beta, T5 *C, const size_t ldc, OPA opa, OPB opb)
        {
            size_t min_kn = n > k ? k : n;
            size_t x = blockIdx.x * TILE_DIM + threadIdx.x; // col
            size_t y = blockIdx.y * TILE_DIM + threadIdx.y; // row

            size_t sstr = TILE_DIM + 1;
            size_t askip = TILE_DIM * sstr;
            extern __shared__ T5 local_buffer[];
            T2 *alocal = reinterpret_cast<T2 *>(local_buffer);
            decltype(T3() * T1()) *blocal = reinterpret_cast<decltype(T3() * T1()) *>(local_buffer + askip);

//__shared__ T2 alocal[TILE_DIM*(TILE_DIM+1)];
//__shared__ T3 blocal[TILE_DIM];

// read the A array to shared memory
#pragma unroll
            for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
            {
                if (y + j < min_kn && x < m)
                {
                    alocal[(threadIdx.y + j) * sstr + threadIdx.x] = A[(y + j) * lda + x];
                }
            }

            if (threadIdx.x == 0)
            {
                if (alpha != T1(1.0))
                {
                    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                    {
                        if (y + j < min_kn)
                        {
                            blocal[threadIdx.y + j] = alpha * opb(B[(y + j) * incb]);
                        }
                    }
                }
                else
                {
                    for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                    {
                        if (y + j < min_kn)
                        {
                            blocal[threadIdx.y + j] = opb(B[(y + j) * incb]);
                        }
                    }
                }
            }
            // read the B array to shared memory

            __syncthreads();
            x = blockIdx.y * TILE_DIM + threadIdx.x; // col
            y = blockIdx.x * TILE_DIM + threadIdx.y; // row

            if (beta != T4(0.0))
            {
#pragma unroll
                for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                {
                    if (y + j < m)
                    {
                        if (x < min_kn)
                        {
                            C[(y + j) * ldc + x] = blocal[threadIdx.x] * opa(alocal[threadIdx.x * sstr + threadIdx.y + j]) + beta * C[(y + j) * ldc + x];
                        }
                        else if (x < n)
                        {
                            C[(y + j) * ldc + x] *= beta;
                        }
                    }
                }
            }
            else
            {
#pragma unroll
                for (size_t j = 0; j < TILE_DIM; j += BLOCK_ROWS)
                {
                    if (y + j < m)
                    {
                        if (x < min_kn)
                        {
                            C[(y + j) * ldc + x] = blocal[threadIdx.x] * opa(alocal[threadIdx.x * sstr + threadIdx.y + j]);
                        }
                        else if (x < n)
                        {
                            C[(y + j) * ldc + x] = T5(0.0);
                        }
                    }
                }
            }
        }

    } // namespace cuda_kernels

} // namespace linalg

#endif

#endif // PYTTN_LINALG_BACKENDS_CUDA_KERNELS_HPP_//
