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

#ifndef PYTTN_LINALG_BACKENDS_HPP_
#define PYTTN_LINALG_BACKENDS_HPP_


namespace linalg
{
    class backend_base{};

    template <typename backend>
    class backend_algebra
    {
    public:
        using size_type = typename traits<backend>::size_type;
        using int_type = typename traits<backend>::int_type;
        using index_type = typename traits<backend>::index_type;
        using transform_type = typename traits<backend>::transform_type;

    public:
        template <typename F, typename... Args> static void async_for(size_type start, size_type end, F &&f, Args &&...args);
        template <typename T> static bool is_equal(const T *a, const T *b, size_type N);
        template <typename T> static void vector_scalar_product(int_type N, T A, const T *X, int_type INCX, T *Y, int_type INCY);
        template <typename T> static T trace(int_type N, const T *X, int_type INCX);
        template <typename T> static void axpy(int_type N, T A, const T *X, int_type INCX, T *Y, int_type INCY);
        template <typename T> static void scal(int_type N, T A, T *X, int_type INCX); 
        template <typename T> static T dot(bool conj, int_type N, const T *X, int_type INCX, const T *Y, int_type INCY);    
        template <typename T> static void gemm(transform_type TRANSA, transform_type TRANSB, int_type M, int_type N, int_type K, T ALPHA, const T *A, int_type LDA, const T *B, int_type LDB, T BETA, T *C, int_type LDC);
        template <typename T> static void gemv(transform_type trans, int_type m, int_type n, T alpha, const T *A, int_type lda, const T *x, int_type incx, T beta, T *y, int_type incy);
        template <typename T> static void batched_gemm(transform_type opA, transform_type opB, int_type m, int_type n, int_type k, T alpha, const T *A, int_type lda, long long int strideA, const T *B, int_type ldb, long long int strideB, T beta, T *C, int_type ldc, long long int strideC, int_type batchCount);        
        template <typename T> static void outer_contract(transform_type opA, transform_type opB, int_type m, int_type n, int_type k, T alpha, const T *A,
                                          int_type lda, long long int strideA, const T *B, int_type ldb, long long int strideB, T beta, T *C, int_type ldc,
                                          long long int strideC, int_type batchCount, T *res);
        template <typename T1, typename T2, typename T3, typename T4, typename T5> static void dgmv(bool conjA, bool conjB, int_type m, int_type n, T1 alpha, const T2 *A, int_type inca, const T3 *X, int_type incx, T4 beta, T5 *Y, int_type incy);
        template <typename T1, typename T2, typename T3, typename T4, typename T5> static void dgmm(bool sparse_left, transform_type opA, transform_type opB, size_type m, size_type n, size_type k, T1 alpha, const T2 *A, size_type inca, const T3 *B, size_type ldb, T4 beta, T5 *C, size_type ldc);
        template <typename T> static void copy_matrix_subblock(size_type m1, size_type n1, const T *src, size_type lda, T *dest, size_type ldb); 
        template <typename T, size_t D> static void set_tensor_block(const T* src, const std::array<size_type, D>& src_dims, T* dest, const std::array<size_type, D>& dest_dims, const std::array<size_type, D>& skip);
        template <typename T> static void copy(const T *src, size_type n, T *dest);
    };


    template <typename T, typename backend>
    struct device_type
    {
        using type = T;
    };
    
    //extern template class backend_algebra<cuda_backend>;

} // namespace linalg

#endif
