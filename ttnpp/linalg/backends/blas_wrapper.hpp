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

#ifndef PYTTN_LINALG_BACKENDS_BLAS_WRAPPER_HPP_
#define PYTTN_LINALG_BACKENDS_BLAS_WRAPPER_HPP_

#include "../utils/linalg_utils.hpp"
#ifdef USE_MKL
#define MKL_Complex8 linalg::complex<float>
#define MKL_Complex16 linalg::complex<double>

#include "mkl_types.h"
#include "mkl.h"
#endif

namespace linalg
{
    namespace blas
    {
#ifdef USE_MKL
        using blas_int_type = MKL_INT;
#else
#ifndef BLAS_64_BIT
        using blas_int_type = int;
#else
        using blas_int_type = int64_t;
#endif
#endif
#ifndef BLAS_HEADER_INCLUDED
        extern "C"
        {
            void saxpy_(const blas_int_type *const N, const float *const A, const float *const X, const blas_int_type *const INCX, float *const Y, const blas_int_type *const INCY);
            void daxpy_(const blas_int_type *const N, const double *const A, const double *const X, const blas_int_type *const INCX, double *const Y, const blas_int_type *const INCY);
            void caxpy_(const blas_int_type *const N, const std::complex<float> *const A, const std::complex<float> *const X, const blas_int_type *const INCX, std::complex<float> *const Y, const blas_int_type *const INCY);
            void zaxpy_(const blas_int_type *const N, const std::complex<double> *const A, const std::complex<double> *const X, const blas_int_type *const INCX, std::complex<double> *const Y, const blas_int_type *const INCY);

            void sscal_(const blas_int_type *const N, const float *const A, const float *const X, const blas_int_type *const INCX);
            void dscal_(const blas_int_type *const N, const double *const A, const double *const X, const blas_int_type *const INCX);
            void cscal_(const blas_int_type *const N, const std::complex<float> *const A, const std::complex<float> *const X, const blas_int_type *const INCX);
            void zscal_(const blas_int_type *const N, const std::complex<double> *const A, const std::complex<double> *const X, const blas_int_type *const INCX);

            void sgemv_(const char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const float *const ALPHA, const float *const A, const blas_int_type *const LDA, const float *const X, const blas_int_type *const INCX, const float *const BETA, float *const Y, const blas_int_type *const INCY);
            void dgemv_(const char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const double *const ALPHA, const double *const A, const blas_int_type *const LDA, const double *const X, const blas_int_type *const INCX, const double *const BETA, double *const Y, const blas_int_type *const INCY);
            void cgemv_(const char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const std::complex<float> *const ALPHA, const std::complex<float> *const A, const blas_int_type *const LDA, const std::complex<float> *const X, const blas_int_type *const INCX, const std::complex<float> *const BETA, std::complex<float> *const Y, const blas_int_type *const INCY);
            void zgemv_(const char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const std::complex<double> *const ALPHA, const std::complex<double> *const A, const blas_int_type *const LDA, const std::complex<double> *const X, const blas_int_type *const INCX, const std::complex<double> *const BETA, std::complex<double> *const Y, const blas_int_type *const INCY);

            void sgemm_(const char *const TRANSA, const char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const float *const ALPHA, const float *const A, const blas_int_type *const LDA, const float *const B, const blas_int_type *const LDB, const float *const BETA, float *const C, const blas_int_type *const LDC);
            void dgemm_(const char *const TRANSA, const char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const double *const ALPHA, const double *const A, const blas_int_type *const LDA, const double *const B, const blas_int_type *const LDB, const double *const BETA, double *const C, const blas_int_type *const LDC);
            void cgemm_(const char *const TRANSA, const char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const std::complex<float> *const ALPHA, const std::complex<float> *const A, const blas_int_type *const LDA, const std::complex<float> *const B, const blas_int_type *const LDB, const std::complex<float> *const BETA, std::complex<float> *const C, const blas_int_type *const LDC);
            void zgemm_(const char *const TRANSA, const char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const std::complex<double> *const ALPHA, const std::complex<double> *const A, const blas_int_type *const LDA, const std::complex<double> *const B, const blas_int_type *const LDB, const std::complex<double> *const BETA, std::complex<double> *const C, const blas_int_type *const LDC);
        } // extern functions for the blas routines required
#endif

#ifndef BLAS_NO_TRAILING_UNDERSCORE
        // now we provide overloaded interface to each of these routines
        // overloads for all three different axpy calls
        // y = a* constx+y
        static inline void axpy(const blas_int_type *const N, const float *const A, const float *const X, const blas_int_type *const INCX, float *const Y, const blas_int_type *const INCY) { saxpy_(N, A, X, INCX, Y, INCY); }
        static inline void axpy(const blas_int_type *const N, const double *const A, const double *const X, const blas_int_type *const INCX, double *const Y, const blas_int_type *const INCY) { daxpy_(N, A, X, INCX, Y, INCY); }
        static inline void axpy(const blas_int_type *const N, const complex<float> *const A, const complex<float> *const X, const blas_int_type *const INCX, complex<float> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<float> *;
            using type = std::complex<float> *;
            caxpy_(N, reinterpret_cast<ctype>(A), reinterpret_cast<ctype>(X), INCX, reinterpret_cast<type>(Y), INCY);
        }
        static inline void axpy(const blas_int_type *const N, const complex<double> *const A, const complex<double> *const X, const blas_int_type *const INCX, complex<double> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<double> *;
            using type = std::complex<double> *;
            zaxpy_(N, reinterpret_cast<ctype>(A), reinterpret_cast<ctype>(X), INCX, reinterpret_cast<type>(Y), INCY);
        }

        // scalar multiplication - x = a* constx
        static inline void scal(const blas_int_type *const N, const float *const A, float *const X, const blas_int_type *const INCX) { sscal_(N, A, X, INCX); }
        static inline void scal(const blas_int_type *const N, const double *const A, double *const X, const blas_int_type *const INCX) { dscal_(N, A, X, INCX); }
        static inline void scal(const blas_int_type *const N, const complex<float> *const A, complex<float> *const X, const blas_int_type *const INCX)
        {
            using ctype = const std::complex<float> *;
            using type = std::complex<float> *;
            cscal_(N, reinterpret_cast<ctype>(A), reinterpret_cast<type>(X), INCX);
        }
        static inline void scal(const blas_int_type *const N, const complex<double> *const A, complex<double> *const X, const blas_int_type *const INCX)
        {
            using ctype = const std::complex<double> *;
            using type = std::complex<double> *;
            zscal_(N, reinterpret_cast<ctype>(A), reinterpret_cast<type>(X), INCX);
        }

        // overloads of matrix vector product calls
        static inline void gemv(char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const float *const ALPHA, const float *const A, const blas_int_type *const LDA, const float *const X, const blas_int_type *const INCX, const float *const BETA, float *const Y, const blas_int_type *const INCY)
        {
            sgemv_(TRANSA, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
        }
        static inline void gemv(char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const double *const ALPHA, const double *const A, const blas_int_type *const LDA, const double *const X, const blas_int_type *const INCX, const double *const BETA, double *const Y, const blas_int_type *const INCY)
        {
            dgemv_(TRANSA, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
        }
        static inline void gemv(char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const complex<float> *const ALPHA, const complex<float> *const A, const blas_int_type *const LDA, const complex<float> *const X, const blas_int_type *const INCX, const complex<float> *const BETA, complex<float> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<float> *;
            using type = std::complex<float> *;
            cgemv_(TRANSA, M, N, reinterpret_cast<ctype>(ALPHA), reinterpret_cast<ctype>(A), LDA, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(BETA), reinterpret_cast<type>(Y), INCY);
        }
        static inline void gemv(char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const complex<double> *const ALPHA, const complex<double> *const A, const blas_int_type *const LDA, const complex<double> *const X, const blas_int_type *const INCX, const complex<double> *const BETA, complex<double> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<double> *;
            using type = std::complex<double> *;
            zgemv_(TRANSA, M, N, reinterpret_cast<ctype>(ALPHA), reinterpret_cast<ctype>(A), LDA, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(BETA), reinterpret_cast<type>(Y), INCY);
        }

        // overloads of matrix matrix product calls
        static inline void gemm(char *const TRANSA, char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const float *const ALPHA, const float *const A, const blas_int_type *const LDA, const float *const B, const blas_int_type *const LDB, const float *const BETA, float *const C, const blas_int_type *const LDC)
        {
            sgemm_(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
        }
        static inline void gemm(char *const TRANSA, char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const double *const ALPHA, const double *const A, const blas_int_type *const LDA, const double *const B, const blas_int_type *const LDB, const double *const BETA, double *const C, const blas_int_type *const LDC)
        {
            dgemm_(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
        }
        static inline void gemm(char *const TRANSA, char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const complex<float> *const ALPHA, const complex<float> *const A, const blas_int_type *const LDA, const complex<float> *const B, const blas_int_type *const LDB, const complex<float> *const BETA, complex<float> *const C, const blas_int_type *const LDC)
        {
            using ctype = const std::complex<float> *;
            using type = std::complex<float> *;
            cgemm_(TRANSA, TRANSB, M, N, K, reinterpret_cast<ctype>(ALPHA), reinterpret_cast<ctype>(A), LDA, reinterpret_cast<ctype>(B), LDB, reinterpret_cast<ctype>(BETA), reinterpret_cast<type>(C), LDC);
        }
        static inline void gemm(char *const TRANSA, char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const complex<double> *const ALPHA, const complex<double> *const A, const blas_int_type *const LDA, const complex<double> *const B, const blas_int_type *const LDB, const complex<double> *const BETA, complex<double> *const C, const blas_int_type *const LDC)
        {
            using ctype = const std::complex<double> *;
            using type = std::complex<double> *;
            zgemm_(TRANSA, TRANSB, M, N, K, reinterpret_cast<ctype>(ALPHA), reinterpret_cast<ctype>(A), LDA, reinterpret_cast<ctype>(B), LDB, reinterpret_cast<ctype>(BETA), reinterpret_cast<type>(C), LDC);
        }

        // now we add some blas like functions for performing additional operations
        // complex conjugation overload
        static inline void conj(int /*N */, const float *const /*X */, int /* INCX */, float *const /* Y */, int /* INCY */) {}
        static inline void conj(int /* N */, const double *const /*  X */, int /*  INCX */, double *const /* Y */, int /* INCY */) {}
        static inline void conj(int N, const complex<float> *const X, int INCX, complex<float> *const Y, int INCY)
        {
            using std::conj;
            for (int i = 0; i < N; ++i)
            {
                Y[i * INCY] = conj(X[i * INCX]);
            }
        }
        static inline void conj(int N, const complex<double> *const X, int INCX, complex<double> *const Y, int INCY)
        {
            using std::conj;
            for (int i = 0; i < N; ++i)
            {
                Y[i * INCY] = conj(X[i * INCX]);
            }
        }

#else

        // now we provide overloaded interface to each of these routines
        // overloads for all three different axpy calls
        // y = a* constx+y
        static inline void axpy(const blas_int_type *const N, const float *const A, const float *const X, const blas_int_type *const INCX, float *const Y, const blas_int_type *const INCY) { saxpy(N, A, X, INCX, Y, INCY); }
        static inline void axpy(const blas_int_type *const N, const double *const A, const double *const X, const blas_int_type *const INCX, double *const Y, const blas_int_type *const INCY) { daxpy(N, A, X, INCX, Y, INCY); }
        static inline void axpy(const blas_int_type *const N, const complex<float> *const A, const complex<float> *const X, const blas_int_type *const INCX, complex<float> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<float> *;
            using type = std::complex<float> *;
            caxpy(N, reinterpret_cast<ctype>(A), reinterpret_cast<ctype>(X), INCX, reinterpret_cast<type>(Y), INCY);
        }
        static inline void axpy(const blas_int_type *const N, const complex<double> *const A, const complex<double> *const X, const blas_int_type *const INCX, complex<double> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<double> *;
            using type = std::complex<double> *;
            zaxpy(N, reinterpret_cast<ctype>(A), reinterpret_cast<ctype>(X), INCX, reinterpret_cast<type>(Y), INCY);
        }

        // scalar multiplication - x = a* constx
        static inline void scal(const blas_int_type *const N, const float *const A, float *const X, const blas_int_type *const INCX) { sscal(N, A, X, INCX); }
        static inline void scal(const blas_int_type *const N, const double *const A, double *const X, const blas_int_type *const INCX) { dscal(N, A, X, INCX); }
        static inline void scal(const blas_int_type *const N, const complex<float> *const A, complex<float> *const X, const blas_int_type *const INCX)
        {
            using ctype = const std::complex<float> *;
            using type = std::complex<float> *;
            cscal(N, reinterpret_cast<ctype>(A), reinterpret_cast<type>(X), INCX);
        }
        static inline void scal(const blas_int_type *const N, const complex<double> *const A, complex<double> *const X, const blas_int_type *const INCX)
        {
            using ctype = const std::complex<double> *;
            using type = std::complex<double> *;
            zscal(N, reinterpret_cast<ctype>(A), reinterpret_cast<type>(X), INCX);
        }

        // overloads of the dot product calls.  Here we also add the abili
        static inline float dot(bool /* conj */, const blas_int_type *const N, const float *const X, const blas_int_type *const INCX, const float *const Y, const blas_int_type *const INCY) { return sdot(N, X, INCX, Y, INCY); }
        static inline double dot(bool /* conj */, const blas_int_type *const N, const double *const X, const blas_int_type *const INCX, const double *const Y, const blas_int_type *const INCY) { return ddot(N, X, INCX, Y, INCY); }

#ifdef NOT_ALLOWS_RETURN_TYPE
        static inline complex<float> dot(bool conj, const blas_int_type *const N, const complex<float> *const X, const blas_int_type *const INCX, const complex<float> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<float> *;
            using cctype = const std::complex<float> *;
            complex<float> ret;
            if (conj)
            {
                cdotc(reinterpret_cast<cctype>(&ret), N, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(Y), INCY);
            }
            else
            {
                cdotu(reinterpret_cast<cctype>(&ret), N, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(Y), INCY);
            }
            return ret;
        }
        static inline complex<double> dot(bool conj, const blas_int_type *const N, const complex<double> *const X, const blas_int_type *const INCX, const complex<double> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<double> *;
            using cctype = const std::complex<double> *;
            complex<double> ret;
            if (conj)
            {
                zdotc(reinterpret_cast<cctype>(&ret), N, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(Y), INCY);
            }
            else
            {
                zdotu(reinterpret_cast<cctype>(&ret), N, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(Y), INCY);
            }
            return ret;
        }
#else
        static inline complex<float> dot(bool conj, const blas_int_type *const N, const complex<float> *const X, const blas_int_type *const INCX, const complex<float> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<float> *;
            if (conj)
            {
                return cdotc(N, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(Y), INCY);
            }
            else
            {
                return cdotu(N, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(Y), INCY);
            }
        }
        static inline complex<double> dot(bool conj, const blas_int_type *const N, const complex<double> *const X, const blas_int_type *const INCX, const complex<double> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<double> *;
            if (conj)
            {
                return zdotc(N, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(Y), INCY);
            }
            else
            {
                return zdotu(N, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(Y), INCY);
            }
        }
#endif

        // overloads of matrix vector product calls
        static inline void gemv(char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const float *const ALPHA, const float *const A, const blas_int_type *const LDA, const float *const X, const blas_int_type *const INCX, const float *const BETA, float *const Y, const blas_int_type *const INCY)
        {
            sgemv(TRANSA, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
        }
        static inline void gemv(char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const double *const ALPHA, const double *const A, const blas_int_type *const LDA, const double *const X, const blas_int_type *const INCX, const double *const BETA, double *const Y, const blas_int_type *const INCY)
        {
            dgemv(TRANSA, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);
        }
        static inline void gemv(char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const complex<float> *const ALPHA, const complex<float> *const A, const blas_int_type *const LDA, const complex<float> *const X, const blas_int_type *const INCX, const complex<float> *const BETA, complex<float> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<float> *;
            using type = std::complex<float> *;
            cgemv(TRANSA, M, N, reinterpret_cast<ctype>(ALPHA), reinterpret_cast<ctype>(A), LDA, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(BETA), reinterpret_cast<type>(Y), INCY);
        }
        static inline void gemv(char *const TRANSA, const blas_int_type *const M, const blas_int_type *const N, const complex<double> *const ALPHA, const complex<double> *const A, const blas_int_type *const LDA, const complex<double> *const X, const blas_int_type *const INCX, const complex<double> *const BETA, complex<double> *const Y, const blas_int_type *const INCY)
        {
            using ctype = const std::complex<double> *;
            using type = std::complex<double> *;
            zgemv(TRANSA, M, N, reinterpret_cast<ctype>(ALPHA), reinterpret_cast<ctype>(A), LDA, reinterpret_cast<ctype>(X), INCX, reinterpret_cast<ctype>(BETA), reinterpret_cast<type>(Y), INCY);
        }

        // overloads of matrix matrix product calls
        static inline void gemm(char *const TRANSA, char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const float *const ALPHA, const float *const A, const blas_int_type *const LDA, const float *const B, const blas_int_type *const LDB, const float *const BETA, float *const C, const blas_int_type *const LDC)
        {
            sgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
        }
        static inline void gemm(char *const TRANSA, char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const double *const ALPHA, const double *const A, const blas_int_type *const LDA, const double *const B, const blas_int_type *const LDB, const double *const BETA, double *const C, const blas_int_type *const LDC)
        {
            dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
        }
        static inline void gemm(char *const TRANSA, char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const complex<float> *const ALPHA, const complex<float> *const A, const blas_int_type *const LDA, const complex<float> *const B, const blas_int_type *const LDB, const complex<float> *const BETA, complex<float> *const C, const blas_int_type *const LDC)
        {
            using ctype = const std::complex<float> *;
            using type = std::complex<float> *;
            cgemm(TRANSA, TRANSB, M, N, K, reinterpret_cast<ctype>(ALPHA), reinterpret_cast<ctype>(A), LDA, reinterpret_cast<ctype>(B), LDB, reinterpret_cast<ctype>(BETA), reinterpret_cast<type>(C), LDC);
        }
        static inline void gemm(char *const TRANSA, char *const TRANSB, const blas_int_type *const M, const blas_int_type *const N, const blas_int_type *const K, const complex<double> *const ALPHA, const complex<double> *const A, const blas_int_type *const LDA, const complex<double> *const B, const blas_int_type *const LDB, const complex<double> *const BETA, complex<double> *const C, const blas_int_type *const LDC)
        {
            using ctype = const std::complex<double> *;
            using type = std::complex<double> *;
            zgemm(TRANSA, TRANSB, M, N, K, reinterpret_cast<ctype>(ALPHA), reinterpret_cast<ctype>(A), LDA, reinterpret_cast<ctype>(B), LDB, reinterpret_cast<ctype>(BETA), reinterpret_cast<type>(C), LDC);
        }

        // now we add some blas like functions for performing additional operations
        // complex conjugation overload
        static inline void conj(int /*N */, const float *const /*X */, int /* INCX */, float *const /* Y */, int /* INCY */) {}
        static inline void conj(int /* N */, const double *const /*  X */, int /*  INCX */, double *const /* Y */, int /* INCY */) {}
        static inline void conj(int N, const complex<float> *const X, int INCX, complex<float> *const Y, int INCY)
        {
            using std::conj;
            for (int i = 0; i < N; ++i)
            {
                Y[i * INCY] = conj(X[i * INCX]);
            }
        }
        static inline void conj(int N, const complex<double> *const X, int INCX, complex<double> *const Y, int INCY)
        {
            using std::conj;
            for (int i = 0; i < N; ++i)
            {
                Y[i * INCY] = conj(X[i * INCX]);
            }
        }

#endif

    } // namespace blas
} // namespace linalg

#endif // PYTTN_LINALG_BACKENDS_BLAS_WRAPPER_HPP_//
