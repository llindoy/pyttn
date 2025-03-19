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

#ifndef PYTTN_LINALG_BACKENDS_CUSOLVER_WRAPPER_HPP_
#define PYTTN_LINALG_BACKENDS_CUSOLVER_WRAPPER_HPP_

#ifdef PYTTN_BUILD_CUDA

#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>

static const char *cusolverGetErrorName(cusolverStatus_t error)
{
    switch (error)
    {
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:
        return "<unrecognised cusolver error>";
    };
}

static inline void cusolver_safe_call(cusolverStatus_t err)
{
    if (err != CUSOLVER_STATUS_SUCCESS)
    {
        RAISE_EXCEPTION_STR(cusolverGetErrorName(err));
    }
}

namespace linalg
{
    namespace cusolver
    {

        static inline void getrf_error_handling(int INFO, const char type)
        {
            if (INFO < 0)
            {
                std::ostringstream oss;
                if (INFO < 0)
                {
                    oss << type << "getrf call failed.  The argument at the " << -INFO << " position is invalid.";
                    RAISE_EXCEPTION_MESSSTR("Failed to compute the LU decomposition of a matrix.  ", oss.str());
                }
                else
                {
                    oss << type << "getrf call failed.  U(i, i) is exactly zero for i = " << INFO << ".  The factorisation has been completed but the factor U is exactly singular.";
                    RAISE_NUMERIC_MESSSTR("Failed to compute the LU decomposition of a matrix.  ", oss.str());
                }
            }
        }

        static void getrf(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *work, int *ipiv, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnSgetrf(handle, m, n, A, lda, work, ipiv, devinfo)));
        }
        static void getrf(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *work, int *ipiv, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnDgetrf(handle, m, n, A, lda, work, ipiv, devinfo)));
        }
        static void getrf(cusolverDnHandle_t handle, int m, int n, complex<float> *A, int lda, complex<float> *work, int *ipiv, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnCgetrf(handle, m, n, (cuComplex *)A, lda, (cuComplex *)work, ipiv, devinfo)));
        }
        static void getrf(cusolverDnHandle_t handle, int m, int n, complex<double> *A, int lda, complex<double> *work, int *ipiv, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnZgetrf(handle, m, n, (cuDoubleComplex *)A, lda, (cuDoubleComplex *)work, ipiv, devinfo)));
        }

        static void getrf_buffersize(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *lwork)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, lwork)));
        }
        static void getrf_buffersize(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *lwork)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, lwork)));
        }
        static void getrf_buffersize(cusolverDnHandle_t handle, int m, int n, complex<float> *A, int lda, int *lwork)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnCgetrf_bufferSize(handle, m, n, (cuComplex *)A, lda, lwork)));
        }
        static void getrf_buffersize(cusolverDnHandle_t handle, int m, int n, complex<double> *A, int lda, int *lwork)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnZgetrf_bufferSize(handle, m, n, (cuDoubleComplex *)A, lda, lwork)));
        }

        static inline void heev_error_handling(int INFO, const char type)
        {
            if (INFO != 0)
            {
                std::string method_name;
                if (type == 's' || type == 'd')
                {
                    method_name = "syev";
                }
                else
                {
                    method_name = "heev";
                }
                std::ostringstream oss;
                if (INFO < 0)
                {
                    oss << type << method_name << "call failed.  The argument at the " << -INFO << " position is invalid.";
                    RAISE_EXCEPTION_MESSSTR("Failed to compute the eigenvalues of a hermitian (or real symmetric matrix).  ", oss.str());
                }
                else
                {
                    oss << type << method_name << "call failed.  The algorithm failed to converge.  " << INFO << " off-diagonal elements of an intermediate tridiagonal form did not converge to zero.";
                    RAISE_NUMERIC_MESSSTR("Failed to compute the eigenvalues of a hermitian (or real symmetric matrix).  ", oss.str());
                }
            }
        }

        static void heev(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *A, int lda, float *W, float *work, int lwork, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devinfo)));
        }
        static void heev(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *A, int lda, double *W, double *work, int lwork, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devinfo)));
        }
        static void heev(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, complex<float> *A, int lda, float *W, complex<float> *work, int lwork, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnCheevd(handle, jobz, uplo, n, (cuComplex *)A, lda, W, (cuComplex *)work, lwork, devinfo)));
        }
        static void heev(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, complex<double> *A, int lda, double *W, complex<double> *work, int lwork, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnZheevd(handle, jobz, uplo, n, (cuDoubleComplex *)A, lda, W, (cuDoubleComplex *)work, lwork, devinfo)));
        }

        static void heev_buffersize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *A, int lda, float *W, int *lwork)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)));
        }
        static void heev_buffersize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *A, int lda, double *W, int *lwork)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork)));
        }
        static void heev_buffersize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, complex<float> *A, int lda, float *W, int *lwork)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, (cuComplex *)A, lda, W, lwork)));
        }
        static void heev_buffersize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, complex<double> *A, int lda, double *W, int *lwork)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, (cuDoubleComplex *)A, lda, W, lwork)));
        }

        static inline void gesvd_error_handling(int INFO, const char type)
        {
            if (INFO != 0)
            {
                std::ostringstream oss;
                if (INFO < 0)
                {
                    oss << type << "gesvd call failed.  The argument at the " << -INFO << " position is invalid.";
                    RAISE_EXCEPTION_MESSSTR("Failed to compute the singular values decomposition of a matrix.  ", oss.str());
                }
                else
                {
                    oss << type << "gesvd call failed.  " << INFO << " superdiagonal of an intermediate bidiagonal form did not converge to zero.";
                    RAISE_NUMERIC_MESSSTR("Failed to compute the singular values decomposition of a matrix.  ", oss.str());
                }
            }
        }

        static void gesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobv, const int m, const int n, float *A, const int lda, float *S, float *U, const int ldu, float *VT, const int ldvt, float *work, const int lwork, float *rwork, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnSgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devinfo)));
        }

        static void gesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobv, const int m, const int n, double *A, const int lda, double *S, double *U, const int ldu, double *VT, const int ldvt, double *work, const int lwork, double *rwork, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnDgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devinfo)));
        }

        static void gesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobv, const int m, const int n, complex<float> *A, const int lda, float *S, complex<float> *U, const int ldu, complex<float> *VT, const int ldvt, complex<float> *work, const int lwork, float *rwork, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnCgesvd(handle, jobu, jobv, m, n, (cuComplex *)A, lda, S, (cuComplex *)U, ldu, (cuComplex *)VT, ldvt, (cuComplex *)work, lwork, rwork, devinfo)));
        }

        static void gesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobv, const int m, const int n, complex<double> *A, const int lda, double *S, complex<double> *U, const int ldu, complex<double> *VT, const int ldvt, complex<double> *work, const int lwork, double *rwork, int *devinfo)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnZgesvd(handle, jobu, jobv, m, n, (cuDoubleComplex *)A, lda, S, (cuDoubleComplex *)U, ldu, (cuDoubleComplex *)VT, ldvt, (cuDoubleComplex *)work, lwork, rwork, devinfo)));
        }

        template <typename T>
        struct gesvd_params;
        template <>
        struct gesvd_params<float>
        {
            static inline void buffersize(cusolverDnHandle_t handle, int m, int n, int &lwork) { CALL_AND_RETHROW(cusolver_safe_call(cusolverDnSgesvd_bufferSize(handle, m, n, &lwork))); }
        };
        template <>
        struct gesvd_params<double>
        {
            static inline void buffersize(cusolverDnHandle_t handle, int m, int n, int &lwork) { CALL_AND_RETHROW(cusolver_safe_call(cusolverDnDgesvd_bufferSize(handle, m, n, &lwork))); }
        };
        template <>
        struct gesvd_params<complex<float>>
        {
            static inline void buffersize(cusolverDnHandle_t handle, int m, int n, int &lwork) { CALL_AND_RETHROW(cusolver_safe_call(cusolverDnCgesvd_bufferSize(handle, m, n, &lwork))); }
        };
        template <>
        struct gesvd_params<complex<double>>
        {
            static inline void buffersize(cusolverDnHandle_t handle, int m, int n, int &lwork) { CALL_AND_RETHROW(cusolver_safe_call(cusolverDnZgesvd_bufferSize(handle, m, n, &lwork))); }
        };

        static inline void gesvdj_error_handling(int INFO, const char type)
        {
            if (INFO != 0)
            {
                std::ostringstream oss;
                if (INFO < 0)
                {
                    oss << type << "gesvdj call failed.  The argument at the " << -INFO << " position is invalid.";
                    RAISE_EXCEPTION_MESSSTR("Failed to compute the singular values decomposition of a matrix.  ", oss.str());
                }
                else
                {
                    oss << type << "gesvdj call failed. The jacobi iteration failed to converge to the given tolerance and number of steps";
                    RAISE_NUMERIC_MESSSTR("Failed to compute the singular values decomposition of a matrix.  ", oss.str());
                }
            }
        }

        static void gesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, const int econ, const int m, const int n, float *A, const int lda, float *S, float *U, const int ldu, float *VT, const int ldvt, float *work, const int lwork, int *devinfo, gesvdjInfo_t params)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, devinfo, params)));
        }

        static void gesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, const int econ, const int m, const int n, double *A, const int lda, double *S, double *U, const int ldu, double *VT, const int ldvt, double *work, const int lwork, int *devinfo, gesvdjInfo_t params)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, devinfo, params)));
        }

        static void gesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, const int econ, const int m, const int n, complex<float> *A, const int lda, float *S, complex<float> *U, const int ldu, complex<float> *VT, const int ldvt, complex<float> *work, const int lwork, int *devinfo, gesvdjInfo_t params)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnCgesvdj(handle, jobz, econ, m, n, (cuComplex *)A, lda, S, (cuComplex *)U, ldu, (cuComplex *)VT, ldvt, (cuComplex *)work, lwork, devinfo, params)));
        }

        static void gesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, const int econ, const int m, const int n, complex<double> *A, const int lda, double *S, complex<double> *U, const int ldu, complex<double> *VT, const int ldvt, complex<double> *work, const int lwork, int *devinfo, gesvdjInfo_t params)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnZgesvdj(handle, jobz, econ, m, n, (cuDoubleComplex *)A, lda, S, (cuDoubleComplex *)U, ldu, (cuDoubleComplex *)VT, ldvt, (cuDoubleComplex *)work, lwork, devinfo, params)));
        }

        // get buffer size of gesvdj
        static void gesvdj_buffersize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, const int econ, const int m, const int n, float *A, const int lda, float *S, float *U, const int ldu, float *VT, const int ldvt, int &lwork, gesvdjInfo_t params)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, VT, ldvt, &lwork, params)));
        }

        static void gesvdj_buffersize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, const int econ, const int m, const int n, double *A, const int lda, double *S, double *U, const int ldu, double *VT, const int ldvt, int &lwork, gesvdjInfo_t params)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, VT, ldvt, &lwork, params)));
        }

        static void gesvdj_buffersize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, const int econ, const int m, const int n, complex<float> *A, const int lda, float *S, complex<float> *U, const int ldu, complex<float> *VT, const int ldvt, int &lwork, gesvdjInfo_t params)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, (cuComplex *)A, lda, S, (cuComplex *)U, ldu, (cuComplex *)VT, ldvt, &lwork, params)));
        }

        static void gesvdj_buffersize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, const int econ, const int m, const int n, complex<double> *A, const int lda, double *S, complex<double> *U, const int ldu, complex<double> *VT, const int ldvt, int &lwork, gesvdjInfo_t params)
        {
            CALL_AND_RETHROW(cusolver_safe_call(cusolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, (cuDoubleComplex *)A, lda, S, (cuDoubleComplex *)U, ldu, (cuDoubleComplex *)VT, ldvt, &lwork, params)));
        }

    } // namespace cusolver
} // namespace linalg

#endif

#endif // PYTTN_LINALG_BACKENDS_CUSOLVER_WRAPPER_HPP_//
