#ifndef LINALG_ALGEBRA_CUBLAS_WRAPPER_HPP
#define LINALG_ALGEBRA_CUBLAS_WRAPPER_HPP

#ifdef __NVCC__

#include "cuda_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>


static const char *cublasGetErrorName(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        
        default:
            return "<unrecognised cublas error>";
    };
}

static inline void cublas_safe_call(cublasStatus_t err){if(err != CUBLAS_STATUS_SUCCESS){RAISE_EXCEPTION_STR(cublasGetErrorName(err));}}

namespace linalg
{
namespace cublas
{

//now we provide overloaded interface to each of these routines
//overloads for all three different axpy calls
//y = a*x+y
static inline void axpy(cublasHandle_t handle, int N, const float* A, const float* X, int INCX, float* Y, int INCY){CALL_AND_RETHROW(cublas_safe_call(cublasSaxpy(handle, N, A, X, INCX, Y, INCY)));}
static inline void axpy(cublasHandle_t handle, int N, const double* A, const double* X, int INCX, double* Y, int INCY){CALL_AND_RETHROW(cublas_safe_call(cublasDaxpy(handle, N, A, X, INCX, Y, INCY)));}
static inline void axpy(cublasHandle_t handle, int N, const complex<float>* A, const complex<float>* X, int INCX, complex<float>* Y, int INCY){CALL_AND_RETHROW(cublas_safe_call(cublasCaxpy(handle, N, (const cuComplex*)A, (const cuComplex*)X, INCX, (cuComplex*)Y, INCY)));}
static inline void axpy(cublasHandle_t handle, int N, const complex<double>* A, const complex<double>* X, int INCX, complex<double>* Y, int INCY){CALL_AND_RETHROW(cublas_safe_call(cublasZaxpy(handle, N, (const cuDoubleComplex*)A, (const cuDoubleComplex*)X, INCX, (cuDoubleComplex*)Y, INCY)));}

//scalar multiplication - x = a*x 
static inline void scal(cublasHandle_t handle, int N, const float* A, float* X, int INCX){CALL_AND_RETHROW(cublas_safe_call(cublasSscal(handle, N, A, X, INCX)));}
static inline void scal(cublasHandle_t handle, int N, const double* A, double* X, int INCX){CALL_AND_RETHROW(cublas_safe_call(cublasDscal(handle, N, A, X, INCX)));}
static inline void scal(cublasHandle_t handle, int N, const complex<float>* A, complex<float>* X, int INCX){CALL_AND_RETHROW(cublas_safe_call(cublasCscal(handle, N, (const cuComplex*)A, (cuComplex*)X, INCX)));}
static inline void scal(cublasHandle_t handle, int N, const complex<double>* A, complex<double>* X, int INCX){CALL_AND_RETHROW(cublas_safe_call(cublasZscal(handle, N, (const cuDoubleComplex*)A, (cuDoubleComplex*)X, INCX)));}

//overloads of the dot product calls.  Here we also add the abili
static inline float dot(cublasHandle_t handle, bool /* conj */, int N, const float* X, int INCX, const float* Y, int INCY)
{
    float result;
    CALL_AND_RETHROW(cublas_safe_call(cublasSdot(handle, N, X, INCX, Y, INCY, &result)));
    return result;
}
static inline double dot(cublasHandle_t handle, bool /* conj */, int N, const double* X, int INCX, const double* Y, int INCY)
{
    double result;
    CALL_AND_RETHROW(cublas_safe_call(cublasDdot(handle, N, X, INCX, Y, INCY, &result)));
    return result;
}
static inline complex<float> dot(cublasHandle_t handle, bool conj, int N, const complex<float>* X, int INCX, const complex<float>* Y, int INCY)
{
    complex<float> result;
    if(conj){CALL_AND_RETHROW(cublas_safe_call(cublasCdotc(handle, N, (const cuComplex*)X, INCX, (const cuComplex*)Y, INCY, (cuComplex*)&result)));}
    else{CALL_AND_RETHROW(cublas_safe_call(cublasCdotu(handle, N, (const cuComplex*)X, INCX, (const cuComplex*)Y, INCY, (cuComplex*)&result)));}
    return result;
}
static inline complex<double> dot(cublasHandle_t handle, bool conj, int N, const complex<double>* X, int INCX, const complex<double>* Y, int INCY)
{
    complex<double> result;
    if(conj){CALL_AND_RETHROW(cublas_safe_call(cublasZdotc(handle, N, (const cuDoubleComplex*)X, INCX, (const cuDoubleComplex*)Y, INCY, (cuDoubleComplex*)&result)));}
    else{CALL_AND_RETHROW(cublas_safe_call(cublasZdotu(handle, N, (const cuDoubleComplex*)X, INCX, (const cuDoubleComplex*)Y, INCY, (cuDoubleComplex*)&result)));} 
    return result;
}

//overloads of matrix matrix product calls
static inline void gemm(cublasHandle_t handle, cublasOperation_t TRANSA, cublasOperation_t TRANSB, int M, int N, int K, const float* ALPHA, const float* A, int LDA, const float* B, int LDB, const float* BETA, float* C, int LDC)
{CALL_AND_RETHROW(cublas_safe_call(cublasSgemm(handle, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)));}
static inline void gemm(cublasHandle_t handle, cublasOperation_t TRANSA, cublasOperation_t TRANSB, int M, int N, int K, const double* ALPHA, const double* A, int LDA, const double* B, int LDB, const double* BETA, double* C, int LDC)
{CALL_AND_RETHROW(cublas_safe_call(cublasDgemm(handle, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)));}
static inline void gemm(cublasHandle_t handle, cublasOperation_t TRANSA, cublasOperation_t TRANSB, int M, int N, int K, const complex<float>* ALPHA, const complex<float>* A, int LDA, const complex<float>* B, int LDB, const complex<float>* BETA, complex<float>* C, int LDC)
{CALL_AND_RETHROW(cublas_safe_call(cublasCgemm(handle, TRANSA, TRANSB, M, N, K, (const cuComplex*)ALPHA, (const cuComplex*)A, LDA, (const cuComplex*)B, LDB,(const cuComplex*)BETA, (cuComplex*)C, LDC)));}
static inline void gemm(cublasHandle_t handle, cublasOperation_t TRANSA, cublasOperation_t TRANSB, int M, int N, int K, const complex<double>* ALPHA, const complex<double>* A, int LDA, const complex<double>* B, int LDB, const complex<double>* BETA, complex<double>* C, int LDC)
{CALL_AND_RETHROW(cublas_safe_call(cublasZgemm(handle, TRANSA, TRANSB, M, N, K, (const cuDoubleComplex*)ALPHA, (const cuDoubleComplex*)A, LDA, (const cuDoubleComplex*)B, LDB, (const cuDoubleComplex*)BETA, (cuDoubleComplex*)C, LDC)));}

//overloads of matrix vector product calls
static inline void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy)
{CALL_AND_RETHROW(cublas_safe_call(cublasSgemv(handle,  trans, m, n, alpha, A, lda, x, incx, beta, y, incy)));}
static inline void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy)
{CALL_AND_RETHROW(cublas_safe_call(cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)));}
static inline void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const complex<float>* alpha, const complex<float>* A, int lda, const complex<float>* x, int incx, const complex<float>* beta, complex<float>* y, int incy)
{CALL_AND_RETHROW(cublas_safe_call(cublasCgemv(handle, trans, m, n, (const cuComplex*)alpha, (const cuComplex*)A, lda, (const cuComplex*)x, incx, (const cuComplex*)beta, (cuComplex*)y, incy)));}
static inline void gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const complex<double>* alpha, const complex<double>* A, int lda, const complex<double>* x, int incx, const complex<double>* beta, complex<double>* y, int incy)
{CALL_AND_RETHROW(cublas_safe_call(cublasZgemv(handle, trans, m, n, (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)A, lda, (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy)));}


//overloads of batched matrix matrix products
static inline void batched_gemm(cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k, const float* alpha, const float* A, 
                            int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, 
                            long long int strideC, int batchCount)
{CALL_AND_RETHROW(cublas_safe_call(cublasSgemmStridedBatched(handle, opA, opB, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)));}

static inline void batched_gemm(cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k, const double* alpha, const double* A, 
                            int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, 
                            long long int strideC, int batchCount)
{CALL_AND_RETHROW(cublas_safe_call(cublasDgemmStridedBatched(handle, opA, opB, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount)));}

static inline void batched_gemm(cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k, const complex<float>* alpha, const complex<float>* A, 
                            int lda, long long int strideA, const complex<float>* B, int ldb, long long int strideB, const complex<float>* beta, complex<float>* C, int ldc, 
                            long long int strideC, int batchCount)
{CALL_AND_RETHROW(cublas_safe_call(cublasCgemmStridedBatched(handle, opA, opB, m, n, k, (const cuComplex*)alpha, (const cuComplex*)A, lda, strideA, (const cuComplex*)B, ldb, strideB, (const cuComplex*)beta, (cuComplex*)C, ldc, strideC, batchCount)));}

static inline void batched_gemm(cublasHandle_t handle, cublasOperation_t opA, cublasOperation_t opB, int m, int n, int k, const complex<double>* alpha, const complex<double>* A, 
                            int lda, long long int strideA, const complex<double>* B, int ldb, long long int strideB, const complex<double>* beta, complex<double>* C, int ldc, 
                            long long int strideC, int batchCount)
{CALL_AND_RETHROW(cublas_safe_call(cublasZgemmStridedBatched(handle, opA, opB, m, n, k, (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)A, lda, strideA, (const cuDoubleComplex*)B, ldb, strideB, (const cuDoubleComplex*)beta, (cuDoubleComplex*)C, ldc, strideC, batchCount)));}


//overloads for the geam call
static inline void geam(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc)
{CALL_AND_RETHROW(cublas_safe_call(cublasSgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc)));}
static inline void geam(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc)
{CALL_AND_RETHROW(cublas_safe_call(cublasDgeam(handle, transA, transB, m, n, alpha, A, lda, beta, B, ldb, C, ldc)));}
static inline void geam(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int m, int n, const complex<float>* alpha, const complex<float>* A, int lda, const complex<float>* beta, const complex<float>* B, int ldb, complex<float>* C, int ldc)
{CALL_AND_RETHROW(cublas_safe_call(cublasCgeam(handle, transA, transB, m, n, (const cuComplex*)alpha, (const cuComplex*)A, lda, (const cuComplex*)beta, (const cuComplex*)B, ldb, (cuComplex*)C, ldc)));}
static inline void geam(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB, int m, int n, const complex<double>* alpha, const complex<double>* A, int lda, const complex<double>* beta, const complex<double>* B, int ldb, complex<double>* C, int ldc)
{CALL_AND_RETHROW(cublas_safe_call(cublasZgeam(handle, transA, transB, m, n, (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)A, lda, (const cuDoubleComplex*)beta, (const cuDoubleComplex*)B, ldb, (cuDoubleComplex*)C, ldc)));}

}   //namespace cublas
}   //namespace linalg

#endif

#endif //LINALG_ALGEBRA_CUBLAS_WRAPPER_HPP//


