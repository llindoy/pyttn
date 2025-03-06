#ifndef LINALG_ALGEBRA_MKL_WRAPPER_HPP
#define LINALG_ALGEBRA_MKL_WRAPPER_HPP

#include "blas_wrapper.hpp"

#ifdef USE_MKL
    #define MKL_Complex8 linalg::complex<float>
    #define MKL_Complex16 linalg::complex<double>

    #include "mkl_types.h"
    #include "mkl.h"

namespace linalg
{
namespace mkl_extensions
{
using blas_int_type = blas::blas_int_type;

CBLAS_TRANSPOSE get_cblas_transpose(char a)
{
    switch(a)
    {
        case('N'):
            return CblasNoTrans;
        case('n'):
            return CblasNoTrans;
        case('T'):
            return CblasTrans;
        case('t'):
            return CblasTrans;
        case('C'):
            return CblasConjTrans;
        case('c'):
            return CblasConjTrans;
        default:
            RAISE_EXCEPTION("Invalid type");
    };
}

void gemm3m(char TRANSA, char TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, float ALPHA, const float* A, 
                                blas_int_type LDA, const float* B, blas_int_type LDB, float BETA, float* C, blas_int_type LDC)
{
    blas::gemm(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void gemm3m(char TRANSA, char TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, double ALPHA, const double* A, 
                                blas_int_type LDA, const double* B, blas_int_type LDB, double BETA, double* C, blas_int_type LDC)
{
    blas::gemm(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}
                                
void gemm3m(char TRANSA, char TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, complex<float> ALPHA, const complex<float>* A, 
                                blas_int_type LDA, const complex<float>* B, blas_int_type LDB, complex<float> BETA, complex<float>* C, blas_int_type LDC)
{
    cgemm3m(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void gemm3m(char TRANSA, char TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, complex<double> ALPHA, const complex<double>* A, 
                                blas_int_type LDA, const complex<double>* B, blas_int_type LDB, complex<double> BETA, complex<double>* C, blas_int_type LDC)
{
    zgemm3m(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void batched_strided_gemm(char TRANSA, char TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, float ALPHA, const float* A, 
                                blas_int_type LDA, blas_int_type strideA, const float* B, blas_int_type LDB, blas_int_type strideB, float BETA, float* C, blas_int_type LDC, 
                                blas_int_type strideC, blas_int_type batchCount)
{
    CBLAS_TRANSPOSE transa = get_cblas_transpose(TRANSA);
    CBLAS_TRANSPOSE transb = get_cblas_transpose(TRANSB);
    cblas_sgemm_batch_strided(CblasColMajor, transa, transb, M, N, K, ALPHA, A, LDA, strideA, B, LDB, strideB, BETA, C, LDC, strideC, batchCount);
}

void batched_strided_gemm(char TRANSA, char TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, double ALPHA, const double* A, 
                                blas_int_type LDA, blas_int_type strideA, const double* B, blas_int_type LDB, blas_int_type strideB, double BETA, double* C, blas_int_type LDC, 
                                blas_int_type strideC, blas_int_type batchCount)
{
    CBLAS_TRANSPOSE transa = get_cblas_transpose(TRANSA);
    CBLAS_TRANSPOSE transb = get_cblas_transpose(TRANSB);
    cblas_dgemm_batch_strided(CblasColMajor, transa, transb, M, N, K, ALPHA, A, LDA, strideA, B, LDB, strideB, BETA, C, LDC, strideC, batchCount);
}

void batched_strided_gemm(char TRANSA, char TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, complex<float> ALPHA, const complex<float>* A, 
                                blas_int_type LDA, blas_int_type strideA, const complex<float>* B, blas_int_type LDB, blas_int_type strideB, complex<float> BETA, complex<float>* C, blas_int_type LDC, 
                                blas_int_type strideC, blas_int_type batchCount)
{
    CBLAS_TRANSPOSE transa = get_cblas_transpose(TRANSA);
    CBLAS_TRANSPOSE transb = get_cblas_transpose(TRANSB);
    cblas_cgemm_batch_strided(CblasColMajor, transa, transb, M, N, K, reinterpret_cast<const void*>(&ALPHA), reinterpret_cast<const void*>(A), LDA, strideA, reinterpret_cast<const void*>(B), LDB, strideB, reinterpret_cast<const void*>(&BETA), reinterpret_cast<void*>(C), LDC, strideC, batchCount);
}

void batched_strided_gemm(char TRANSA, char TRANSB, blas_int_type M, blas_int_type N, blas_int_type K, complex<double> ALPHA, const complex<double>* A, 
                                blas_int_type LDA, blas_int_type strideA, const complex<double>* B, blas_int_type LDB, blas_int_type strideB, complex<double> BETA, complex<double>* C, blas_int_type LDC, 
                                blas_int_type strideC, blas_int_type batchCount)
{
    CBLAS_TRANSPOSE transa = get_cblas_transpose(TRANSA);
    CBLAS_TRANSPOSE transb = get_cblas_transpose(TRANSB);
    cblas_zgemm_batch_strided(CblasColMajor, transa, transb, M, N, K, reinterpret_cast<const void*>(&ALPHA), reinterpret_cast<const void*>(A), LDA, strideA, reinterpret_cast<const void*>(B), LDB, strideB, reinterpret_cast<const void*>(&BETA), reinterpret_cast<void*>(C), LDC, strideC, batchCount);
}

}
}
#else
    inline void mkl_set_dynamic(size_t) { }
    inline void mkl_set_num_threads(size_t) { }
#endif

#endif

