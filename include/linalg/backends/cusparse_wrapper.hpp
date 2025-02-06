#ifndef LINALG_ALGEBRA_CUSPARSE_WRAPPER_HPP
#define LINALG_ALGEBRA_CUSPARSE_WRAPPER_HPP

#ifdef PYTTN_BUILD_CUDA

#include "cuda_utils.hpp"

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>

#include "cublas_wrapper.hpp"

//detect if the cuda runtime version is less than or equal to 10.0 in which case we still have the older version of 
//cusparse and it is necessary to define some of the helper functions included in later version.  We will additionally
//define the LINALG_CUSPARSE_USE_OLD macro which will be checked when performing operations to use the old implementation
//if required.
#if CUDART_VERSION <= 10000
#define LINALG_CUSPARSE_OLD
static const char *cusparseGetErrorName(cusparseStatus_t error)
{
    switch (error)
    {
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default:
            return "<unrecognised cusparse error>";
    };
}
#endif


static inline void cusparse_safe_call(cusparseStatus_t err){if(err != CUSPARSE_STATUS_SUCCESS){RAISE_EXCEPTION_STR(cusparseGetErrorName(err));}}

namespace linalg
{
namespace cusparse
{

using size_type = std::size_t;
using index_type = int32_t;

inline cusparseOperation_t convert_operation(cublasOperation_t op)
{
    if(op == CUBLAS_OP_N){return CUSPARSE_OPERATION_NON_TRANSPOSE;}
    else if(op == CUBLAS_OP_T){return CUSPARSE_OPERATION_TRANSPOSE;}
    else if(op == CUBLAS_OP_C){return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;}
    else{return static_cast<cusparseOperation_t>('I');}
}

template <typename T>
static inline void spmm(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, size_type m, size_type n, size_type k, size_type nnz, T alpha, const T* A, const index_type* rowptr, const index_type* colind, const T* B, size_type ldb, T beta, T* C, size_type ldc)
{
    cusparseConstSpMatDescr_t matA;
    cusparseConstDnMatDescr_t matB;
    cusparseDnMatDescr_t matC;
    cusparseIndexType_t offset_type = cusparse_index<index_type>::type_enum();
    cudaDataType value_type = cuda_type<T>::type_enum();

    //construct the sparse matrix descriptor
    cusparse_safe_call(cusparseCreateConstCsr(&matA, m, k, nnz, static_cast<const void*>(rowptr), static_cast<const void*>(colind), static_cast<const void*>(A), offset_type, offset_type, CUSPARSE_INDEX_BASE_ZERO, value_type));

    //construct the dense matrix descriptors
    cusparse_safe_call(cusparseCreateConstDnMat(&matB, k, n, ldb, static_cast<const void*>(B), value_type, CUSPARSE_ORDER_ROW));
    cusparse_safe_call(cusparseCreateDnMat(&matC, m, n, ldc, static_cast<void*>(C), value_type, CUSPARSE_ORDER_ROW));

    void * buffer;
    size_t bufferSize=0;

    // allocate an external buffer if needed
    cusparse_safe_call( cusparseSpMM_bufferSize(handle, opA, opB, &alpha, matA, matB, &beta, matC, value_type, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
    cuda_safe_call( cudaMalloc(&buffer, bufferSize) );

    cusparse_safe_call( cusparseSpMM(handle, opA, opB, &alpha, matA, matB, &beta, matC, value_type, CUSPARSE_SPMM_ALG_DEFAULT, buffer) );

    cusparse_safe_call(cusparseDestroySpMat(matA));
    cusparse_safe_call(cusparseDestroyDnMat(matB));
    cusparse_safe_call(cusparseDestroyDnMat(matC));
    cuda_safe_call(cudaFree(buffer));
}


template <typename T>
static inline void spmv(cusparseHandle_t handle, cusparseOperation_t opA, size_type m, size_type n, size_type nnz, T alpha, const T* A, const int* rowptr, const int* colind, const T* X, T beta, T* Y)
{
    cusparseConstSpMatDescr_t matA;
    cusparseConstDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    cusparseIndexType_t offset_type = cusparse_index<index_type>::type_enum();
    cudaDataType value_type = cuda_type<T>::type_enum();

    //construct the sparse matrix descriptor
    cusparse_safe_call(cusparseCreateConstCsr(&matA, m, n, nnz, static_cast<const void*>(rowptr), static_cast<const void*>(colind), static_cast<const void*>(A), offset_type, offset_type, CUSPARSE_INDEX_BASE_ZERO, value_type));

    //construct the dense matrix descriptors
    cusparse_safe_call(cusparseCreateConstDnVec(&vecX, n, static_cast<const void*>(X), value_type));
    cusparse_safe_call(cusparseCreateDnVec(&vecY, m, static_cast<void*>(Y), value_type));

    void * buffer;
    size_t bufferSize=0;

    // allocate an external buffer if needed
    cusparse_safe_call( cusparseSpMV_bufferSize(handle, opA, static_cast<const void*>(&alpha), matA, vecX, static_cast<const void*>(&beta), vecY, value_type, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) );
    cuda_safe_call( cudaMalloc(&buffer, bufferSize) );

    cusparse_safe_call( cusparseSpMV(handle, opA, static_cast<const void*>(&alpha), matA, vecX, static_cast<const void*>(&beta), vecY, value_type, CUSPARSE_SPMV_ALG_DEFAULT, buffer) );

    cusparse_safe_call(cusparseDestroySpMat(matA));
    cusparse_safe_call(cusparseDestroyDnVec(vecX));
    cusparse_safe_call(cusparseDestroyDnVec(vecY));
    cuda_safe_call(cudaFree(buffer));
}

}   //namespace cusparse
}   //namespace linalg

#endif

#endif //LINALG_ALGEBRA_CUSPARSE_WRAPPER_HPP//


