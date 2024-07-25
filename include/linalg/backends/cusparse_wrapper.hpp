#ifndef LINALG_ALGEBRA_CUSPARSE_WRAPPER_HPP
#define LINALG_ALGEBRA_CUSPARSE_WRAPPER_HPP

#ifdef __NVCC__

#include "cuda_utils.hpp"

#include <cusparse_v2.h>
#include <cuda_runtime.h>


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


}   //namespace cusparse
}   //namespace linalg

#endif

#endif //LINALG_ALGEBRA_CUSPARSE_WRAPPER_HPP//


