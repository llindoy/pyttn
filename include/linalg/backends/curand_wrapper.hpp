#ifndef LINALG_ALGEBRA_CURAND_WRAPPER_HPP
#define LINALG_ALGEBRA_CURAND_WRAPPER_HPP

#ifdef PYTTN_BUILD_CUDA

#include "cuda_utils.hpp"

#include <curand.h>
#include <cuda_runtime.h>

static const char *curandGetErrorName(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_VERSION_MISMATCH :
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";
            
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";

        default:
            return "<unrecognised curand error>";
    };
}

static inline void curand_safe_call(curandStatus_t err)
{
    if(err != CURAND_STATUS_SUCCESS){RAISE_EXCEPTION_STR(curandGetErrorName(err));}
    }
namespace linalg
{
namespace curand
{


}   //namespace curand
}   //namespace linalg

#endif

#endif //LINALG_ALGEBRA_CURAND_WRAPPER_HPP//


