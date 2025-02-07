#ifndef LINALG_CUDA_UTILS_HPP
#define LINALG_CUDA_UTILS_HPP

#ifdef PYTTN_BUILD_CUDA
#include <cusparse_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

static inline void cuda_safe_call(cudaError_t err){if(err != cudaSuccess){RAISE_EXCEPTION_STR(cudaGetErrorName(err));}}

namespace linalg
{
template <typename T> class cuda_type;
template <> 
class cuda_type<float>
{
public:
    using type = float;
    static inline cudaDataType_t type_enum(){return CUDA_R_32F;}
};

template <> 
class cuda_type<double>
{
public:
    using type = double;
    static inline cudaDataType_t type_enum(){return CUDA_R_64F;}
};

template <> class cuda_type<complex<float> >
{
public:
    using type = cuComplex;
    static inline cudaDataType_t type_enum(){return CUDA_C_32F;}
};

template <> class cuda_type<complex<double> >
{
public:
    using type = cuDoubleComplex;
    static inline cudaDataType_t type_enum(){return CUDA_C_64F;}
};


template <typename T> class cusparse_index;

template <> class cusparse_index<int32_t>
{
public:
    static inline cusparseIndexType_t type_enum(){return CUSPARSE_INDEX_32I;}
};

template <> class cusparse_index<int64_t>
{
public:
    static inline cusparseIndexType_t type_enum(){return CUSPARSE_INDEX_64I;}
};
}

#endif

#endif  //LINALG_CUDA_UTILS_HPP



