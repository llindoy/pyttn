#ifndef LINALG_CUDA_UTILS_HPP
#define LINALG_CUDA_UTILS_HPP

#ifdef PYTTN_BUILD_CUDA
#include <cusparse_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

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

}

#endif

#endif  //LINALG_CUDA_UTILS_HPP



