#ifndef TTNS_OPS_SERIALISATION_HELPER_HPP
#define TTNS_OPS_SERIALISATION_HELPER_HPP

#include <complex>
#include <linalg/linalg.hpp>

#define TTNS_COMMA ,

#ifdef CEREAL_LIBRARY_FOUND
    #define TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, type, backend) \
        CEREAL_REGISTER_TYPE(op_name<type TTNS_COMMA backend>) \
        CEREAL_REGISTER_POLYMORPHIC_RELATION(base_name<type TTNS_COMMA backend>, op_name<type TTNS_COMMA backend>)

    //define macros to help serialize operator types
    #ifdef SERIALIZE_CUDA_TYPES
        #ifdef TTNS_REGISTER_REAL_FLOAT
            #define TTNS_REGISTER_REAL_FLOAT_CUDA(op_name, base_name) TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, float, linalg::cuda_backend)
        #else
            #define TTNS_REGISTER_REAL_FLOAT_CUDA(op_name, base_name)
        #endif

        #ifdef TTNS_REGISTER_REAL_DOUBLE
            #define TTNS_REGISTER_REAL_DOUBLE_CUDA(op_name, base_name) TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, double, linalg::cuda_backend)
        #else
            #define TTNS_REGISTER_REAL_DOUBLE_CUDA(op_name, base_name) 
        #endif

        #ifdef TTNS_REGISTER_COMPLEX_FLOAT
            #define TTNS_REGISTER_COMPLEX_FLOAT_CUDA(op_name, base_name) TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, ttns::complex<float>, linalg::cuda_backend)
        #else
            #define TTNS_REGISTER_COMPLEX_FLOAT_CUDA(op_name, base_name)
        #endif

        #ifdef TTNS_REGISTER_COMPLEX_DOUBLE
            #define TTNS_REGISTER_COMPLEX_DOUBLE_CUDA(op_name, base_name) TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, ttns::complex<double>, linalg::cuda_backend)
        #else
            #define TTNS_REGISTER_COMPLEX_DOUBLE_CUDA(op_name, base_name) 
        #endif
    #else
        #define TTNS_REGISTER_REAL_FLOAT_CUDA(op_name, base_name)
        #define TTNS_REGISTER_REAL_DOUBLE_CUDA(op_name, base_name) 
        #define TTNS_REGISTER_COMPLEX_FLOAT_CUDA(op_name, base_name)
        #define TTNS_REGISTER_COMPLEX_DOUBLE_CUDA(op_name, base_name) 
    #endif


    //register blas operators
    #ifdef TTNS_REGISTER_REAL_FLOAT
        #define TTNS_REGISTER_REAL_FLOAT_BLAS(op_name, base_name) TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, float, linalg::blas_backend)
    #else
        #define TTNS_REGISTER_REAL_FLOAT_BLAS(op_name, base_name)
    #endif

    #ifdef TTNS_REGISTER_REAL_DOUBLE
        #define TTNS_REGISTER_REAL_DOUBLE_BLAS(op_name, base_name) TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, double, linalg::blas_backend)
    #else
        #define TTNS_REGISTER_REAL_DOUBLE_BLAS(op_name, base_name) 
    #endif

    #ifdef TTNS_REGISTER_COMPLEX_FLOAT
        #define TTNS_REGISTER_COMPLEX_FLOAT_BLAS(op_name, base_name) TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, ttns::complex<float>, linalg::blas_backend)
    #else
        #define TTNS_REGISTER_COMPLEX_FLOAT_BLAS(op_name, base_name)
    #endif

    #ifdef TTNS_REGISTER_COMPLEX_DOUBLE
        #define TTNS_REGISTER_COMPLEX_DOUBLE_BLAS(op_name, base_name) TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, ttns::complex<double>, linalg::blas_backend)
    #else
        #define TTNS_REGISTER_COMPLEX_DOUBLE_BLAS(op_name, base_name) 
    #endif

    //macro for registering all possible serializations 
    #define TTNS_REGISTER_SERIALIZATION(op_name, base_name)       \
        TTNS_REGISTER_REAL_FLOAT_CUDA(op_name, base_name)         \
        TTNS_REGISTER_REAL_DOUBLE_CUDA(op_name, base_name)        \
        TTNS_REGISTER_COMPLEX_FLOAT_CUDA(op_name, base_name)      \
        TTNS_REGISTER_COMPLEX_DOUBLE_CUDA(op_name, base_name)     \
        TTNS_REGISTER_REAL_FLOAT_BLAS(op_name, base_name)         \
        TTNS_REGISTER_REAL_DOUBLE_BLAS(op_name, base_name)        \
        TTNS_REGISTER_COMPLEX_FLOAT_BLAS(op_name, base_name)      \
        TTNS_REGISTER_COMPLEX_DOUBLE_BLAS(op_name, base_name)     
#else
    #define TTNS_REGISTER_SERIALIZATION(op_name, base_name) 
#endif




#endif
