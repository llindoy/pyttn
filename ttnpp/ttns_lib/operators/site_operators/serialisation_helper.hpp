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

#ifndef PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_SERIALISATION_HELPER_HPP_
#define PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_SERIALISATION_HELPER_HPP_

#include <complex>
#include <linalg/linalg.hpp>

#define TTNS_COMMA ,

#ifdef CEREAL_LIBRARY_FOUND
#define TTNS_REGISTER_POLYMORPHIC_SERIALIZATION(op_name, base_name, type, backend) \
    CEREAL_REGISTER_TYPE(op_name<type TTNS_COMMA backend>)                         \
    CEREAL_REGISTER_POLYMORPHIC_RELATION(base_name<type TTNS_COMMA backend>, op_name<type TTNS_COMMA backend>)

// define macros to help serialize operator types
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

// register blas operators
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

// macro for registering all possible serializations
#define TTNS_REGISTER_SERIALIZATION(op_name, base_name)   \
    TTNS_REGISTER_REAL_FLOAT_CUDA(op_name, base_name)     \
    TTNS_REGISTER_REAL_DOUBLE_CUDA(op_name, base_name)    \
    TTNS_REGISTER_COMPLEX_FLOAT_CUDA(op_name, base_name)  \
    TTNS_REGISTER_COMPLEX_DOUBLE_CUDA(op_name, base_name) \
    TTNS_REGISTER_REAL_FLOAT_BLAS(op_name, base_name)     \
    TTNS_REGISTER_REAL_DOUBLE_BLAS(op_name, base_name)    \
    TTNS_REGISTER_COMPLEX_FLOAT_BLAS(op_name, base_name)  \
    TTNS_REGISTER_COMPLEX_DOUBLE_BLAS(op_name, base_name)
#else
#define TTNS_REGISTER_SERIALIZATION(op_name, base_name)
#endif

#endif // PYTTN_TTNS_LIB_OPERATORS_SITE_OPERATORS_SERIALISATION_HELPER_HPP_
