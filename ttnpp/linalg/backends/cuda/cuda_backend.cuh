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

#ifndef PYTTN_LINALG_BACKENDS_CUDA_BACKENDS_IMPL_CUH_
#define PYTTN_LINALG_BACKENDS_CUDA_BACKENDS_IMPL_CUH_

#include "cuda_backend.hpp"
#include <common/exception_handling.hpp>

#include "cuda_utils.cuh"
#include "cuda_kernels.cuh"
#include "cublas_wrapper.cuh"
#include "cusparse_wrapper.cuh"
#include "cutensor_wrapper.cuh"
#include "cusolver_wrapper.cuh"

#include <complex>
#include <type_traits>

#include <cuda/std/complex>
#include <cusparse_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

namespace linalg
{
    template <typename T>
    struct get_real_type<cuda::std::complex<T>>
    {
        using type = T;
    };
    
    namespace internal
    {
        template <typename T>
        struct test_is_cuda_complex : std::false_type
        {
        };
        template <typename T>
        struct test_is_cuda_complex<cuda::std::complex<T>>
            : std::integral_constant<bool, std::is_arithmetic<T>::value>
        {
        };        
    }

    template <typename T>
    struct is_cuda_complex : public internal::test_is_cuda_complex<remove_cvref_t<T>>
    {
    };


    template <typename T>
    struct cuda_conj
    {
        static inline __host__ __device__ T eval(const T& t){return t;}
    };
    template <typename T>
    struct cuda_conj<cuda::std::complex<T>>
    {
        static inline __host__ __device__ cuda::std::complex<T> eval(const cuda::std::complex<T>& t){return cuda::std::conj(t);}
    };

    template <typename T>
    __host__ __device__  T abs(const cuda::std::complex<T> &t) { return cuda::std::abs(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> conj(const cuda::std::complex<T> &t) { return cuda::std::conj(t); }
    template <typename T>
    __host__ __device__ T real(const cuda::std::complex<T> &t) { return t.real(); }
    template <typename T>
    __host__ __device__ T imag(const cuda::std::complex<T> &t) { return t.imag(); }
    template <typename T>
    __host__ __device__ T norm(const cuda::std::complex<T> &t) { return t.norm(); }
    template <typename T>
    __host__ __device__ T arg(const cuda::std::complex<T> &t) { return cuda::std::arg(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> exp(const cuda::std::complex<T> &t) { return cuda::std::exp(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> cosh(const cuda::std::complex<T> &t) { return cuda::std::cosh(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> sinh(const cuda::std::complex<T> &t) { return cuda::std::sinh(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> tanh(const cuda::std::complex<T> &t) { return cuda::std::tanh(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> cos(const cuda::std::complex<T> &t) { return cuda::std::cos(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> sin(const cuda::std::complex<T> &t) { return cuda::std::sin(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> tan(const cuda::std::complex<T> &t) { return cuda::std::tan(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> sqrt(const cuda::std::complex<T> &t) { return cuda::std::sqrt(t); }
    template <typename T>
    __host__ __device__ cuda::std::complex<T> acos(const cuda::std::complex<T> &t) { return cuda::std::acos(t); }

    template <typename T, typename U>
    __host__ __device__ typename std::enable_if<is_cuda_complex<T>::value || is_cuda_complex<U>::value, decltype(cuda::std::pow(T(), U()))>::type pow(const T &t, const U& u) { return cuda::std::pow(t, u); }


    template <typename T>
    struct device_type<std::complex<T>, cuda_backend>
    {
        using type = cuda::std::complex<T>;
    };


    template <typename T>
    struct device_type<const std::complex<T>, cuda_backend>
    {
        using type = const cuda::std::complex<T>;
    };

    template <typename T>
    struct device_type<const std::complex<T>&, cuda_backend>
    {
        using type = const cuda::std::complex<T>&;
    };

    // functions for retreiving handle objects from the opaque void*
    inline cudaStream_t hCuda(){return reinterpret_cast<cudaStream_t>(const_cast<void *>(cuda_backend::environment().current_stream()));}
    inline cublasHandle_t hCublas() { return reinterpret_cast<cublasHandle_t>(const_cast<void *>(cuda_backend::environment().cublas_handle())); }
    inline cusparseHandle_t hCusparse() { return reinterpret_cast<cusparseHandle_t>(const_cast<void *>(cuda_backend::environment().cusparse_handle())); }
    inline cusolverDnHandle_t hCusolver() { return reinterpret_cast<cusolverDnHandle_t>(const_cast<void *>(cuda_backend::environment().cusolver_dn_handle())); }
    inline cutensorHandle_t hCutensor() { return reinterpret_cast<cutensorHandle_t>(const_cast<void *>(cuda_backend::environment().cutensor_handle())); }


    namespace cuda_internals
    {
        template <typename T>
        struct ones_indexer;
        template <>
        struct ones_indexer<float>
        {
            static inline constexpr size_t index() { return 0; }
        };
        template <>
        struct ones_indexer<double>
        {
            static inline constexpr size_t index() { return 1; }
        };
        template <>
        struct ones_indexer<cuda::std::complex<float>>
        {
            static inline constexpr size_t index() { return 2; }
        };
        template <>
        struct ones_indexer<cuda::std::complex<double>>
        {
            static inline constexpr size_t index() { return 3; }
        };



        class cuda_ones
        {
        public:
            template <typename T>
            using ones_type = std::pair<T*, typename cuda_backend::size_type>;
            using ones_state = std::tuple<ones_type<float>, ones_type<double>, ones_type<cuda::std::complex<float>>, ones_type<cuda::std::complex<double>>>;

            static ones_state& ones()
            {
                static ones_state s;
                return s;
            }
        };

        template <typename T>   
        static inline void clean_up_ones()
        {
            auto &_ones = std::get<ones_indexer<T>::index()>(cuda_internals::cuda_ones::ones());
            if(std::get<0>(_ones) != nullptr)
            {  
                CALL_AND_HANDLE(cuda_safe_call(cudaFree(&std::get<0>(_ones))), "Error when calling cudaMalloc.");
            }
            std::get<0>(_ones) = nullptr;
            std::get<1>(_ones) = 0;
        }

        template <typename T>  
        static inline void allocate_ones(size_t n)
        {
            try
            {
                ASSERT(cuda_backend::environment().is_initialised(), "his must be allocated following the instantiation of the cuda environment.");
                auto &_ones = std::get<cuda_internals::ones_indexer<T>::index()>(cuda_internals::cuda_ones::ones());
                if (std::get<1>(_ones) > n)
                {
                    return;
                }
                else
                {
                    // if the ones array has previously been allocated but its size is too small we need to deallocate it
                    if (std::get<0>(_ones) != nullptr)
                    {
                        CALL_AND_HANDLE(cuda_safe_call((cudaFree(std::get<0>(_ones)))), "Unable to free previously allocated ones array.");
                    }
                    std::get<1>(_ones) = n;
                }
                // now we need to allocate the ones buffers.
                CALL_AND_HANDLE(cuda_safe_call((cudaMalloc(&std::get<0>(_ones), std::get<1>(_ones) * sizeof(T)))), "Error when calling cudaMalloc.");
                // now we fill the ones array with ones
                size_t nthreads = cuda_backend::environment().maximum_dimensions_threads_per_block()[0];
                dim3 dg((std::get<1>(_ones) + nthreads - 1) / nthreads);
                dim3 db(nthreads);
                cuda_kernels::fill_n<<<dg, db, 0, hCuda()>>>(std::get<0>(_ones), std::get<1>(_ones), T(1.0));
            }
            catch (const std::exception &ex)
            {
                logging::error(ex.what());
                RAISE_EXCEPTION("Failed to allocate ones array.");
            }
        }

        template <typename T>   
        static inline void initialise_empty_ones_buffer()
        {
            auto &_ones = std::get<ones_indexer<T>::index()>(cuda_internals::cuda_ones::ones());
            std::get<0>(_ones) = nullptr;
            std::get<1>(_ones) = 0;
        }
    }

    // functions for mapping from the user defined enums to cuda enums.
    inline cublasOperation_t map_op(cuda_transform_type op)
    {
        switch (op)
        {
        case cuda_transform_type::n:
            return CUBLAS_OP_N;
        case cuda_transform_type::t:
            return CUBLAS_OP_T;
        case cuda_transform_type::h:
            return CUBLAS_OP_C;
        case cuda_transform_type::c:
            return static_cast<cublasOperation_t>('I');
        }
        return CUBLAS_OP_N;
    }
    inline cusparseOperation_t map_sp_op(cuda_transform_type op)
    {
        switch (op)
        {
        case cuda_transform_type::n:
            return CUSPARSE_OPERATION_NON_TRANSPOSE;
        case cuda_transform_type::t:
            return CUSPARSE_OPERATION_TRANSPOSE;
        case cuda_transform_type::h:
            return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        case cuda_transform_type::c:
            return static_cast<cusparseOperation_t>('I');
        }
        return CUSPARSE_OPERATION_NON_TRANSPOSE;
    }
    inline cusolverEigMode_t map_eig_mode(eig_mode m)
    {
        return (m == eig_mode::vectors) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    }
    inline cublasFillMode_t map_fill(fill_mode f)
    {
        return (f == fill_mode::upper) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    }


}

#endif // PYTTN_LINALG_BACKENDS_CUDA_UTILS_CUH_
