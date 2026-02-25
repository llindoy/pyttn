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

#ifndef PYTTN_LINALG_UTILS_LINALG_UTILS_CUH_
#define PYTTN_LINALG_UTILS_LINALG_UTILS_CUH_

#include <cassert>
#include <array>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "linalg_utils.hpp"
#include <common/omp.hpp>
#include <cuda/std/complex>

namespace linalg
{
/*
    namespace internal
    {
        template <typename T>
        struct test_is_cuda_complex : std::false_type
        {
        };
        template <typename T>
        struct test_is_cuda_complex<std::complex<T>>
            : std::integral_constant<bool, std::is_arithmetic<T>::value>
        {
        };

    }

    template <typename T>
    struct is_cuda_complex : public internal::test_is_cuda_complex<remove_cvref_t<T>>
    {
    };
    template <typename T>
    struct is_cuda_number : std::integral_constant<bool, std::is_arithmetic<T>::value || is_cuda_complex<typename std::remove_cv<T>::type>::value>
    {
    };
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type abs(const T &t) { return std::abs(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, typename get_real_type<T>::type>::type abs(const T &t) { return cuda::std::abs(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type conj(const T &t) { return t; }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type conj(const T &t) { return cuda::std::conj(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type real(const T &t) { return t; }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, typename get_real_type<T>::type>::type real(const T &t) { return t.real(); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type imag(const T & t ) { return T(0.0); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, typename get_real_type<T>::type>::type imag(const T &t) { return t.imag(); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type norm(const T &t) { return t * t; }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, typename get_real_type<T>::type>::type norm(const T &t) { return t.norm(); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type arg(const T & t ) { return T(0.0); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, typename get_real_type<T>::type>::type arg(const T &t) { return cuda::std::arg(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, cuda::std::complex<T>>::type polar(const T &r, const T &theta) { return cuda::std::polar(r, theta); }

    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type exp(const T &t) { return std::exp(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type exp(const T &t) { return cuda::std::exp(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type cosh(const T &t) { return std::cosh(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type cosh(const T &t) { return cuda::std::cosh(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type sinh(const T &t) { return std::sinh(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type sinh(const T &t) { return cuda::std::sinh(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type tanh(const T &t) { return std::tanh(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type tanh(const T &t) { return cuda::std::tanh(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type cos(const T &t) { return std::cos(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type cos(const T &t) { return cuda::std::cos(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type sin(const T &t) { return std::sin(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type sin(const T &t) { return cuda::std::sin(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type tan(const T &t) { return std::tan(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type tan(const T &t) { return cuda::std::tan(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type sqrt(const T &t) { return std::sqrt(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type sqrt(const T &t) { return cuda::std::sqrt(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, T>::type acos(const T &t) { return std::acos(t); }
    template <typename T>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, T>::type acos(const T &t) { return cuda::std::acos(t); }
    template <typename T, typename U>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && !is_cuda_complex<T>::value, decltype(std::pow(T(), U()))>::type pow(const T &t, const U& u) { return std::pow(t, u); }
    template <typename T, typename U>
    __host__ __device__ typename std::enable_if<is_cuda_number<T>::value && is_cuda_complex<T>::value, decltype(cuda::std::pow(T(), U()))>::type pow(const T &t, const U& u) { return cuda::std::pow(t, u); }
    */

}

#endif // PYTTN_LINALG_UTILS_LINALG_UTILS_CUH_
