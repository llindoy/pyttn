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

#ifndef PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_APPLICATIVE_COMPLEX_HPP_
#define PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_APPLICATIVE_COMPLEX_HPP_

#include "../../../../linalg_forward_decl.hpp"

namespace linalg
{

    namespace expression_templates
    {

        template <>
        class complex_op<blas_backend>
        {
        public:
            using size_type = blas_backend::size_type;

        public:
            template <typename T1, typename T2>
            static inline auto apply(const T1 *a, const T2 *b, size_type i) -> complex<decltype(a[i] + b[i])> { return complex<decltype(a[i] + b[i])>(a[i], b[i]); }

            template <typename T, typename ltype, typename rtype, template <typename> class op>
            static inline auto apply(const T *a, binary_expression<ltype, rtype, op, blas_backend> b, size_type i) -> complex<decltype(a[i] + b[i])> { return complex<decltype(a[i] + b[i])>(a[i], b[i]); }

            template <typename T, typename vtype, template <typename> class op>
            static inline auto apply(const T *a, unary_expression<vtype, op, blas_backend> b, size_type i) -> complex<decltype(a[i] + b[i])> { return complex<decltype(a[i] + b[i])>(a[i], b[i]); }
        };
#ifdef PYTTN_BUILD_CUDA
        template <>
        class complex_op<cuda_backend>
        {
        public:
            using size_type = cuda_backend::size_type;

        public:
            template <typename T1, typename T2>
            static inline __device__ auto apply(const T1 *a, const T2 *b, size_type i) -> complex<decltype(a[i] + b[i])> { return complex<decltype(a[i] + b[i])>(a[i], b[i]); }

            template <typename T, typename ltype, typename rtype, template <typename> class op>
            static inline __device__ auto apply(const T *a, binary_expression<ltype, rtype, op, cuda_backend> b, size_type i) -> complex<decltype(a[i] + b[i])> { return complex<decltype(a[i] + b[i])>(a[i], b[i]); }

            template <typename T, typename vtype, template <typename> class op>
            static inline __device__ auto apply(const T *a, unary_expression<vtype, op, cuda_backend> b, size_type i) -> complex<decltype(a[i] + b[i])> { return complex<decltype(a[i] + b[i])>(a[i], b[i]); }
        };
#endif

        template <>
        class polar_op<blas_backend>
        {
        public:
            using size_type = blas_backend::size_type;

        public:
            template <typename T1, typename T2>
            static inline auto apply(const T1 *a, const T2 *b, size_type i) -> complex<decltype(a[i] + b[i])> { return polar(a[i], b[i]); }

            template <typename T, typename ltype, typename rtype, template <typename> class op>
            static inline auto apply(const T *a, binary_expression<ltype, rtype, op, blas_backend> b, size_type i) -> complex<decltype(a[i] + b[i])> { return polar(a[i], b[i]); }

            template <typename T, typename vtype, template <typename> class op>
            static inline auto apply(const T *a, unary_expression<vtype, op, blas_backend> b, size_type i) -> complex<decltype(a[i] + b[i])> { return polar(a[i], b[i]); }
        };
#ifdef PYTTN_BUILD_CUDA
        template <>
        class polar_op<cuda_backend>
        {
        public:
            using size_type = cuda_backend::size_type;

        public:
            template <typename T1, typename T2>
            static inline __device__ auto apply(const T1 *a, const T2 *b, size_type i) -> complex<decltype(a[i] + b[i])> { return polar(a[i], b[i]); }

            template <typename T, typename ltype, typename rtype, template <typename> class op>
            static inline __device__ auto apply(const T *a, binary_expression<ltype, rtype, op, cuda_backend> b, size_type i) -> complex<decltype(a[i] + b[i])> { return polar(a[i], b[i]); }

            template <typename T, typename vtype, template <typename> class op>
            static inline __device__ auto apply(const T *a, unary_expression<vtype, op, cuda_backend> b, size_type i) -> complex<decltype(a[i] + b[i])> { return polar(a[i], b[i]); }
        };
#endif

        // unary expression for evaluating the real part of a complex array
        template <>
        class unit_polar_op<blas_backend>
        {
        public:
            using size_type = blas_backend::size_type;

        public:
            template <typename T>
            static inline auto apply(const T *a, size_type i) -> complex<decltype(a[i])> { return complex<decltype(a[i])>(cos(a[i]), sin(a[i])); }
            template <typename T>
            static inline auto apply(const T &a, size_type i) -> complex<decltype(a[i])> { return complex<decltype(a[i])>(cos(a[i]), sin(a[i])); }
        };
#ifdef PYTTN_BUILD_CUDA
        template <>
        class unit_polar_op<cuda_backend>
        {
        public:
            using size_type = cuda_backend::size_type;

        public:
            template <typename T>
            static inline __device__ auto apply(const T *a, size_type i) -> complex<decltype(a[i])> { return complex<decltype(a[i])>(cos(a[i]), sin(a[i])); }
            template <typename T>
            static inline __device__ auto apply(const T &a, size_type i) -> complex<decltype(a[i])> { return complex<decltype(a[i])>(cos(a[i]), sin(a[i])); }
        };
#endif

        // unary expression for evaluating the real part of a complex array
        template <>
        class real_op<blas_backend>
        {
        public:
            using size_type = blas_backend::size_type;

        public:
            template <typename T>
            static inline typename get_real_type<T>::type apply(const T *a, size_type i) { return real(a[i]); }
            template <typename T>
            static inline typename get_real_type<typename T::value_type>::type apply(const T &a, size_type i) { return real(a[i]); }
        };
#ifdef PYTTN_BUILD_CUDA
        template <>
        class real_op<cuda_backend>
        {
        public:
            using size_type = cuda_backend::size_type;

        public:
            template <typename T>
            static inline __device__ typename get_real_type<T>::type apply(const T *a, size_type i) { return real(a[i]); }
            template <typename T>
            static inline __device__ typename get_real_type<typename T::value_type>::type apply(const T &a, size_type i) { return real(a[i]); }
        };
#endif

        template <>
        class imag_op<blas_backend>
        {
        public:
            using size_type = blas_backend::size_type;

        public:
            template <typename T>
            static inline typename get_real_type<T>::type apply(const T *a, size_type i) { return imag(a[i]); }
            template <typename T>
            static inline typename get_real_type<typename T::value_type>::type apply(const T &a, size_type i) { return imag(a[i]); }
        };
#ifdef PYTTN_BUILD_CUDA
        template <>
        class imag_op<cuda_backend>
        {
        public:
            using size_type = cuda_backend::size_type;

        public:
            template <typename T>
            static inline __device__ typename get_real_type<T>::type apply(const T *a, size_type i) { return imag(a[i]); }
            template <typename T>
            static inline __device__ typename get_real_type<typename T::value_type>::type apply(const T &a, size_type i) { return imag(a[i]); }
        };
#endif

        template <>
        class norm_op<blas_backend>
        {
        public:
            using size_type = blas_backend::size_type;

        public:
            template <typename T>
            static inline typename get_real_type<T>::type apply(const T *a, size_type i) { return norm(a[i]); }
            template <typename T>
            static inline typename get_real_type<typename T::value_type>::type apply(const T &a, size_type i) { return norm(a[i]); }
        };
#ifdef PYTTN_BUILD_CUDA
        template <>
        class norm_op<cuda_backend>
        {
        public:
            using size_type = cuda_backend::size_type;

        public:
            template <typename T>
            static inline __device__ typename get_real_type<T>::type apply(const T *a, size_type i) { return norm(a[i]); }
            template <typename T>
            static inline __device__ typename get_real_type<typename T::value_type>::type apply(const T &a, size_type i) { return norm(a[i]); }
        };
#endif

        template <>
        class arg_op<blas_backend>
        {
        public:
            using size_type = blas_backend::size_type;

        public:
            template <typename T>
            static inline typename get_real_type<T>::type apply(const T *a, size_type i) { return arg(a[i]); }
            template <typename T>
            static inline typename get_real_type<typename T::value_type>::type apply(const T &a, size_type i) { return arg(a[i]); }
        };
#ifdef PYTTN_BUILD_CUDA
        template <>
        class arg_op<cuda_backend>
        {
        public:
            using size_type = cuda_backend::size_type;

        public:
            template <typename T>
            static inline __device__ typename get_real_type<T>::type apply(const T *a, size_type i) { return arg(a[i]); }
            template <typename T>
            static inline __device__ typename get_real_type<typename T::value_type>::type apply(const T &a, size_type i) { return arg(a[i]); }
        };
#endif

    } // namespace expression_templates
} // namespace linalg

#endif // PYTTN_LINALG_ALGEBRA_EXPRESSIONS_ELEMENTAL_APPLICATIVE_COMPLEX_HPP_
