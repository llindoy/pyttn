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

#ifndef PYTTN_LINALG_UTILS_LINALG_UTILS_HPP_
#define PYTTN_LINALG_UTILS_LINALG_UTILS_HPP_

#include <cassert>
#include <array>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <complex>

#include <common/omp.hpp>

///@cond INTERNAL

/**
 *  This file provides declarations of various types.
// */
//#ifdef 
//#define  __restrict__
//#else
//#define 
//#endif

namespace linalg
{

    template <class T>
    struct remove_reference
    {
        using type = T;
    };
    template <class T>
    struct remove_reference<T &>
    {
        using type = T;
    };
    template <class T>
    struct remove_reference<T &&>
    {
        using type = T;
    };

    template <typename T>
    using remove_reference_t = typename remove_reference<T>::type;

    template <typename T>
    struct remove_cvref
    {
        using type = typename std::remove_cv<remove_reference_t<T>>::type;
    };

    template <class T>
    using remove_cvref_t = typename remove_cvref<T>::type;

    template <typename T>
    struct get_real_type
    {
        using type = T;
    };
    template <typename T>
    struct get_real_type<std::complex<T>>
    {
        using type = T;
    };

    template <typename T>
    struct is_valid_value_type : std::false_type
    {
    };
    template <>
    struct is_valid_value_type<float> : std::true_type
    {
    };
    template <>
    struct is_valid_value_type<double> : std::true_type
    {
    };
    template <>
    struct is_valid_value_type<std::complex<float>> : std::true_type
    {
    };
    template <>
    struct is_valid_value_type<std::complex<double>> : std::true_type
    {
    };

    template <typename T, typename Q>
    struct is_same : std::false_type
    {
    };
    template <typename T>
    struct is_same<T, T> : std::true_type
    {
    };

    namespace internal
    {
        template <typename T>
        struct test_is_complex : std::false_type
        {
        };
        template <typename T>
        struct test_is_complex<std::complex<T>>
            : std::integral_constant<bool, std::is_arithmetic<T>::value>
        {
        };

    }

    template <typename T>
    struct is_complex : public internal::test_is_complex<remove_cvref_t<T>>
    {
    };
    template <typename T>
    struct is_number : std::integral_constant<bool, std::is_arithmetic<T>::value || is_complex<typename std::remove_cv<T>::type>::value>
    {
    };

    template <typename T>
    typename std::enable_if<is_number<T>::value && !is_complex<T>::value, T>::type conj(const T &t) { return t; }
    template <typename T>
    typename std::enable_if<is_number<T>::value && is_complex<T>::value, T>::type conj(const T &t) { return std::conj(t); }
    
    namespace internal
    {
        template <typename Int, typename size_type>
        static inline constexpr typename std::enable_if<std::is_unsigned<Int>::value && std::is_unsigned<size_type>::value, bool>::type compare_bounds(const Int &i, const size_type &bounds) { return i < bounds; }

        template <typename Int, typename size_type>
        static inline constexpr typename std::enable_if<std::is_integral<Int>::value && !std::is_unsigned<Int>::value && std::is_unsigned<size_type>::value, bool>::type compare_bounds(const Int &i, const size_type &bounds) { return (i >= 0 && static_cast<size_type>(i) < bounds); }

        template <typename Int, typename size_type>
        static inline constexpr typename std::enable_if<std::is_integral<size_type>::value && std::is_unsigned<Int>::value && !std::is_unsigned<size_type>::value, bool>::type compare_bounds(const Int &i, const size_type &bounds) { return (i < static_cast<Int>(bounds) || bounds < 0); }

        template <typename Int, typename size_type>
        static inline constexpr typename std::enable_if<std::is_integral<Int>::value && std::is_integral<size_type>::value && !std::is_unsigned<Int>::value && !std::is_unsigned<size_type>::value, bool>::type compare_bounds(const Int &i, const size_type &bounds) { return i < bounds; }

        template <typename... Args>
        struct check_integral;

        template <typename first_type, typename... Rest>
        struct check_integral<first_type, Rest...>
        {
            typedef first_type pack_type;
            enum
            {
                tmp = std::is_integral<first_type>::value
            };
            enum
            {
                value = tmp && check_integral<Rest...>::value
            };
            static_assert(value, "Non integer type found in parameter pack.");
        };

        template <typename last_type>
        struct check_integral<last_type>
        {
            typedef last_type pack_type;
            enum
            {
                value = std::is_integral<last_type>::value
            };
        };

    } // bool compare bounds

}

#endif // PYTTN_LINALG_UTILS_LINALG_UTILS_HPP_
