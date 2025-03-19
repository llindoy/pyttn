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

#ifndef PYTTN_COMMON_TMP_FUNCS_HPP_
#define PYTTN_COMMON_TMP_FUNCS_HPP_

// many of these function are only present to all for compilation on outdated clusters that do not support all of the routines present

#include <iterator>
#include <tuple>
#include <type_traits>

#include <linalg/linalg.hpp>

namespace common
{
    template <typename T>
    using complex = linalg::complex<T>;
}

namespace tmp
{

    template <typename... types>
    struct _all;

    template <>
    struct _all<> : public std::true_type
    {
    };

    template <typename... types>
    struct _all<std::false_type, types...> : public std::false_type
    {
    };

    template <typename... types>
    struct _all<std::true_type, types...> : public _all<types...>
    {
    };

    template <typename T>
    struct get_real_type
    {
        using type = T;
    };

    template <typename T>
    struct get_real_type<common::complex<T>>
    {
        using type = T;
    };

    template <typename T>
    struct is_complex : std::false_type
    {
    };

    template <typename T>
    struct is_complex<common::complex<T>>
        : std::integral_constant<bool, std::is_arithmetic<T>::value>
    {
    };

    template <typename T>
    using is_number = linalg::is_number<T>;

    // Some tmp functions
    template <bool flag, class IsTrue, class IsFalse>
    struct choose;

    template <class IsTrue, class IsFalse>
    struct choose<true, IsTrue, IsFalse>
    {
        typedef IsTrue type;
    };

    template <class IsTrue, class IsFalse>
    struct choose<false, IsTrue, IsFalse>
    {
        typedef IsFalse type;
    };

    template <typename T>
    struct is_const_pointer
    {
        static constexpr bool value = false;
    };

    template <typename T>
    struct is_const_pointer<const T *>
    {
        static constexpr bool value = true;
    };

    template <typename T>
    struct is_const_iterator
    {
        typedef typename std::iterator_traits<T>::pointer pointer;
        static constexpr bool value = is_const_pointer<pointer>::value;
    };

    template <typename T>
    struct is_reverse_iterator
    {
        static constexpr bool value = false;
    };

    template <typename T>
    struct is_reverse_iterator<std::reverse_iterator<T>>
    {
        static constexpr bool value = true;
    };

    template <typename T>
    struct reversion_wrapper
    {
        T &iterable;
    };

    template <typename T>
    auto begin(reversion_wrapper<T> w) -> decltype(w.iterable.rbegin()) { return w.iterable.rbegin(); }

    template <typename T>
    auto end(reversion_wrapper<T> w) -> decltype(w.iterable.rend()) { return w.iterable.rend(); }

} // namespace tmp

template <typename T>
tmp::reversion_wrapper<T> reverse(T &&iterable) { return {iterable}; }

#endif // PYTTN_COMMON_TMP_FUNCS_HPP_//
