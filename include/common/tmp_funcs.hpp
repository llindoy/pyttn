#ifndef COMMON_TEMPLATE_METAPROGRAMMING_FUNCS_HPP_
#define COMMON_TEMPLATE_METAPROGRAMMING_FUNCS_HPP_

//many of these function are only present to all for compilation on outdated clusters that do not support all of the routines present

#include <iterator>
#include <tuple>
#include <type_traits>

#include <linalg/utils/linalg_utils.hpp>

namespace common
{
template <typename T>
using complex = linalg::complex<T>;
}

namespace tmp
{


template <typename ... types>
struct _all;

template <>
struct _all<> : public std::true_type{};

template <typename ... types>
struct _all<std::false_type, types... > : public std::false_type{};

template <typename ... types>
struct _all<std::true_type, types... > : public _all<types...> {};

template <typename T> 
struct get_real_type{using type = T;};

template <typename T> 
struct get_real_type<common::complex<T> >{using type = T;};

template <typename T> struct is_complex : std::false_type {};

template <typename T> struct is_complex<common::complex<T> > 
    : std::integral_constant<bool, std::is_arithmetic<T>::value>{};

template <typename T> 
using is_number = linalg::is_number<T>;

//Some tmp functions
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
struct is_const_pointer<const T*> 
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
struct is_reverse_iterator<std::reverse_iterator<T> >
{
    static constexpr bool value = true;
};

template <typename T>
struct reversion_wrapper { T& iterable; };

template <typename T>
auto begin (reversion_wrapper<T> w) -> decltype(w.iterable.rbegin()){ return w.iterable.rbegin(); }

template <typename T>
auto end (reversion_wrapper<T> w) -> decltype(w.iterable.rend()){ return w.iterable.rend(); }

}   //namespace tmp

template <typename T>
tmp::reversion_wrapper<T> reverse (T&& iterable) { return { iterable }; }

#endif  //COMMON_TEMPLATE_METAPROGRAMMING_FUNCS_HPP_//


