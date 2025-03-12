#ifndef COMMON_ITER_ZIP_HPP_
#define COMMON_ITER_ZIP_HPP_

#include <tuple>
#include <type_traits>
#include <utility>
#include <cstddef>

namespace common
{
template<int... Is>
struct seq { };

template<int N, int... Is>
struct gen_seq : gen_seq<N - 1, N - 1, Is...> { };

template<int... Is>
struct gen_seq<0, Is...> : seq<Is...> { }; 


template <typename T>
using riter_t = decltype(std::declval<T&>().rbegin());

template <typename T>
using iter_t = decltype(std::declval<T&>().begin());

template <typename T>
using ref_t = decltype(*std::declval<iter_t<T>>());


template <std::size_t I = 0, typename ... Args >
inline typename std::enable_if< I == sizeof...(Args), void>::type increment(std::tuple<Args...>& /* tup */)
{ }

template <std::size_t I = 0, typename ... Args >
inline typename std::enable_if< I < sizeof...(Args), void>::type increment(std::tuple<Args...>& tup)
{
    ++std::get<I>(tup);
    increment<I+1>(tup);
}

template <std::size_t I = 0, typename ... Args>
inline typename std::enable_if< I == sizeof...(Args), bool>::type any_equal(const std::tuple<Args...>& /* lhs */, const std::tuple<Args...>& /* rhs */)
{
    return false;
}

template <std::size_t I = 0, typename ... Args>
inline typename std::enable_if< I < sizeof...(Args), bool>::type any_equal(const std::tuple<Args...>& lhs, const std::tuple<Args...>& rhs)
{
    return std::get<I>(lhs) == std::get<I>(rhs) || any_equal<I+1>(lhs, rhs);
}

template<typename... Args, int... Is>
std::tuple<decltype(*std::declval<Args>())...> dereference_tuple(std::tuple<Args...>& t, seq<Is...>)
{
    return std::tuple<decltype(*std::declval<Args>())...>((*std::get<Is>(t))...);
}

template<typename... Args>
auto dereference_tuple(std::tuple<Args...>& t)
    -> decltype(dereference_tuple(t, gen_seq<sizeof...(Args)>()))
{
    return dereference_tuple(t, gen_seq<sizeof...(Args)>());
}

template <typename ... Args>
class zip_container
{
    static_assert(sizeof...(Args) != 0, "Cannot create zip_container object containing zero elements");

public:
    class iterator 
    {
    protected:
        std::tuple<iter_t<Args>... > m_loc;
    public:
        explicit iterator( iter_t<Args>... args) : m_loc(args...) {}
        iterator(const iterator& other) = default;

        iterator& operator++()
        {
            increment(m_loc);
            return* this;
        }

        iterator& operator++(int)
        {
            auto res = *this;
            increment(m_loc);
            return res;
        }

        bool operator==(const iterator& other) 
        {
            return any_equal(m_loc, other.m_loc);
        }

        bool operator!=(const iterator& rhs) 
        {       
            return !(*this == rhs); 
        }

        std::tuple<ref_t<Args>... > operator*()
        {
            return dereference_tuple(m_loc);
        }
        //iterator traits
        using difference_type = long;
        using value_type = long;
        using pointer = const long*;
        using reference = const long&;
        using iterator_category = std::forward_iterator_tag;
    };

    explicit zip_container(Args&... containers)
        : m_begin(containers.begin()...)
        , m_end(containers.end()...)
        { }       


    zip_container(const zip_container& other) = default;   
    zip_container& operator=(const zip_container& other) = default;

    zip_container::iterator& begin() {
        return m_begin;
    }

    zip_container::iterator& end() {
        return m_end;
    }

    zip_container::iterator m_begin;
    zip_container::iterator m_end;
};

template <typename ... Args>
class reverse_zip_container
{
    static_assert(sizeof...(Args) != 0, "Cannot create reverse_zip_container object containing zero elements");

public:
    class iterator 
    {
    protected:
        std::tuple<riter_t<Args>... > m_loc;
    public:
        explicit iterator( riter_t<Args>... args) : m_loc(args...) {}
        iterator(const iterator& other) = default;

        iterator& operator++()
        {
            increment(m_loc);
            return* this;
        }

        iterator& operator++(int)
        {
            auto res = *this;
            increment(m_loc);
            return res;
        }

        bool operator==(const iterator& other) 
        {
            return any_equal(m_loc, other.m_loc);
        }

        bool operator!=(const iterator& rhs) 
        {       
            return !(*this == rhs); 
        }

        std::tuple<ref_t<Args>... > operator*()
        {
            return dereference_tuple(m_loc);
        }
        //iterator traits
        using difference_type = long;
        using value_type = long;
        using pointer = const long*;
        using reference = const long&;
        using iterator_category = std::forward_iterator_tag;
    };

    explicit reverse_zip_container(Args&... containers)
        : m_begin(containers.rbegin()...)
        , m_end(containers.rend()...)
        { }       


    reverse_zip_container(const reverse_zip_container& other) = default;   
    reverse_zip_container& operator=(const reverse_zip_container& other) = default;

    reverse_zip_container::iterator& begin() {
        return m_begin;
    }

    reverse_zip_container::iterator& end() {
        return m_end;
    }

    reverse_zip_container::iterator m_begin;
    reverse_zip_container::iterator m_end;
};


//from cppreference.com: 
template <class T>
  struct special_decay
  {
     using type = typename std::decay<T>::type;
  };

//allows the use of references:
template <class T>
 struct special_decay<std::reference_wrapper<T>>
 {
   using type = T&;
 };

template <class T>
 using special_decay_t = typename special_decay<T>::type;

//allows template type deduction for zipper:
template <class... Args>
zip_container<Args...> zip(Args&... args)
{
  return zip_container<Args...>(args...);
}

//allows template type deduction for zipper:
template <class... Args>
reverse_zip_container<Args...> rzip(Args&... args)
{
  return reverse_zip_container<Args...>(args...);
}

}   //namespace ttns

#endif //COMMON_ITER_ZIP_HPP_//
