#ifndef TTNPP_LOGGING_HPP_
#define TTNPP_LOGGING_HPP_

#ifdef SPDLOG_LIBRARY_FOUND
#include <spdlog/spdlog.h>
#endif

#include <string_view>
#include <array>
#include <iostream>


template <std::string_view const&... Strs>
struct join
{
    // Join all strings into a single std::array of chars
    static constexpr auto impl() noexcept
    {
        constexpr std::size_t len = (Strs.size() + ... + 0);
        std::array<char, len + 1> _arr{};
        auto append = [i = 0, &_arr](auto const& s) mutable {
            for (auto c : s) _arr[i++] = c;
        };
        (append(Strs), ...);
        _arr[len] = 0;
        return _arr;
    }
    // Give the joined string static storage
    static constexpr auto arr = impl();
    // View as a std::string_view
    static constexpr std::string_view value {arr.data(), arr.size() - 1};
};
// Helper to get the value out
template <std::string_view const&... Strs>
static constexpr auto join_v = join<Strs...>::value;

struct logging
{

#ifdef SPDLOG_LIBRARY_FOUND

template <typename T> static inline void critical(const T& msg){spdlog::critical(msg);}
template <typename T> static inline void error(const T& msg){spdlog::error(msg);}
template <typename T> static inline void warn(const T& msg){spdlog::warn(msg);}
template <typename T> static inline void info(const T& msg){spdlog::info(msg);}
template <typename T> static inline void debug(const T& msg){spdlog::debug(msg);}
template <typename T> static inline void trace(const T& msg){spdlog::trace(msg);}

#else 
template <typename T> static inline void critical(const T& msg){std::cerr << "critical:" << msg << std::endl;}
template <typename T> static inline void error(const T& msg){std::cerr << "error:" << msg << std::endl;}
template <typename T> static inline void warn(const T& msg){std::cerr << "warn:" << msg << std::endl;}
template <typename T> static inline void info(const T& msg){std::cout << "info:" << msg << std::endl;}
#ifdef DEBUG
template <typename T> static inline void debug(const T& msg)
{
    std::cout << "debug:" << msg << std::endl;
}
#else
template <typename T> static inline void debug(const T& /*msg*/){}
#endif

template <typename T> static inline void trace(const T& /*msg*/){}

#endif
};   //struct logging


#endif //TTNPP_LOGGING_HPP_