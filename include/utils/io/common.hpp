#ifndef UTILS_IO_CONFIG_COMMON_HPP_
#define UTILS_IO_CONFIG_COMMON_HPP_

#include <linalg/utils/linalg_utils.hpp>
#include <common/exception_handling.hpp>
#include <memory>

namespace utils
{
namespace io
{
template <typename T, typename ... Args>
std::unique_ptr<T> make_unique(Args&&... args){return std::unique_ptr<T>(new T(std::forward<Args>(args)...));}

template <typename T>
std::shared_ptr<T> make_shared(){return std::shared_ptr<T>(new T());}

template <typename T, typename ... Args>
std::shared_ptr<T> make_shared(Args&&... args)
{
    return std::shared_ptr<T>(new T(std::forward<Args>(args)...));
}

}   //namespace io
}   //namespace utils

#endif

