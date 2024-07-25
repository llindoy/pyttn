#ifndef UTILS_IO_CONFIG_TYPE_INFO_HPP_
#define UTILS_IO_CONFIG_TYPE_INFO_HPP_

#include "common.hpp"

namespace utils
{
namespace io
{
template <typename T> class type_info;
}
}

#define CONFIG_STRINGIFY(x) #x
#define CONFIG_TOSTRING(x) CONFIG_STRINGIFY(x)

#define REGISTER_TYPE_INFO_WITH_NAME(T, _name, _desc, _req)                                                     \
namespace utils                                                                                                 \
{                                                                                                               \
namespace io                                                                                                    \
{                                                                                                               \
template <> class type_info<T>                                                                                  \
{                                                                                                               \
public:                                                                                                         \
    template <typename INTERFACE>                                                                               \
    static std::shared_ptr<INTERFACE> create(){return make_shared<T>();}                                        \
    template <typename INTERFACE, typename ... Args>                                                            \
    static std::shared_ptr<INTERFACE> create_from_obj(const rapidjson::Value& obj, Args&& ... args)             \
    {return make_shared<T>(obj, std::forward<Args>(args)...);}                                                  \
    static const std::string& get_alias(){static std::string alias(CONFIG_TOSTRING(T)); return alias;}          \
    static const std::string& get_name(){static std::string name(_name); return name;}                          \
    static const std::string& get_description(){static std::string desc(_desc);   return desc;}                 \
    static const std::string& get_required_inputs(){static std::string req(_req);    return req;}               \
};                                                                                                              \
}                                                                                                               \
}                                                                                                               


#define REGISTER_TEMPLATE_TYPE_INFO_WITH_NAME(T, _name, _desc, _req)                                            \
namespace utils                                                                                                 \
{                                                                                                               \
namespace io                                                                                                    \
{                                                                                                               \
template <typename ... Args> class type_info<T<Args...>>                                                        \
{                                                                                                               \
public:                                                                                                         \
    template <typename INTERFACE>                                                                               \
    static std::shared_ptr<INTERFACE> create(){return make_shared<T<Args...>>();}                               \
    template <typename INTERFACE, typename ... FArgs>                                                           \
    static std::shared_ptr<INTERFACE> create_from_obj(const rapidjson::Value& obj, FArgs&& ... args)            \
    {return make_shared<T<Args...>>(obj, std::forward<FArgs>(args)...);}                                        \
    static const std::string& get_alias(){static std::string alias(CONFIG_TOSTRING(T<Args...>)); return alias;} \
    static const std::string& get_name(){static std::string name(_name); return name;}                          \
    static const std::string& get_description(){static std::string desc(_desc);   return desc;}                 \
    static const std::string& get_required_inputs(){static std::string req(_req);    return req;}               \
};                                                                                                              \
}                                                                                                               \
}                                                                                                               

#endif

