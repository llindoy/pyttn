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

#ifndef PYTTN_UTILS_IO_FACTORY_HPP_
#define PYTTN_UTILS_IO_FACTORY_HPP_

#include "input_wrapper.hpp"
#include "type_info.hpp"
#include <common/exception_handling.hpp>
#include <map>
#include <memory>
#include <sstream>
#include <string>

namespace utils
{
    namespace io
    {

        template <typename IB>
        struct factory_info_type
        {
            using create_function = std::shared_ptr<IB> (*)();
            using create_function_obj =
                std::shared_ptr<IB> (*)(const typename IOWRAPPER::input_object &);

            create_function m_f;
            create_function_obj m_fobj;
            std::string m_desc;
            std::string m_req;
            std::string m_name;
        };

        template <typename INTERFACE_TYPE>
        class factory
        {
        public:
            using info_type = factory_info_type<INTERFACE_TYPE>;

        public:
            factory() = delete;
            static bool register_type(const std::string name, const std::string alias,
                                      info_type info)
            {
                if (get_info().find(alias) == get_info().end())
                {
                    get_info()[alias] = info;
                }
                else
                {
                    return false;
                }
                if (get_alias().find(name) == get_alias().end())
                {
                    get_alias()[name] = alias;
                }
                else
                {
                    return false;
                }
                return true;
            }

            static bool is_registered(const std::string &name)
            {
                return (get_alias().find(name) != get_alias().end());
            }

            static bool is_loadable(const typename IOWRAPPER::input_object &obj)
            {
                try
                {
                    std::string s;
                    CALL_AND_HANDLE(
                        IOWRAPPER::load<std::string>(obj, INTERFACE_TYPE::key().c_str(), s),
                        "Failed to load spectral density type string.");
                    remove_whitespace_and_to_lower(s);

                    return (get_alias().find(s) != get_alias().end());
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION(
                        "Failed to test whether input file type object was loadable.");
                }
            }

            static std::shared_ptr<INTERFACE_TYPE> create(const std::string &name)
            {
                std::string s(name);
                remove_whitespace_and_to_lower(s);
                auto alias_it = get_alias().find(s);
                if (alias_it != get_alias().end())
                {
                    auto it = get_info().find(alias_it->second);
                    if (it != get_info().end())
                    {
                        return it->second.m_f();
                    }
                    else
                    {
                        RAISE_EXCEPTION("Invalid type requested.");
                    }
                }
                else
                {
                    RAISE_EXCEPTION("Invalid type requested.");
                }
            }

            template <typename... Args>
            static std::shared_ptr<INTERFACE_TYPE>
            create(const typename IOWRAPPER::input_object &obj, Args &&...args)
            {
                try
                {
                    std::string s;
                    CALL_AND_HANDLE(
                        IOWRAPPER::load<std::string>(obj, INTERFACE_TYPE::key().c_str(), s),
                        "Failed to load spectral density type string.");
                    remove_whitespace_and_to_lower(s);

                    auto alias_it = get_alias().find(s);
                    if (alias_it != get_alias().end())
                    {
                        auto it = get_info().find(alias_it->second);
                        if (it != get_info().end())
                        {
                            return it->second.m_fobj(obj, std::forward<Args>(args)...);
                        }
                        else
                        {
                            RAISE_EXCEPTION("Invalid type requested.");
                        }
                    }
                    else
                    {
                        RAISE_EXCEPTION("Invalid type requested.");
                    }
                }
                catch (const std::exception &ex)
                {
                    std::cerr << ex.what() << std::endl;
                    RAISE_EXCEPTION("Failed to create object from input.");
                }
            }

            static std::string description(const std::string &name)
            {
                std::string s(name);
                remove_whitespace_and_to_lower(s);
                auto alias_it = get_alias().find(s);
                if (alias_it != get_alias().end())
                {
                    auto it = get_info().find(alias_it->second);
                    if (it != get_info().end())
                    {
                        return it->second.m_desc;
                    }
                    else
                    {
                        RAISE_EXCEPTION("Invalid type requested.");
                    }
                }
                else
                {
                    RAISE_EXCEPTION("Invalid type requested.");
                }
            }

            static std::string required_inputs(const std::string &name)
            {
                std::string s(name);
                remove_whitespace_and_to_lower(s);
                auto alias_it = get_alias().find(s);
                if (alias_it != get_alias().end())
                {
                    auto it = get_info().find(alias_it->second);
                    if (it != get_info().end())
                    {
                        return it->second.m_req;
                    }
                    else
                    {
                        RAISE_EXCEPTION("Invalid type requested.");
                    }
                }
                else
                {
                    RAISE_EXCEPTION("Invalid type requested.");
                }
            }

            static std::string get_all_info()
            {
                std::ostringstream oss;
                std::string tab("\t");
                std::string doubletab("\t\t");

                std::istringstream sstr;
                std::string line;
                for (auto it = get_info().begin(); it != get_info().end(); ++it)
                {
                    oss << it->second.m_name << ": " << std::endl;
                    std::string str = it->second.m_desc;
                    sstr.clear();
                    sstr.str(str);
                    while (std::getline(sstr, line))
                    {
                        oss << "\t" << line << std::endl;
                    }
                    str = it->second.m_req;
                    sstr.clear();
                    sstr.str(str);
                    while (std::getline(sstr, line))
                    {
                        oss << "\t\t" << line << std::endl;
                    }
                }
                return oss.str();
            }

        private:
            static std::map<std::string, std::string> &get_alias()
            {
                static std::map<std::string, std::string> m_alias;
                return m_alias;
            }

            static std::map<std::string, info_type> &get_info()
            {
                static std::map<std::string, info_type> m_info;
                return m_info;
            }
        };

        template <typename T>
        inline void FORCE_INSTANTIATE(T) {}

        template <typename INTERFACE_TYPE, typename TYPE>
        class registered_in_factory
        {
        public:
            registered_in_factory() { FORCE_INSTANTIATE(m_registered); }
            virtual ~registered_in_factory() {}

        private:
            static int m_registered;
        };

        template <typename INTERFACE_TYPE, class TYPE>
        int registered_in_factory<INTERFACE_TYPE, TYPE>::m_registered =
            factory<INTERFACE_TYPE>::register_type(
                type_info<TYPE>::get_name(), type_info<TYPE>::get_alias(),
                typename factory<INTERFACE_TYPE>::info_type{
                    type_info<TYPE>::create,
                    type_info<TYPE>::create_from_obj,
                    type_info<TYPE>::get_description(),
                    type_info<TYPE>::get_required_inputs(),
                    type_info<TYPE>::get_name(),
                });

    } // namespace io
} // namespace utils

#endif // PYTTN_UTILS_IO_FACTORY_HPP_
