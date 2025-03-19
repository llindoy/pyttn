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

#ifndef PYTTN_UTILS_IO_COMMON_HPP_
#define PYTTN_UTILS_IO_COMMON_HPP_

#include <linalg/utils/linalg_utils.hpp>
#include <common/exception_handling.hpp>
#include <memory>

namespace utils
{
    namespace io
    {
        template <typename T, typename... Args>
        std::unique_ptr<T> make_unique(Args &&...args) { return std::unique_ptr<T>(new T(std::forward<Args>(args)...)); }

        template <typename T>
        std::shared_ptr<T> make_shared() { return std::shared_ptr<T>(new T()); }

        template <typename T, typename... Args>
        std::shared_ptr<T> make_shared(Args &&...args)
        {
            return std::shared_ptr<T>(new T(std::forward<Args>(args)...));
        }

    } // namespace io
} // namespace utils

#endif // PYTTN_UTILS_IO_COMMON_HPP_
