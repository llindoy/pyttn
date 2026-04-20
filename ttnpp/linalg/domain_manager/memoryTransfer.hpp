/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2026 NPL Management Limited
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
 
#ifndef PYTTN_LINALG_DOMAIN_MANAGER_DOMAIN_PARTITION_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_DOMAIN_PARTITION_HPP_

#include <cstdint>
#include <cstddef>
#include <memory>
#include <unordered_map>

#include "../../linalg_forward_decl.hpp"

namespace linalg
{
    namespace memory
    {
        template <typename src_backend, typename dst_backend>
        class MemoryTransfer
        {
            using size_type = typename traits<B1>::size_type;

        public:
            template <typename T>
            static void copy(const T *const src, size_type n, T *dest);

        };

    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_EXECDOMAIN_HPP_
