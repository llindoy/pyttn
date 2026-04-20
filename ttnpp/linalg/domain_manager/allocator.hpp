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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_ALLOCATOR_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_ALLOCATOR_HPP_

#include <cstdint>
#include <cstddef>
#include "execProps.hpp"

namespace linalg
{
    namespace memory
    {
        template <typename T, typename backend>
        struct alignment_for_type
        {
            std::size_t eval(const ExecDomain<backend>& domain)
            {
                return std::alignof(T);
            }
        };

        /* The abstract allocator class for handling memory allocation within the linalg library*/
        template <typename backend>
        class Allocator
        {
        protected:
            const ExecDomain<backend>* m_domain;

        public:
            Allocator(const ExecDomain<backend>& domain) : m_domain(&domain){}

            virtual void* allocate_bytes(std::size_t nbytes, std::size_t alignment) = 0;
            virtual void deallocate_bytes(void *v, std::size_t alignment) = 0;
            const ExecDomain<backend>& domain() const{return *m_domain;}

            template <typename T>
            T* allocate(std::size_t n)
            {
                static_assert(std::is_trivially_copyable<T>::value, "Allocation requires trivially copyable T");
                const std::size_t alignment = alignment_for_type<T>::eval(domain());
                return static_cast<T*>(allocate_bytes(n*sizeof<T>, alignment));
            }           

            template <typename T> 
            void deallocate(T* ptr)
            {
                if(!ptr){return;}
                const std::size_t alignment = alignment_for_type<T>::eval(domain());
                deallocate_bytes(ptr, alignment);

            }
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_ALLOCATOR_HPP_
