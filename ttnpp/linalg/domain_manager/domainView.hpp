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
        template <typename T, typename backend = typename traits<T>::backend_type>
        class DomainView
        {
        protected:
            const T& m_src;
            const ExecDomain<backend>* m_src_domain;
            mutable std::unordered__map<const ExecDomain<backend>*, std::unique_ptr<T> > m_replicas;

        public:
            DomainView(const T& src, const ExecDomain<backend>& src_domain) : src(m_src), m_src_domain(&src_domain){}

            const T& get(Allocator<backend>* allocator = nullptr) const
            {
                auto& ctx = ExecContextScope<blas_backend>::current();
                return get(ctx.domain(), allocator);
            }

            const T& get(const ExecDomain<backend>& domain, Allocator<backend>* allocator = nullptr) const
            {
                auto& ctx = ExecContextScope<blas_backend>::current();

                auto it = m_replicas.find(&domain);
                if(it != m_replicas.end())
                {
                    return *it->second;
                }

                //functionality for creating the replica on this domain
                std::unique_ptr<T> replica ;
                const T& ref = *replica;
                m_replicas.emplace(&domain, std::move(replica));
                return ref;
            }

        protected:
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_EXECDOMAIN_HPP_
