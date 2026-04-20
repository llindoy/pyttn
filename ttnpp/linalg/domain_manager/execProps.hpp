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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_EXECDOMAIN_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_EXECDOMAIN_HPP_

#include <cstdint>
#include <cstddef>

namespace linalg
{
    namespace memory
    {
        /* The abstract allocator class for handling memory allocation within the linalg library*/
        template <typename backend>
        class ExecDomain;

        //a class storing the exe
        template <typename backend>
        class ExecContext;

        template <typename backend>
        class ExecContextScope
        {
        public:
            explicit ExecContextScope(ExecContext<backend>& ctx)
            {
                m_prev = s_current;
                s_current = &ctx;
            }

            ~ExecContextScope()
            {
                s_current = m_prev;
            }

            static bool is_active()
            {
                return s_current != nullptr;
            }

            static ExecContext<backend>& current() 
            {
                ASSERT(s_current != nullptr, "ExecContext accessed outside of scheduler execution.");
                return *s_current;
            }

        protected:
            static thread_local ExecContext<backend>* s_current;
            ExecContext<backend>* m_prev;
        };

        template <typename backend>
        thread_local ExecContext<backend>* ExecContextScope<backend>::s_current = nullptr;

        template <typename backend>
        class ExecScheduler
        {
        public:
            virtual ~ExecScheduler() = default;

            virtual std::size_t width() const = 0;
            virtual ExecContext<backend>& context(std::size_t i) = 0;
        };

        template <typename backend>
        class SerialScheduler;

        template <typename backend>
        class ParallelScheduler;

        class DomainRange
        {
        protected:
            std::size_t m_begin;
            std::size_t m_end;
        
        public:
            DomainRange(std::size_t begin, std::size_t end) : m_begin(begin), m_end(end){}
            DomainRange(const DomainRange& o) = default;
            DomainRange(DomainRange&& o) = default;
            DomainRange& operator=(const DomainRange& o) = default;
            DomainRange& operator=(DomainRange&& o) = default;

            bool empty() const{return m_begin >= m_end;}
            std::size_t begin() const{return m_begin;}
            std::size_t end() const{return m_end;}
        };

        template <typename backend>
        class DomainPartition
        {
        protected:
            std::vector<const ExecDomain<backend>*> m_domains;
            std::vector<DomainRange> m_ranges;
            std::unordered_map<const ExecDomain<backend>*, std::size_t> m_index;

        public:
            DomainPartition(){}
            DomainPartition(const DomainPartition& o) = default;
            DomainPartition(DomainPartition&& o) = default;
            DomainPartition& operator=(const DomainPartition& o) = default;
            DomainPartition& operator=(DomainPartition&& o) = default;

            void add_domain(const ExecDomain<backend>* domain, const DomainRange& range)
            {
                ASSERT(domain != nullptr, "Cannot insert empty domain into DomainPartition.");
                ASSERT(m_index.count(domain) == 0, "Domain already present in Partition.");
                std::size_t index = m_domains.size();
                m_domains.push_back(domain);
                m_ranges.push_back(range);
                m_index[domain] = index;
            }

            const std::vector<const ExecDomain<backend>*>& domains() const{return m_domains;}

            const DomainRange& range(const ExecDomain<backend>& domain) const
            {
                auto it = m_index.find(domain);
                ASSERT(it != m_index.end(), "Domain not found in DomainPartition.");
                return m_ranges[it->second];
            }

            bool has_domain(const ExecDomain<backend>& domain) const
            {
                return m_index.count(domain) != 0;
            }
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_EXECDOMAIN_HPP_
