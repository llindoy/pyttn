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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_ALLOCATOR_RECORD_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_ALLOCATOR_RECORD_HPP_

#include <cstdint>
#include <cstddef>
#include <memory>
#include "execProps.hpp"
#include <common/exception_handling.hpp>

namespace linalg
{
    namespace memory
    {

        /**
         * An EvictionSubscription represents an object that wishes to be notified
         * immediately before an allocator invalidates (overwrites) a specific
         * allocation region.
         *
         * Rules:
         *  - The callback is host-side and synchronous.
         *  - The callback must be noexcept.
         *  - The allocator does not depend on the callback for correctness.
         *  
         */
        class evictionSubscription
        {
        public:
            virtual void on_evict() noexcept = 0;
        protected:
            ~evictionSubscription() = default;
        };

        class allocationRecord
        {
        protected:
            std::size_t m_offset;
            std::size_t m_size;

            std::weak_ptr<evictionSubscription> m_subscriber;
        public:
            allocationRecord(std::size_t offset, std::size_t size) noexcept : m_offset(offset), m_size(size){}
            allocationRecord(const allocationRecord& o) = delete;
            allocationRecord& operator=(const allocationRecord& o) = delete;

            allocationRecord(allocationRecord&& o) = default;
            allocationRecord& operator=(allocationRecord&& o) = default;

            //basic accessors
            std::size_t offset() const noexcept{return m_offset;}
            std::size_t size() const noexcept{return m_size;}

            //subscriber handling
            bool has_subscriber() const noexcept{return !m_subscriber.expired();}

            void set_subscriber(std::weak_ptr<evictionSubscription> sub)
            {
                ASSERT(m_subscriber.expired(), "allocationRecord already has a subscriber");
                m_subscriber = std::move(sub);
            }

            void clear_subscriber(){m_subscriber.reset();}

            //eviction logic 
            /**
             * Called by the allocator immediately before this allocation
             * is invalidated and reused.
             *
             * This method:
             *  - notifies the subscriber (if any)
             *  - clears the subscription
             */
            void evict()
            {
                if(auto sub = m_subscriber.lock())
                {
                    CALL_AND_HANDLE(sub->on_evict(), "Failed when calling eviction logic");;
                }

                m_subscriber.reset();
            }
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_ALLOCATOR_RECORD_HPP_
