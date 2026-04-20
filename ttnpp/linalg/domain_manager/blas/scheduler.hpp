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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_BLAS_SCHEDULER_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_BLAS_SCHEDULER_HPP_

#include <cstdint>
#include <cstdlib>
#include <common/exception_handling.hpp>
#include "../../backends/blas/blas_backend.hpp"
#include "../scheduler.hpp"
#include "execProps.hpp"

namespace linalg
{
    namespace memory
    {
        template <>
        class SerialScheduler<blas_backend> : public ExecScheduler<blas_backend>
        {
        protected:
            ExecContext<blas_backend> m_ctx;
            using Domain = ExecDomain<blas_backend>;

        public:
            SerialScheduler(const ExecDomain<blas_backend>& dom) : m_ctx{dom, 1, 0} {}
            SerialScheduler(const SerialScheduler& o) = default;
            SerialScheduler(SerialScheduler&& o) = default;

            SerialScheduler& operator=(const SerialScheduler& o) = default;
            SerialScheduler& operator=(SerialScheduler&& o) = default;

            std::size_t lane_count() const{return 1;}

            template <typename Fn>
            void execute(const Domain& domain, Fn&& fn)
            {
                ASSERT(domain == m_ctx.domain(), "Failed to execute task.  Invalid domain.");
                ExecContextScope<blas_backend> scope(m_ctx);
                fn();
            }

            template <typename Fn>
            void for_each(const DomainPartition<blas_backend>& partition, Fn&& fn)
            {
                ASSERT(partition.domains().size() == 1, "Serial Scheduler is only valid over a single domain.");
                ASSERT(partition.has_domain(m_ctx.domain()), "The Serial Scheduler is not able to work with the domain in the partition.");
                const DomainRange& range = partitions.range(*domain);

                for(size_t i = range.begin(); i < range.end(); ++i)
                {
                    ExecContextScope<blas_backend> scope(m_ctx);
                    fn(i);
                }
            }

            //barrier control.  no-op as everything is serial
            void synchronise(){}

            template <typename T, typename Op>
            void reduce_into(const std::vector<T>& locals,  T& global, Op&& combine){global = locals[0];}

            //barrier control.  no-op as everything is serial
            void enter_barrier(){}
            void exit_barrier(){}
        };

//#ifdef USE_OPENMP
        template <>
        class ParallelScheduler<blas_backend> : public ExecScheduler<blas_backend>
        {
        public:
            using Context = ExecContext<blas_backend>;
            using Domain = ExecDomain<blas_backend>;
        protected:
            Context m_ctx;
            std::size_t m_barrier_depth;
            std::size_t m_nthreads;
            std::size_t m_grainsize;
            bool m_pool_active;

        private:
            void start_pool()
            {
                #pragma omp parallel num_threads(m_nthreads)
                {
                    #pragma omp single
                    {
                        m_pool_active = true;
                    }

                    while (m_pool_active) 
                    {
                        #pragma omp taskyield
                    }
                }
            }

            void shutdown()
            {
                m_pool_active = false;

                #pragma omp taskwait
            }

        public:
            ParallelScheduler(const ExecDomain<blas_backend>& dom, std::size_t nthreads, std::size_t grainsize = 128) : m_ctx{dom, nthreads}, m_barrier_depth(0), m_nthreads(nthreads), m_grainsize(grainsize), m_pool_active(false)
            {
                ASSERT(nthreads > 0, "Cannot allocate parallel scheduler with only a single thread.");
                start_pool();
            }
            ~ParallelScheduler()
            {
                shutdown();
            }
        
            ParallelScheduler(const ParallelScheduler& o) = default;
            ParallelScheduler(ParallelScheduler&& o) = default;

            ParallelScheduler& operator=(const ParallelScheduler& o) = default;
            ParallelScheduler& operator=(ParallelScheduler&& o) = default;

            std::size_t lane_count() const{return m_nthreads;}

            template <typename Fn>
            void execute(const Domain& domain, Fn&& fn)
            {
                ASSERT(domain == m_ctx.domain(), "Failed to execute task.  Invalid domain.");

                #pragma omp task
                {
                    Context local = m_ctx;
                    local.id() = omp_get_thread_num();
                    local.nthreads() = omp_get_num_threads();
                    ExecContextScope<blas_backend> scope(local);

                    fn();
                }
            }

            template < typename Fn>
            void for_each(const DomainPartition<blas_backend>& partition, Fn&& fn)
            {
                ASSERT(partition.domains().size() == 1, "Serial Scheduler is only valid over a single domain.");
                ASSERT(partition.has_domain(m_ctx.domain()), "The Serial Scheduler is not able to work with the domain in the partition.");

                const DomainRange& range = partitions.range(*domain);
                    
                #pragma omp taskloop grainsize(m_grainsize)
                for(Index i = begin; i < end; ++i)
                {
                    Context local = m_ctx;
                    local.id() = omp_get_thread_num();
                    local.nthreads() = omp_get_num_threads();
                    ExecContextScope<blas_backend> scope(local);

                    fn(i);
                }
            }

            //barrier control.  no-op as everything is serial
            void synchronise()
            {
                #pragma omp taskwait
            }

            //reduce the result of any earlier call e.g. for_each or execute stored in locals into the final result global using the 
            //combine op.  Here we force synchronisation in order to ensure that the data in the locals is correct.
            template <typename T, typename Op>
            void reduce_into(const std::vector<T>& locals, T& global, Op&& combine)
            {
                synchronise();

                global = locals[0];
                for(std::size_t i = 1; i < locals.size(); ++i)
                {
                    combine(global, locals[i]);
                }
                enter_barrier();
                exit_barrier();
            }

            //barrier control. 
            void enter_barrier(){++m_barrier_depth;}
            void exit_barrier()
            {
                ASSERT(m_barrier_depth != 0, "Failed when exiting barrier.  More barriers have been exited than entered.");
                --m_barrier_depth;
                if(m_barrier_depth == 0)
                {
                    synchronise();
                }
            }
        };
//#endif

        template <>
        class MultiDeviceScheduler<blas_backend> : public ExecScheduler<blas_backend>
        {
        public:
            MultiDeviceScheduler(const ExecDomain<blas_backend>&) {RAISE_EXCEPTION("MultiDeviceScheduler is not supported for blas_backend.");}
            std::size_t lane_count() const{RAISE_EXCEPTION("MultiDeviceScheduler is not supported for blas_backend.");}

            template <typename Fn>
            void execute(const Domain&, Fn&&){RAISE_EXCEPTION("MultiDeviceScheduler is not supported for blas_backend.");}

            template <typename Fn>
            void for_each(const DomainPartition<blas_backend>&, Fn&& ){RAISE_EXCEPTION("MultiDeviceScheduler is not supported for blas_backend.");}

            //barrier control.  no-op as everything is serial
            void synchronise(){RAISE_EXCEPTION("MultiDeviceScheduler is not supported for blas_backend.");}

            template <typename T, typename Op>
            void reduce_into(const std::vector<T>&, T&, Op&&){RAISE_EXCEPTION("MultiDeviceScheduler is not supported for blas_backend.");}

            //barrier control.  no-op as everything is serial
            void enter_barrier(){RAISE_EXCEPTION("MultiDeviceScheduler is not supported for blas_backend.");}
            void exit_barrier(){RAISE_EXCEPTION("MultiDeviceScheduler is not supported for blas_backend.");}
        };


    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_BLAS_SCHEDULER_HPP_
