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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_CUDA_SCHEDULER_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_CUDA_SCHEDULER_HPP_

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <complex>
#include "../../backends/cuda/cuda_backend.cuh"
#include "cuda_environment.hpp"
#include "../scheduler.hpp"
#include "execProps.cuh"

namespace linalg
{
    namespace memory
    {
        //single stream gpu scheduler. 
        template <>
        class SerialScheduler<cuda_backend> : public ExecScheduler<cuda_backend>
        {
        protected:
            ExecContext<cuda_backend> m_ctx;
            std::size_t m_barrier_depth;

        public:
            SerialScheduler(const ExecDomain<cuda_backend>& dom) : m_ctx{dom, 1, 0}, m_barrier_depth(0)
            {
                int gpu_id = m_ctx.domain().gpu_id();
                CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(gpu_id)), "Failed to set correct cuda device for scheduler");
                CALL_AND_HANDLE(cuda_safe_call(cudaStreamCreate(&(m_ctx.m_stream))), "Failed to create cudaStream.");
            }
            ~SerialScheduler()
            {
                CALL_AND_HANDLE(cuda_safe_call(cudaStreamDestroy(m_ctx.m_stream)), "Failed to create cudaStream.");
            }
            SerialScheduler(const SerialScheduler& o) = delete;
            SerialScheduler(SerialScheduler&& o) = delete;

            SerialScheduler& operator=(const SerialScheduler& o) = delete;
            SerialScheduler& operator=(SerialScheduler&& o) = delete;

            std::size_t lane_count() const{return 1;}
            ExecContext<cuda_backend>& context(std::size_t lane) override
            {
                ASSERT(lane == 0, "SerialScheduler only has one lane.");
                return m_ctx;
            }

            template <typename Fn>
            void execute(const Domain& domain, Fn&& fn)
            {
                ASSERT(domain == m_ctx.domain(), "Failed to execute task.  Invalid domain.");
                int gpu_id = m_ctx.domain().gpu_id();
                CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(gpu_id)), "Failed to set correct cuda device for scheduler");
                ExecContextScope<cuda_backend> scope(m_ctx);
                fn();
            }

            template <typename Fn>
            void for_each(const DomainPartition<cuda_backend>& partitions, Fn&& fn)
            {
                ASSERT(partitions.domains().size() == 1, "Serial Scheduler is only valid over a single domain.");
                ASSERT(partitions.has_domain(m_ctx.domain()), "The Serial Scheduler is not able to work with the domain in the partition.");
                const DomainRange& range = partitions.range(m_ctx.domain());

                int gpu_id = m_ctx.domain().gpu_id();
                CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(gpu_id)), "Failed to set correct cuda device for scheduler");

                for(size_t i = range.begin(); i < range.end(); ++i)
                {
                    ExecContextScope<cuda_backend> scope(m_ctx);
                    fn(i);
                }            
            }

            //barrier control.  no-op as everything is serial
            void synchronise()
            {
                int gpu_id = m_ctx.domain().gpu_id();
                CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(gpu_id)), "Failed to set correct cuda device for scheduler");
                CALL_AND_HANDLE(cuda_safe_call(cudaStreamSynchronize(m_ctx.m_stream)), "Failed when synchronising cuda stream.");
                
            }

            template <typename T, typename Op>
            void reduce_into(const std::vector<T>& locals,  T& global, Op&& combine)
            {
                synchronise();
                global = locals[0];
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


        //multi stream gpu scheduler.  This does not support 
        template <>
        class ParallelScheduler<cuda_backend> : public ExecScheduler<cuda_backend>
        {
        protected:
            std::vector<ExecContext<cuda_backend>> m_ctx;
            cudaStream_t m_reduce_stream; 

            std::vector<cudaEvent_t> m_events;
            std::vector<bool> m_stream_active;
            std::size_t m_nstreams;
            std::size_t m_barrier_depth;
            std::size_t m_current_stream;

        public:
            ParallelScheduler(const ExecDomain<cuda_backend>& dom, std::size_t nstreams) : m_nstreams(nstreams), m_stream_active(nstreams, false), m_barrier_depth(0), m_current_stream(0)
            {
                ASSERT(nstreams > 0, "Cannot create ParallelScheduler with 0 streams.");
                int gpu_id = dom.gpu_id();
                CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(gpu_id)), "Failed to set correct cuda device for scheduler");
                m_ctx.resize(nstreams);
                for(size_t i = 0; i < nstreams; ++i)
                {
                    m_ctx[i] = ExecContext<cuda_backend>(dom, nstreams, i);
                    CALL_AND_HANDLE(cuda_safe_call(cudaStreamCreate(&(m_ctx[i].m_stream))), "Failed to create cudaStream.");
                }
                //create the streams and events
                m_events.resize(nstreams);

                for(std::size_t i = 0; i < nstreams; ++i)
                {
                    CALL_AND_HANDLE(cuda_safe_call(cudaEventCreateWithFlags(&m_events[i], cudaEventDisableTimings)), "Failed to create cudaEvent.");
                }

                //now create the stream for performing the reduction operations
                CALL_AND_HANDLE(cuda_safe_call(cudaStreamCreate(&m_reduce_stream)), "Failed to create cudaStream.");

            }
            ~ParallelScheduler()
            {
                CALL_AND_HANDLE(cuda_safe_call(cudaStreamDestroy(m_reduce_stream)), "Failed to create cudaStream.");
                for(std::size_t i = 0; i < m_events.size(); ++i)
                {
                    CALL_AND_HANDLE(cuda_safe_call(cudaEventDestroy(m_events[i])), "Failed to destroy cudaEvent.");
                }
                for(std::size_t i = 0; i < m_ctx.size(); ++i)
                {
                    CALL_AND_HANDLE(cuda_safe_call(cudaStreamDestroy(m_ctx[i].m_stream)), "Failed to destroy cudaStream.");
                }
            }
            ParallelScheduler(const ParallelScheduler& o) = delete;
            ParallelScheduler(ParallelScheduler&& o) = delete;

            ParallelScheduler& operator=(const ParallelScheduler& o) = delete;
            ParallelScheduler& operator=(ParallelScheduler&& o) = delete;

            std::size_t lane_count() const{return m_nstreams;}
            ExecContext<cuda_backend>& context(std::size_t lane) override
            {
                ASSERT(lane < m_nstreams, "ParallelScheduler lane out of bounds.");
                return m_ctx[lane];
            }
            template <typename Fn>
            void execute(const Domain& domain, Fn&& fn)
            {
                ASSERT(domain == m_ctx[0].domain(), "Failed to execute task.  Invalid domain.");

                int gpu_id = m_ctx[0].domain().gpu_id();
                CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(gpu_id)), "Failed to set correct cuda device for scheduler");

                std::size_t sid = m_current_stream;
                m_current_stream = (m_current_stream + 1) % m_nstreams;

                if (m_stream_active[sid]) 
                {
                     CALL_AND_HANDLE(cuda_safe_call(cudaStreamWaitEvent(m_ctx[sid].m_stream, m_events[sid], 0)), "Failed to tell stream to wait for event to complete before starting next item");
                }

                {
                    ExecContextScope<cuda_backend> scope(m_ctx[sid]);
                    fn();
                }

                CALL_AND_HANDLE(cuda_safe_call(cudaEventRecord(m_events[sid], m_ctx[sid].m_stream)), "Failed to record event.");
                m_stream_active[sid] = true;
            }

            template <typename Fn>
            void for_each(const DomainPartition<cuda_backend>& partitions, Fn&& fn)
            {
                ASSERT(partitions.domains().size() == 1, "Serial Scheduler is only valid over a single domain.");
                ASSERT(partitions.has_domain(m_ctx[0].domain()), "The Serial Scheduler is not able to work with the domain in the partition.");
                const DomainRange& range = partitions.range(m_ctx[0].domain());

                int gpu_id = m_ctx[0].domain().gpu_id();
                CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(gpu_id)), "Failed to set correct cuda device for scheduler");

                for(size_t i = range.begin(); i < range.end(); ++i)
                {
                    std::size_t sid = m_current_stream;
                    m_current_stream = (m_current_stream + 1) % m_nstreams;

                    if (m_stream_active[sid]) 
                    {
                        CALL_AND_HANDLE(cuda_safe_call(cudaStreamWaitEvent(m_ctx[sid].m_stream, m_events[sid], 0)), "Failed to tell stream to wait for event to complete before starting next item");
                    }                    
                    
                    {
                        ExecContextScope<cuda_backend> scope(m_ctx[sid]);
                        fn(i);
                    }

                    CALL_AND_HANDLE(cuda_safe_call(cudaEventRecord(m_events[sid], m_ctx[sid].m_stream)), "Failed to record event.");
                    m_stream_active[sid] = true;
                }            
            }

            //barrier control.  no-op as everything is serial
            void synchronise()
            {
                int gpu_id = m_ctx[0].domain().gpu_id();
                CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(gpu_id)), "Failed to set correct cuda device for scheduler");

                for (size_t i = 0; i < m_nstreams; ++i) 
                {
                    if (m_stream_active[i]) 
                    {
                        CALL_AND_HANDLE(cuda_safe_call(cudaEventSynchronize(m_events[i])), "Failed to synchronise events");
                        m_stream_active[i] = false;
                    }
                }
                 CALL_AND_HANDLE(cuda_safe_call(cudaStreamSynchronize(m_reduce_stream)), "Failed to synchronise reduction stream");
            }


            // PRECONDITION:
            //   locals[i] contains the reduction value produced by stream/lane i.
            // POSTCONDITION:
            //   global contains the reduction of all locals after full synchronisation.
            template <typename T, typename Op>
            void reduce_into(const std::vector<T>& locals, T& global, Op&& combine)
            {
                synchronise();
                global = locals[0];
            
                for (std::size_t i = 1; i < locals.size(); ++i) {
                    combine(global, locals[i]);
                }

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

    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_CUDA_SCHEDULER_HPP_
