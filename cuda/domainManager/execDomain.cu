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

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <complex>

#include "../execProps.hpp"

#include <common/exception_handling.hpp>
#include <linalg/backends/cuda/cuda_backend.cuh>

namespace linalg
{
    namespace memory
    {

        struct ExecDomain<cuda_backend>::Impl
        {
            cudaDeviceProp prop;
            std::string name;

            explicit Impl(int gpu_id)
            {
                CALL_AND_HANDLE(cuda_safe_call(cudaGetDeviceProperties(&m_impl->prop, device_id)), 
                    "Failed to construct ExecDomain<cuda_backend>::environment. Error when accessing device properties.");  
                name = dom.m_impl->name;
            }

            Impl(const Impl& o) = default;
        };


        ExecDomain<cuda_backend>::ExecDomain(int mpi_rank=0, int gpu_id=0) : m_mpi_rank(mpi_rank), m_gpu_id(gpu_id), m_impl(new Impl(gpu_id)){}
        ExecDomain<cuda_backend>::ExecDomain(const ExecDomain<cuda_backend>& o) : m_mpi_rank(o.m_mpi_rank), m_gpu_id(o.m_gpu_id), m_impl(new Impl(*o.m_impl)) {}
        ExecDomain<cuda_backend>::ExecDomain( ExecDomain<cuda_backend>&& o) : m_mpi_rank(o.m_mpi_rank), m_gpu_id(o.m_gpu_id), m_impl(o.m_impl) {o.m_impl = nullptr;}
        ExecDomain<cuda_backend>& ExecDomain<cuda_backend>::operator=(const ExecDomain<cuda_backend>& o)
        {
            if(this != &o)
            {
                m_mpi_rank = o.m_mpi_rank;
                m_gpu_id = o.m_gpu_id;
                if(m_impl != nullptr){delete m_impl;}
                m_impl = new Impl(*o.m_impl);
            }
            return *this;
        }

        ExecDomain<cuda_backend>& ExecDomain<cuda_backend>::operator=(ExecDomain<cuda_backend>&& o)
        {
            if(this != &o)
            {
                m_mpi_rank = o.m_mpi_rank;
                m_gpu_id = o.m_gpu_id;
                if(m_impl != nullptr){delete m_impl;}
                m_impl = o.m_impl;
                o.m_impl = nullptr;
            }
            return *this;
        }
        ExecDomain<cuda_backend>::~ExecDomain() 
        {
            if(m_impl != nullptr){delete m_impl;}
        }

        const std::string& ExecDomain<cuda_backend>::device_name() const noexcept{return name;}


        ExecDomain<cuda_backend>::ComputeMode ExecDomain<cuda_backend>::compute_mode() const noexcept
        {
            {
                switch (m_impl->dom.m_impl->computeMode) {
                    case cudaComputeModeDefault:
                        return ComputeMode::Default;
                    case cudaComputeModeExclusive:
                        return ComputeMode::Exclusive;
                    default:
                        return ComputeMode::Prohibited;
                }
            }
        }

        bool ExecDomain<cuda_backend>::concurrent_kernels() const noexcept{return m_impl->concurrentKernels == 1;}
        double ExecDomain<cuda_backend>::core_clock_ghz() const noexcept{return m_impl->prop.clockRate / 1.0e6;}
        double ExecDomain<cuda_backend>::memory_clock_ghz() const noexcept{return m_impl->memoryClockRate / 1.0e6;}

        size_type ExecDomain<cuda_backend>::total_global_memory() const{return m_impl->totalGlobalMem / (1.0e9);}
        size_type ExecDomain<cuda_backend>::shared_mem_per_block() const;
 
        int ExecDomain<cuda_backend>::memory_bus_width_bits() const noexcept{return m_impl->memoryBusWidth;}
        double ExecDomain<cuda_backend>::peak_memory_bandwidth_gbps() const noexcept{2.0 * m_impl->memoryClockRate * (m_impl->memoryBusWidth / 8) / 1.0e6;}
           
        int ExecDomain<cuda_backend>::warpsize() const{return m_impl->warpSize;}
        int ExecDomain<cuda_backend>::max_threads_per_block() const{return m_impl->maxThreadsPerBlock;}

        std::array<int,3> ExecDomain<cuda_backend>::max_threads_dim() const noexcept
        {
            return {m_impl->maxThreadsDim[0], m_impl->maxThreadsDim[1], m_impl->maxThreadsDim[2]};
        }
        std::array<int,3> ExecDomain<cuda_backend>::max_grid_size() const noexcept
        {
            return {m_impl->maxGridSize[0], m_impl->maxGridSize[1], m_impl->maxGridSize[2]};
        }

        //utilities for accessing properties about the types of devices
        static int ExecDomain<cuda_backend>::number_of_devices()
        {
            int nDevices;
            cudaGetDeviceCount(&nDevices);
            return nDevices;
        }

        static std::ostream& ExecDomain<cuda_backend>::list_devices(std::ostream &out)
        {
            int n = number_of_devices();
            for(int i = 0; i < n; ++i)
            {
                ExecDomain dom(0, i);
                out << "Device: " << i << std::endl;
                out << dom << std::endl;
            }
            return out;
        }

        friend std::ostream& operator<<(std::ostream& out, const ExecDomain<cuda_backend>& dom)
        {
            out << "\tDevice Name: " << dom.device_name() << std::endl;
            out << "\tCompute Capability: " << dom.m_impl->major << "." << dom.m_impl->minor << std::endl;
            out << "\tCompute Mode: " << (dom.m_impl->computeMode == Domain<cuda_backend>::ComputeMode::Default ? "Multithreaded" : (dom.m_impl->computeMode == ExecDomain<cuda_backend>::ComputeMode::Exclusive ? "Singlethreaded" : "No")) << " Device Access" << std::endl;
            out << "\tConcurrent Kernel Execution: " << (dom.concurrent_kernels() ? "True" : "False") << std::endl
                << std::endl;
            out << "\tClock Speed (GHz): " << dom.core_clock_ghz() << std::endl
                << std::endl;
            out << "\tMemory Clock Speed (GHz): " << dom.memory_clock_ghz() << std::endl;
            out << "\tTotal Global Memory (GB): " << dom.total_global_memory << std::endl;
            out << "\tMemory Bus Width (bits): " << dom.memory_bus_width_bits() << std::endl;
            out << "\tPeak Memory Bandwidth (GB/s): " << dom.peak_memory_bandwidth_gbps() << std::endl
                << std::endl;
            out << "\tWarp Size: " << dom.warpsize() << std::endl;
            out << "\tMaximum Threads Per Block: " << dom.max_threads_per_block() << std::endl;

            auto td = dom.max_threads_dim();
            auto gd = dom.max_grid_size();


            out << "\tMaximum Thread Dimensions: (" << td[0] << ", " << td[1] << ", " << td[2] << ")" << std::endl;
            out << "\tMaximum Grid Size: (" << gd[0] << ", " << gd[1] << ", " <<gd[2] << ")" << std::endl;
        }

    }
}
