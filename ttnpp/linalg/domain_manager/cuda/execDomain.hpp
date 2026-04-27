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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_CUDA_EXEC_DOMAIN_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_CUDA_EXEC_DOMAIN_HPP_

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <complex>

#include "../execProps.hpp"
#include "../../backends/cuda/cuda_backend.hpp"

namespace linalg
{
    namespace memory
    {
        /* A class for handling information about device information for a linalg object.*/
        template <>
        class ExecDomain<cuda_backend>
        {
        protected:
            int m_mpi_rank;     //purely meta data
            int m_gpu_id;

            struct Impl;
            Impl* m_impl;  // opaque, CUDA-backe

        public:
            ExecDomain(int mpi_rank=0, int gpu_id=0);
            ExecDomain(const ExecDomain& o);
            ExecDomain(ExecDomain&& o) noexcept;
            ~ExecDomain();
            ExecDomain& operator=(const ExecDomain& o);
            ExecDomain& operator=(ExecDomain&& o) noexcept;
            

            bool operator==(const ExecDomain& o)
            {
                //if the Domains have the same rank and if (gpu_id is less than 0 check that the other is less than zero (i.e. both cpu devices).  Otherwise check that they are both on the same gpu).
                return m_gpu_id == o.m_gpu_id;
            }

            int mpi_rank() const noexcept{return m_mpi_rank;}
            int gpu_id() const noexcept{return m_gpu_id;}

            const std::string& device_name() const noexcept;


            enum class ComputeMode {
                Default,
                Exclusive,
                Prohibited
            };

            ComputeMode compute_mode() const noexcept;
            bool concurrent_kernels() const noexcept;


            double core_clock_ghz() const noexcept;
            double memory_clock_ghz() const noexcept;

            size_type total_global_memory() const; 
            int memory_bus_width_bits() const noexcept;
            double peak_memory_bandwidth_gbps() const noexcept;
           
            int warpsize() const;
            int max_threads_per_block() const;

            std::array<int,3> max_threads_dim() const noexcept;
            std::array<int,3> max_grid_size() const noexcept;

            //utilities for accessing properties about the types of devices
            static int number_of_devices();
            static std::ostream &list_devices(std::ostream &out);

            friend std::ostream& operator<<(std::ostream& out, const ExecDomain& dom);

        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_CUDA_EXEC_DOMAIN_HPP_
