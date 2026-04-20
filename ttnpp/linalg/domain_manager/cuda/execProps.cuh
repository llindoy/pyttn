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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_CUDA_EXECPROPERTIES_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_CUDA_EXECPROPERTIES_HPP_

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <complex>
#include "../../backends/cuda/cuda_backend.cuh"
#include "cuda_environment.hpp"
#include "../execProps.hpp"

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
        
        public:
            ExecDomain() : m_mpi_rank(0) {}
            ExecDomain(int mpi_rank) : m_mpi_rank(mpi_rank){}
            ExecDomain(const ExecDomain& o) = default;
            ExecDomain(ExecDomain&& o) = default;

            ExecDomain& operator=(const ExecDomain& o) = default;
            ExecDomain& operator=(ExecDomain&& o) = default;

            bool operator==(const ExecDomain& o)
            {
                //if the Domains have the same rank and if (gpu_id is less than 0 check that the other is less than zero (i.e. both cpu devices).  Otherwise check that they are both on the same gpu).
                return m_gpu_id == o.m_gpu_id;
            }

            const int& mpi_rank() const{return m_mpi_rank;}
            int& mpi_rank(){return m_mpi_rank;}
        };

        /* A class for handling information about device information for operators acting on linalg objects.*/
        template <>
        class ExecContext<cuda_backend>
        {
        protected:
            const ExecDomain<cuda_backend>* m_domain;
            cudaStream_t m_stream;
            std::size_t m_nstreams;
            std::size_t m_id;
        
            friend class SerialScheduler<cuda_backend>;
            friend class ParallelScheduler<cuda_backend>;

        public:
            ExecContext(const ExecDomain<cuda_backend>& domain, std::size_t nstreams = 1, std::size_t id = 0) : m_domain(&domain),  m_nstreams(nstreams), m_id(id) {}
            ExecContext(const ExecContext& o) = default;
            ExecContext(ExecContext&& o) = default;
            ExecContext& operator=(const ExecContext& o) = default;
            ExecContext& operator=(ExecContext&& o) = default;

            const ExecDomain<cuda_backend>& domain() const {return *m_domain;}
            cudaStream_t stream() const{return m_stream;}
            std::size_t nstreams() const{return m_nstreams;}
            std::size_t id() const{return m_id;}
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_CUDA_EXECPROPERTIES_HPP_
