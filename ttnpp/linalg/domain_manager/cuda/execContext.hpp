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

#ifndef PYTTN_LINALG_DOMAIN_MANAGER_CUDA_EXEC_CONTEXT_HPP_
#define PYTTN_LINALG_DOMAIN_MANAGER_CUDA_EXEC_CONTEXT_HPP_

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <complex>

#include "execDomain.hpp"

#include "../../backends/cuda/cuda_backend.hpp"
#include "../../backends/cuda/cublas_wrapper.cuh"
#include "../../backends/cuda/cusolver_wrapper.cuh"
#include "../../backends/cuda/cusparse_wrapper.cuh"
#include "../../backends/cuda/cutensor_wrapper.cuh"

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
            ExecDomain(int mpi_rank=0, int gpu_id=0) : m_mpi_rank(mpi_rank), m_gpu_id(gpu_id){}
            ExecDomain(const ExecDomain& o) = default;
            ExecDomain(ExecDomain&& o) = default;

            bool operator==(const ExecDomain& o)
            {
                //if the Domains have the same rank and if (gpu_id is less than 0 check that the other is less than zero (i.e. both cpu devices).  Otherwise check that they are both on the same gpu).
                return m_gpu_id == o.m_gpu_id;
            }

            int mpi_rank() const{return m_mpi_rank;}
            int gpu_id() const{return m_gpu_id;}

        };

        /* A class for handling information about device information for operators acting on linalg objects.*/
        template <>
        class ExecContext<cuda_backend>
        {
        protected:
            const ExecDomain<cuda_backend>& m_domain;

            struct Impl;
            std::unique_ptr<Impl> m_impl;


            cudaStream_t m_stream;
            cudaEvent_t m_event;
            bool m_active;
            std::size_t m_id;

            cusparseHandle_t cusparse_handle;
            cublasHandle_t cublas_handle;
            cusolverDnHandle_t cusolver_dn_handle;
            cutensorHandle_t cutensor_handle;
        
            friend class SerialScheduler<cuda_backend>;
            friend class ParallelScheduler<cuda_backend>;

        public:
            ExecContext(const ExecDomain<cuda_backend>& domain, std::size_t id = 0);s
            ExecContext(const ExecContext& o) = delete;
            ExecContext(ExecContext&& o) = default;
            ExecContext& operator=(const ExecContext& o) = delete;
            ExecContext& operator=(ExecContext&& o) = delete;

            const ExecDomain<cuda_backend>& domain() const {return m_domain;}
            cudaStream_t stream() const{return m_stream;}
            std::size_t id() const{return m_id;}
        };
    }
}

#endif //PYTTN_LINALG_DOMAIN_MANAGER_CUDA_EXEC_CONTEXT_HPP_
