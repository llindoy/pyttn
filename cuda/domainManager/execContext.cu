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

#include "execDomain.hpp"
#include <common/exception_handling.hpp>

#include <linalg/backends/cuda/cuda_backend.cuh>
#include <linalg/domain_manager/cuda/execContext.hpp>
#include <linalg/backends/cuda/cublas_wrapper.cuh>
#include <linalg/backends/cuda/cuda_backend.cuh>
#include <linalg/backends/cuda/cusolver_wrapper.cuh>
#include <linalg/backends/cuda/cusparse_wrapper.cuh>
#include <linalg/backends/cuda/cutensor_wrapper.cuh>

namespace linalg
{
    namespace memory
    {
        struct ExecContext<cuda_backend>::Impl
        {
            cudaStream_t stream;
            cudaEvent_t event;
            bool active;
            std::size_t id;

            cusparseHandle_t cusparse_handle;
            cublasHandle_t cublas_handle;
            cusolverDnHandle_t cusolver_dn_handle;
            cutensorHandle_t cutensor_handle;

            Impl(const ExecDomain<cuda_backend>& domain, const std::size_t l) : id(l), active(false)
            {
                try
                {
                    CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(domain.device_id())), "Failed to set cuda Device");
                    
                    CALL_AND_HANDLE(cuda_safe_call(cudaStreamCreate(&stream)), "Failed to create cudaStream");
                    CALL_AND_HANDLE(cuda_safe_call(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)), "Failed to create cudaEvent");

                    CALL_AND_HANDLE(cublas_safe_call(cublasCreate(&cublas_handle)), "Error when setting up cublas_handle object.");
                    CALL_AND_HANDLE(cusparse_safe_call(cusparseCreate(&cusparse_handle)), "Error when setting up the cusparse_handle object.");
                    CALL_AND_HANDLE(cusolver_safe_call(cusolverDnCreate(&cusolver_dn_handle)), "Error when setting up cusolver_dn_handle object.");
                    CALL_AND_HANDLE(cutensor_safe_call(cutensorCreate(&cutensor_handle)), "Error when setting up cutensor_handle object.");        
                }
                catch(const std::exception& ex)
                {
                    logging::error(ex.what());
                    RAISE_EXCEPTION("Failed to construct ExecContext<cuda_backend>::Impl");
                }    
            }

            ~Impl()
            {
                cudaStreamDestroy(stream);
                cudaEventDestroy(event);

                cublasDestroy(cublas_handle);
                cusolverDnDestroy(cusolver_dn_handle);
                cusparseDestroy(cusparse_handle);
                cutensorDestroy(cutensor_handle);
            }
        };

        /* A class for handling information about device information for operators acting on linalg objects.*/
        ExecContext<cuda_backend>::ExecContext(const ExecDomain<cuda_backend>& domain, std::size_t id = 0) : m_domain(domain), m_impl(std::make_unique<Impl>(domain, id)){}

        void ExecContext<cuda_backend>::prepare_for_reuse()
        {
            if (m_impl->active)
            {
                CALL_AND_HANDLE(cuda_safe_call(cudaStreamWaitEvent(m_impl->stream,m_impl->even, 0)), "Failed when waiting for cudaEvent");
                m_impl->active = false;
            }
        }

        void ExecContext<cuda_backend>::mark_submitted()
        {
            CALL_AND_HANDLE(cuda_safe_call(cudaEventRecord(m_impl->event, m_impl->stream)), "Failed when recording cudaEvent");
            m_impl->active = true;
        }

        void ExecContext<cuda_backend>::synchronise()
        {
            if (m_impl->active)
            {
                CALL_AND_HANDLE(cuda_safe_call(cudaEventSynchronize(m_impl->event)), "Failed when synchronising on cudaEvent");
                m_impl->active = false;
            }
        }

        // Make this context wait on another context
        void ExecContext<cuda_backend>::wait_for(const ExecContext<cuda_backend>& other)
        {
            CALL_AND_HANDLE(cuda_safe_call(cudaStreamWaitEvent(m_impl->stream, static_cast<cudaEvent_t>(other.event()),0)), "Failed when waiting on another event");
        }

        void* ExecContext<cuda_backend>::stream() const{return static_cast<void*>(m_impl->stream);}
        void* ExecContext<cuda_backend>::stream() const{return static_cast<void*>(m_impl->event);}
        void* ExecContext<cuda_backend>::cublas_handle() const{return static_cast<void*>(m_impl->cublas_handle);}
        void* ExecContext<cuda_backend>::cusparse_handle() const{return static_cast<void*>(m_impl->cusparse_handle);}
        void* ExecContext<cuda_backend>::cusolver_handle() const{return static_cast<void*>(m_impl->cusolver_handle);}
        void* ExecContext<cuda_backend>::cutensor_handle() const{return static_cast<void*>(m_impl->cutensor_handle);}

    }
}
