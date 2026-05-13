/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
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

#include <linalg/linalg_forward_decl.hpp>
#include <common/exception_handling.hpp>

#include <linalg/utils/linalg_utils.hpp>


#include <linalg/backends/cuda/cuda_backend.hpp>
#include <linalg/backends/cuda/cuda_backend.cuh>
#include <linalg/backends/cuda/cuda_utils.cuh>
#include <linalg/backends/cuda/cuda_environment.hpp>

#include <linalg/backends/cuda/cublas_wrapper.cuh>
#include <linalg/backends/cuda/cusparse_wrapper.cuh>
#include <linalg/backends/cuda/cutensor_wrapper.cuh>
#include <linalg/backends/cuda/cusolver_wrapper.cuh>

#include <array>
#include <iostream>
#include <vector>
#include <tuple>
#include <utility>

#include <cuda_runtime.h>

namespace linalg
{

    // variable storing the cuda environment configuration
    cuda_environment &cuda_backend::environment()
    {
        static cuda_environment m_environment;
        // if the user has not initialised the environment variable then we will create a new environment variable using the default
        // arguments defined above
        if (!m_environment.is_initialised())
        {
            initialise_empty_ones_buffers();
            CALL_AND_HANDLE(m_environment.init(0, 1), "Failed to initialise cuda_backend.  Error when initialising the cuda environment object.");
        }
        return m_environment;
    }

    bool cuda_backend::is_initialised() { return environment().is_initialised(); }

    // the initialisation routines for the cuda_backend are not thread safe.
    void cuda_backend::initialise(cuda_environment &&env)
    {
        cuda_backend::initialise_empty_ones_buffers();
        environment() = std::move(env);
    }
    void cuda_backend::initialise(size_type device_id, size_type nstreams)
    {
        cuda_backend::initialise_empty_ones_buffers();
        CALL_AND_HANDLE(environment().init(device_id, nstreams), "Failed to initialise cuda_backend.  Error when initialising the cuda environment object.");
    }

    void cuda_backend::destroy()
    {
        CALL_AND_HANDLE(cuda_backend::clean_up_ones(), "Failed to destroy cuda_backend.  Error when clearing the allocated ones vectors.");
        CALL_AND_HANDLE(environment().destroy(), "Failed to destroy cuda_backend.  Error when destroying the cuda environment object.");
    }

    std::ostream &cuda_backend::device_properties(std::ostream &out)
    {
        try
        {
            return cuda_environment::list_devices(out);
        }
        catch (const std::exception &ex)
        {
            logging::error(ex.what());
            RAISE_EXCEPTION("Failed to print_type cuda_backend device properties.");
        }
    }

    void cuda_backend::synchronise()
    {
        cuda_safe_call((cudaDeviceSynchronize()));
    }

    void cuda_backend::set_cublas_stream()
    {
        CALL_AND_HANDLE(cublas_safe_call(cublasSetStream(hCublas(), hCuda())), "Failed to set the current value of the cublas stream.");
    }

    void cuda_backend::initialise_empty_ones_buffers()
    {
        clean_up_ones();
        cuda_internals::initialise_empty_ones_buffer<float>();
        cuda_internals::initialise_empty_ones_buffer<double>();
        cuda_internals::initialise_empty_ones_buffer<cuda::std::complex<float>>();
        cuda_internals::initialise_empty_ones_buffer<cuda::std::complex<double>>();
    }

    void cuda_backend::clean_up_ones()
    {
        cuda_internals::clean_up_ones<float>();
        cuda_internals::clean_up_ones<double>();
        cuda_internals::clean_up_ones<cuda::std::complex<float>>();
        cuda_internals::clean_up_ones<cuda::std::complex<double>>();
    }

    std::ostream &operator<<(std::ostream &out, const cuda_backend &inst)
    {
        out << inst.environment() << std::endl;
        return out;
    }

} // namespace linalg
