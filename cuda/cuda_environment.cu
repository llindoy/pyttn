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

#include <linalg/backends/cuda/cuda_environment.hpp>
#include <linalg/backends/cuda/cuda_utils.cuh>

#include <linalg/backends/cuda/cublas_wrapper.cuh>
#include <linalg/backends/cuda/cusolver_wrapper.cuh>
#include <linalg/backends/cuda/cusparse_wrapper.cuh>
#include <linalg/backends/cuda/cutensor_wrapper.cuh>

#include <common/exception_handling.hpp>

#include <vector>
#include <tuple>
#include <utility>
#include <iostream>

#include <cuda_runtime.h>



std::ostream &operator<<(std::ostream &out, const cudaDeviceProp &prop)
{
    out << "\tDevice Name: " << prop.name << std::endl;
    out << "\tCompute Capability: " << prop.major << "." << prop.minor << std::endl;
    out << "\tCompute Mode: " << (prop.computeMode == cudaComputeModeDefault ? "Multithreaded" : (prop.computeMode == cudaComputeModeExclusive ? "Singlethreaded" : "No")) << " Device Access" << std::endl;
    out << "\tConcurrent Kernel Execution: " << (prop.concurrentKernels == 1 ? "True" : "False") << std::endl
        << std::endl;
    out << "\tClock Speed (GHz): " << prop.clockRate / 1.0e6 << std::endl
        << std::endl;
    out << "\tMemory Clock Speed (GHz): " << prop.memoryClockRate / 1.0e6 << std::endl;
    out << "\tTotal Global Memory (GB): " << prop.totalGlobalMem / (1.0e9) << std::endl;
    out << "\tMemory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    out << "\tPeak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl
        << std::endl;
    out << "\tWarp Size: " << prop.warpSize << std::endl;
    out << "\tMaximum Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    out << "\tMaximum Thread Dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    out << "\tMaximum Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    return out;
}

namespace linalg
{

    struct cuda_environment::impl
    {
        // the cuda device properties
        int device_id = 0;
        bool initialised = false;

        cudaDeviceProp prop;

        size_t nstreams;
        size_t current_stream;
        std::vector<cudaStream_t> streams;

        cusparseHandle_t cusparse_handle;
        cublasHandle_t cublas_handle;
        cusolverDnHandle_t cusolver_dn_handle;
        cutensorHandle_t cutensor_handle;

        impl() = default;
    };

/*
 *  Constructor implementaiton
 */
cuda_environment::cuda_environment() : m_impl(new impl()) 
{
    init(0, 1);
}
cuda_environment::cuda_environment(int device_id, int nstreams) : m_impl(new impl())
{
    CALL_AND_HANDLE(init(device_id, nstreams), "Failed to construct cuda_environment object.  Error when initialising the device properties.");
}
cuda_environment::cuda_environment(cuda_environment &&other) : m_impl(other.m_impl)
{
    other.m_impl = nullptr;
}

cuda_environment & cuda_environment::operator=(cuda_environment &&other)  noexcept 
{
    if(this != &other)
    {
        delete m_impl;
        m_impl = other.m_impl;
        other.m_impl = nullptr;
    }
    return *this;
}

cuda_environment::~cuda_environment() 
{
    if(m_impl)
    {
        destroy();
        delete m_impl;
    }
}



// functions for updating the state of the cuda_environment
void cuda_environment::init(int device_id, int nstreams)
{
    // ASSERT(!m_initialised, "Failed to initialise cuda_environment object.  Cannot initialise an already initialised object.");
    if (m_impl->initialised)
    {
        destroy();
    }

    m_impl->device_id = device_id;

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    ASSERT(device_id < nDevices, "Failed to initialise cuda_environment object.  The requested device id does not exist.");

    // initialise the device object
    CALL_AND_HANDLE(cuda_safe_call(cudaGetDeviceProperties(&m_impl->prop, device_id)), 
        "Failed to initialise cuda_environment object.  Error when accessing device properties.");
    CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(device_id)), 
        "Failed to initialise the cuda_environment object.  Error when calling cudaSetDevice.");

    // now set up the stream objects
    if (nstreams < 1){nstreams = 1;}
    m_impl->nstreams = nstreams;
    m_impl->streams.resize(nstreams - 1);
    m_impl->current_stream = 0;

    for (size_t i = 0; i < nstreams - 1; ++i){cudaStreamCreate(&m_impl->streams[i]);}

    //create library handles
    CALL_AND_HANDLE(cublas_safe_call(cublasCreate(&m_impl->cublas_handle)), "Failed to initialise cuda_environment object.  Error when setting up cublas_handle object.");
    CALL_AND_HANDLE(cusparse_safe_call(cusparseCreate(&m_impl->cusparse_handle)), "Failed to initialise cuda_environment object.  Error when setting up the cusparse_handle object.");
    CALL_AND_HANDLE(cusolver_safe_call(cusolverDnCreate(&m_impl->cusolver_dn_handle)), "Failed to initialise cuda_environment object.  Error when setting up cusolver_dn_handle object.");
    CALL_AND_HANDLE(cutensor_safe_call(cutensorCreate(&m_impl->cutensor_handle)), "Failed to initialise cuda_environment object.  Error when setting up cutensor_handle object.");
    m_impl->initialised = true;
}

void cuda_environment::destroy()
{
    if (m_impl->initialised)
    {
        for (size_t i = 0; i < m_impl->nstreams - 1; ++i)
        {
            CALL_AND_HANDLE(cuda_safe_call(cudaStreamDestroy(m_impl->streams[i])), "Failed to destroy cuda_environment object.  Error when destroying stream objects.");
        }
        CALL_AND_HANDLE(cublas_safe_call(cublasDestroy(m_impl->cublas_handle)), "Failed to destroy cuda_environment object.  Error when destroying cublas_handle object.");
        CALL_AND_HANDLE(cusolver_safe_call(cusolverDnDestroy(m_impl->cusolver_dn_handle)), "Failed to destroy cuda_environment object.  Error when destroying cusolver_dn_handle object.");
        CALL_AND_HANDLE(cusparse_safe_call(cusparseDestroy(m_impl->cusparse_handle)), "Failed to destroy cuda_environment object.  Error when destroying cusparse_handle object.");
        CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroy(m_impl->cutensor_handle)), "Failed to initialise cuda_environment object.  Error when destroying cutensor_handle object.");
        m_impl->initialised = false;
    }
}

/*
 *  Accessor and helper functions
 */

bool cuda_environment::is_initialised() const { return m_impl->initialised; }

void* cuda_environment::current_stream() const { return m_impl->current_stream == 0 ? 0 : m_impl->streams[m_impl->current_stream - 1]; }
void cuda_environment::increment_stream_id()
{
    ++m_impl->current_stream;
    m_impl->current_stream = (m_impl->current_stream == m_impl->nstreams) ? 0 : m_impl->current_stream;
}

void cuda_environment::reset_stream_id() { m_impl->current_stream = 0; }

// accessors device specific properties required for determining kernel execution parameters
size_t cuda_environment::total_global_memory() const
{
    ASSERT(m_impl->initialised, "Failed to access the cuda_environment object's amount of global memory.  The cuda_environment has not been initialised.");
    return m_impl->prop.totalGlobalMem;
}
size_t cuda_environment::shared_mem_per_block() const
{
    ASSERT(m_impl->initialised, "Failed to access the cuda_environment object's amount of shared mem per block.  The cuda_environment has not been initialised.");
    return m_impl->prop.sharedMemPerBlock;
}
int cuda_environment::warpsize() const
{
    ASSERT(m_impl->initialised, "Failed to access the cuda_environment object's warp size.  The cuda_environment has not been initialised.");
    return m_impl->prop.warpSize;
}
int cuda_environment::maximum_threads_per_block() const
{
    ASSERT(m_impl->initialised, "Failed to access the cuda_environment object's maximum threads per block.  The cuda_environment has not been initialised.");
    return m_impl->prop.maxThreadsPerBlock;
}
std::array<int, 3> cuda_environment::maximum_dimensions_threads_per_block() const
{
    ASSERT(m_impl->initialised, "Failed to access the cuda_environment object's maximum thread dimension.  The cuda_environment has not been initialised.");
    return std::array<int, 3>{{m_impl->prop.maxThreadsDim[0], m_impl->prop.maxThreadsDim[1], m_impl->prop.maxThreadsDim[2]}};
}

const void* cuda_environment::cublas_handle() const
{
    ASSERT(m_impl->initialised, "Failed to access the cuda_environment object's cublas handle.  The cuda_environment has not been initialised.");
    return m_impl->cublas_handle;
}

const void* cuda_environment::cusolver_dn_handle() const
{
    ASSERT(m_impl->initialised, "Failed to access the cuda_environment object's cusolver dense handle.  The cuda_environment has not been initialised.");
    return m_impl->cusolver_dn_handle;
}

const void* cuda_environment::cusparse_handle() const
{
    ASSERT(m_impl->initialised, "Failed to acces the cuda_environment object's cusparse handle.  The cuda_environment has not been initialised.");
    return m_impl->cusparse_handle;
}

const void* cuda_environment::cutensor_handle() const
{
    ASSERT(m_impl->initialised, "Failed to acces the cuda_environment object's cutensor handle.  The cuda_environment has not been initialised.");
    return m_impl->cutensor_handle;
}

void cuda_environment::set_device() const
{
    ASSERT(m_impl->initialised, "Failed to move to cuda_environment's device.  The cuda_environment has not been initialised.");
    CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(m_impl->device_id)), "Failed to move to cuda_environment's device.  Error when calling cudaSetDevice.");
}

// accessors for general properties of the cuda install that do not require a cuda_environment instance
int cuda_environment::number_of_devices()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    return nDevices;
}

std::ostream &cuda_environment::list_devices(std::ostream &out)
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        out << "Device Number: " << i << std::endl;
        out << prop;
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, const cuda_environment &inst)
{
    out << "Device Number: " << inst.m_impl->device_id << std::endl;
    out << inst.m_impl->prop << std::endl;
    return out;
}

} // namespace linalg

