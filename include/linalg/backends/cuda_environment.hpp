#ifndef LINALG_BACKENDS_CUDA_ENVIRONMENT_HPP
#define LINALG_BACKENDS_CUDA_ENVIRONMENT_HPP


#ifdef PYTTN_BUILD_CUDA
#include <vector>
#include <tuple>
#include <utility>

#include <cuda_runtime.h>

#include "cuda_utils.hpp"
#include "cublas_wrapper.hpp"
#include "cusolver_wrapper.hpp"
#include "cusparse_wrapper.hpp"
//#include "cutensor_wrapper.hpp"

inline std::ostream& operator<<(std::ostream& out, const cudaDeviceProp& prop)
{
    out << "\tDevice Name: " << prop.name << std::endl;
    out << "\tCompute Capability: " << prop.major << "." << prop.minor << std::endl;
    out << "\tCompute Mode: " << (prop.computeMode == cudaComputeModeDefault ? "Multithreaded" : (prop.computeMode == cudaComputeModeExclusive ? "Singlethreaded" : "No"))  << " Device Access" << std::endl;
    out << "\tConcurrent Kernel Execution: " << (prop.concurrentKernels == 1 ? "True" : "False") << std::endl << std::endl;
    out << "\tClock Speed (GHz): " << prop.clockRate/1.0e6 << std::endl << std::endl;
    out << "\tMemory Clock Speed (GHz): " << prop.memoryClockRate/1.0e6 << std::endl;
    out << "\tTotal Global Memory (GB): " << prop.totalGlobalMem/(1.0e9) << std::endl;
    out << "\tMemory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    out << "\tPeak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl << std::endl;
    out << "\tWarp Size: " << prop.warpSize << std::endl;
    out << "\tMaximum Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    out << "\tMaximum Thread Dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")"  << std::endl;
    out << "\tMaximum Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")"  << std::endl;
    return out;
}


namespace linalg
{

class cuda_backend;

//class wrapping the cuda environment.  This allows for easy setup of devices and allocation of global workspace variables
//that are required for performing linear algebra operations.
class cuda_environment
{
public:
    using size_type = std::size_t;
    using index_type = int32_t;
    using stream_list = std::vector<cudaStream_t>;

    //this is a friend of the cuda_backend type
    friend class cuda_backend;

protected:
    //the cuda device properties
    int m_device_id;
    cudaDeviceProp m_prop;
    bool m_initialised;

    size_type m_nstreams;
    size_type m_current_stream;
    stream_list m_streams;

    cusparseHandle_t m_cusparse_handle;
    cublasHandle_t m_cublas_handle;
    cusolverDnHandle_t m_cusolver_dn_handle;
    //cutensorHandle_t m_cutensor_handle;

    friend std::ostream& operator<<(std::ostream& out, const cuda_environment& s);
public:
    cuda_environment() : m_initialised(false) {}
    cuda_environment(int device_id, int nstreams) : m_device_id(device_id), m_current_stream(0), m_initialised(false)
    {
        CALL_AND_HANDLE(init(device_id, nstreams), "Failed to construct cuda_environment object.  Error when initialising the device properties.");
    }

    ~cuda_environment(){}

    cuda_environment& operator=(cuda_environment&& other)
    {
        m_device_id = std::move(other.m_device_id); other.m_device_id = 0;
        m_prop = std::move(other.m_prop);
        m_initialised = std::move(other.m_initialised); other.m_initialised = false;
        m_nstreams = std::move(other.m_nstreams);   
        other.m_nstreams = 0;
        m_current_stream = std::move(other.m_current_stream);   other.m_current_stream = 0;
        m_streams = std::move(other.m_streams); other.m_streams.clear();
        m_cublas_handle = std::move(other.m_cublas_handle); other.m_cublas_handle = 0;
        m_cusolver_dn_handle = std::move(other.m_cusolver_dn_handle); other.m_cusolver_dn_handle = 0;
        m_cusparse_handle = std::move(other.m_cusparse_handle); other.m_cusparse_handle = 0;
        //m_cutensor_handle = std::move(other.m_cutensor_handle); other.m_cutensor_handle = 0;
        return *this;
    }

    //functions for updating the state of the cuda_environment
    void init(int device_id, int nstreams = 1)
    {
        //ASSERT(!m_initialised, "Failed to initialise cuda_environment object.  Cannot initialise an already initialised object.");
        if(m_initialised)
        {
            destroy();
        }
        int nDevices;
        cudaGetDeviceCount(&nDevices);

        ASSERT(m_device_id < nDevices, "Failed to initialise cuda_environment object.  The requested device id does not exist.");

        //initialise the device object
        CALL_AND_HANDLE(cuda_safe_call(cudaGetDeviceProperties(&m_prop, device_id)), "Failed to initialise cuda_environment object.  Error when accessing device properties.");;
        CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(m_device_id)), "Failed to initialise the cuda_environment object.  Error when calling cudaSetDevice.");

        //now set up the stream objects
        if(nstreams < 1){nstreams = 1;}
        m_nstreams = nstreams;
        m_streams.resize(m_nstreams - 1);

        for(size_type i=0; i<m_nstreams-1; ++i){cudaStreamCreate(&m_streams[i]);}

        CALL_AND_HANDLE(cublas_safe_call(cublasCreate(&m_cublas_handle)), "Failed to initialise cuda_environment object.  Error when setting up cublas_handle object.");
        CALL_AND_HANDLE(cusparse_safe_call(cusparseCreate(&m_cusparse_handle)), "Failed to initialise cuda_environment object.  Error when setting up the cusparse_handle object.");
        CALL_AND_HANDLE(cusolver_safe_call(cusolverDnCreate(&m_cusolver_dn_handle)), "Failed to initialise cuda_environment object.  Error when setting up cusolver_dn_handle object.");
        //CALL_AND_HANDLE(cutensor_safe_call(cutensorCreate(&m_cutensor_handle)), "Failed to initialise cuda_environment object.  Error when setting up cutensor_handle object.");
        m_initialised = true;
    }

    void destroy()
    {
        if(m_initialised)
        {
            m_initialised = false;
            for(size_type i=0; i<m_nstreams-1; ++i){CALL_AND_HANDLE(cuda_safe_call(cudaStreamDestroy(m_streams[i])), "Failed to destroy cuda_environment object.  Error when destroying stream objects.");}
            CALL_AND_HANDLE(cublas_safe_call(cublasDestroy(m_cublas_handle)), "Failed to destroy cuda_environment object.  Error when destroying cublas_handle object.");
            CALL_AND_HANDLE(cusolver_safe_call(cusolverDnDestroy(m_cusolver_dn_handle)), "Failed to destroy cuda_environment object.  Error when destroying cusolver_dn_handle object.");
            CALL_AND_HANDLE(cusparse_safe_call(cusparseDestroy(m_cusparse_handle)), "Failed to destroy cuda_environment object.  Error when destroying cusparse_handle object.");
            //CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroy(m_cutensor_handle)), "Failed to initialise cuda_environment object.  Error when destroying cutensor_handle object.");
        }
    }   

    bool is_initialised() const{return m_initialised;}

    cudaStream_t current_stream() const{return m_current_stream == 0 ? 0 : m_streams[m_current_stream-1];}
    void increment_stream_id()
    {
        ++m_current_stream; m_current_stream = (m_current_stream == m_nstreams) ? 0 : m_current_stream;
    }

    void reset_stream_id(){m_current_stream = 0;}

    //accessors device specific properties required for determining kernel execution parameters
    size_type total_global_memory() const
    {
        ASSERT(m_initialised, "Failed to access the cuda_environment object's amount of global memory.  The cuda_environment has not been initialised.");
        return m_prop.totalGlobalMem;
    }
    size_type shared_mem_per_block() const
    {
        ASSERT(m_initialised, "Failed to access the cuda_environment object's amount of shared mem per block.  The cuda_environment has not been initialised.");
        return m_prop.sharedMemPerBlock;
    }
    int warpsize() const
    {
        ASSERT(m_initialised, "Failed to access the cuda_environment object's warp size.  The cuda_environment has not been initialised.");
        return m_prop.warpSize;
    }
    int maximum_threads_per_block() const
    {
        ASSERT(m_initialised, "Failed to access the cuda_environment object's maximum threads per block.  The cuda_environment has not been initialised.");
        return m_prop.maxThreadsPerBlock;
    }
    std::array<int,3> maximum_dimensions_threads_per_block() const
    {
        ASSERT(m_initialised, "Failed to access the cuda_environment object's maximum thread dimension.  The cuda_environment has not been initialised.");
        return std::array<int, 3>{{m_prop.maxThreadsDim[0], m_prop.maxThreadsDim[1], m_prop.maxThreadsDim[2]}};
    }

    const cublasHandle_t& cublas_handle() const
    {
        ASSERT(m_initialised, "Failed to access the cuda_environment object's cublas handle.  The cuda_environment has not been initialised.");
        return m_cublas_handle;
    }

    const cusolverDnHandle_t& cusolver_dn_handle() const
    {
        ASSERT(m_initialised, "Failed to access the cuda_environment object's cusolver dense handle.  The cuda_environment has not been initialised.");
        return m_cusolver_dn_handle;
    }

    const cusparseHandle_t& cusparse_handle() const
    {
        ASSERT(m_initialised, "Failed to acces the cuda_environment object's cusparse handle.  The cuda_environment has not been initialised.");
        return m_cusparse_handle;
    }

    //const cusparseHandle_t& cutensor_handle() const
    //{
    //    ASSERT(m_initialised, "Failed to acces the cuda_environment object's cutensor handle.  The cuda_environment has not been initialised.");
    //    return m_cutensor_handle;
    //}

    void set_device() const
    {
        ASSERT(m_initialised, "Failed to move to cuda_environment's device.  The cuda_environment has not been initialised.");
        CALL_AND_HANDLE(cuda_safe_call(cudaSetDevice(m_device_id)), "Failed to move to cuda_environment's device.  Error when calling cudaSetDevice.");
    }

    //accessors for general properties of the cuda install that do not require a cuda_environment instance
    static int number_of_devices(){int nDevices;   cudaGetDeviceCount(&nDevices);  return nDevices;}

    static std::ostream& list_devices(std::ostream& out)
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
};


inline  std::ostream& operator<<(std::ostream& out, const cuda_environment& inst)
{
    out << "Device Number: " << inst.m_device_id << std::endl;
    out << inst.m_prop << std::endl;
    return out;
}

}   //namespace linalg


#endif


#endif  //LINALG_BACKENDS_CUDA_ENVIRONMENT_HPP//

