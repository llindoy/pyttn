#ifndef LINALG_ALGEBRA_CUTENSOR_WRAPPER_HPP
#define LINALG_ALGEBRA_CUTENSOR_WRAPPER_HPP

#ifdef PYTTN_BUILD_CUDA

#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cutensor.h>


#include <unordered_map>

static inline void cutensor_safe_call(cutensorStatus_t err){if(err != CUTENSOR_STATUS_SUCCESS){RAISE_EXCEPTION_STR(cutensorGetErrorString(err));}}


namespace linalg
{
namespace cutensor
{

using size_type = std::size_t;
using index_type = int32_t;

//Function for getting cutensor type from the template type
template <typename T> class cutensor_type;

template <>
class cutensor_type<float>
{
public:
    static inline cutensorDataType_t type(){return CUTENSOR_R_32F;}
    static inline cutensorComputeDescriptor_t compute_type(){return CUTENSOR_COMPUTE_DESC_32F;}
};

template <>
class cutensor_type<double>
{
public:
    static inline cutensorDataType_t type(){return CUTENSOR_R_64F;}
    static inline cutensorComputeDescriptor_t compute_type(){return CUTENSOR_COMPUTE_DESC_64F;}
};

template <>
class cutensor_type<complex<float>>
{
public:
    static inline cutensorDataType_t type(){return CUTENSOR_C_32F;}
    static inline cutensorComputeDescriptor_t compute_type(){return CUTENSOR_COMPUTE_DESC_32F;}
};

template <>
class cutensor_type<complex<double>>
{
public:
    static inline cutensorDataType_t type(){return CUTENSOR_C_64F;}
    static inline cutensorComputeDescriptor_t compute_type(){return CUTENSOR_COMPUTE_DESC_64F;}
};

template <typename T>
inline cutensorTensorDescriptor_t create_tensor_descriptor(cutensorHandle_t handle, size_t N, const int64_t* extent, const int64_t* stride)
{
    cutensorTensorDescriptor_t desc;

    using ctt = cutensor_type<T>;

    CALL_AND_HANDLE(cutensor_safe_call(cutensorCreateTensorDescriptor(handle, &desc, N, extent, stride, ctt::type(), 256)), "failed to create tensor descriptor");

    return desc;
}

inline void destroy_tensor_descriptor(cutensorTensorDescriptor_t desc)
{
    CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroyTensorDescriptor(desc)), "Failed to destroy cutensor descriptor");
}

template <typename T> 
inline void transpose(cutensorHandle_t handle, cudaStream_t stream, const T* A, const std::vector<int64_t>& extentA, const std::vector<int64_t>& strideA, T* B, const std::vector<int64_t>& extentB, const std::vector<int64_t>& strideB, const std::vector<size_type>& inds)
{
    try
    {
        using ctt = cutensor_type<T>;

        cutensorComputeDescriptor_t descCompute = ctt::compute_type();
        //set up the cutensor tensor descriptors
        cutensorTensorDescriptor_t descA, descB;
        descA = create_tensor_descriptor<T>(handle, extentA.size(), extentA.data(), strideA.data());
        descB = create_tensor_descriptor<T>(handle, extentB.size(), extentB.data(), strideB.data());

        ASSERT(inds.size() < 26, "The cutensor wrapper currently only supports tensors with 26 or fewer modes.");

        //get the modes of A (the tensor we are not permuting
        std::vector<int> modesA(inds.size());
        for(size_t i = 0; i < inds.size(); ++i)
        {
            modesA[i] = i+97;
        }

        //set up the modes of B
        std::vector<int> modesB(inds.size());
        for(size_t i = 0; i < inds.size(); ++i)
        {
            modesB[i] = inds[i]+97;
        }

        cutensorOperationDescriptor_t perm;
        CALL_AND_HANDLE(cutensor_safe_call(cutensorCreatePermutation(handle, &perm, descA, modesA.data(), CUTENSOR_OP_IDENTITY, descB, modesB.data(), descCompute)), "Failed to construct permutation plan.");

        //set the algorithm to use
        const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

        //create the plan for performing the permutation
        cutensorPlanPreference_t planPref;
        CALL_AND_HANDLE(cutensor_safe_call(cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE)), "Failed to construct permutation plan preferences.");
        cutensorPlan_t plan;
        CALL_AND_HANDLE(cutensor_safe_call(cutensorCreatePlan(handle, &plan, perm, planPref, 0)), "Failed to construct permutation plan.");

        T alpha(1.0);
        //now perform the permutation
        CALL_AND_HANDLE(cutensor_safe_call(cutensorPermute(handle, plan, reinterpret_cast<const void*>(&alpha), reinterpret_cast<const void*>(A), reinterpret_cast<void*>(B), stream)), "Failed to construct permutation plan.");

        //destroy the tensor descriptors
        CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroyPlan(plan)), "Failed to destroy plan.");
        CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroyOperationDescriptor(perm)), "Failed to destroy plan.");
        CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroyPlanPreference(planPref)), "Failed to destroy planPref.");
        CALL_AND_RETHROW(destroy_tensor_descriptor(descA));
        CALL_AND_RETHROW(destroy_tensor_descriptor(descB));
    }
    catch(const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        RAISE_EXCEPTION("Failed to transpose tensor.");
    }
    

}

}   //namespace cusparse
}   //namespace linalg

#endif

#endif //LINALG_ALGEBRA_CUSPARSE_WRAPPER_HPP//


