#ifndef LINALG_ALGEBRA_CUSPARSE_WRAPPER_HPP
#define LINALG_ALGEBRA_CUSPARSE_WRAPPER_HPP

#ifdef PYTTN_BUILD_CUDA

#include "cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cutensor.h>


#include <unordered_map>

namespace linalg
{
namespace cutensor
{

static inline void cutensor_safe_call(cutensorStatus_t err){if(err != CUTENSOR_STATUS_SUCCESS){RAISE_EXCEPTION_STR(cutensorGetErrorSTRING(err));}}


//Function for getting cutensor type from the template type
template <typename T> class cutensor_type;

template <>
class cutensor_type<float>
{
    static cutensorDataType_t type(){return CUTENSOR_R_32F;}
    static constexpr cutensorComputeDescriptor_t compute_type(){return CUTENSOR_COMPUTE_DESC_32F;}
};

template <>
class cutensor_type<double>
{
    static cutensorDataType_t type(){return CUTENSOR_R_64F;}
    static constexpr cutensorComputeDescriptor_t compute_type(){return CUTENSOR_COMPUTE_DESC_64F;}
};

template <>
class cutensor_type<complex<float>>
{
    static cutensorDataType_t type(){return CUTENSOR_C_32F;}
    static constexpr cutensorComputeDescriptor_t compute_type(){return CUTENSOR_COMPUTE_DESC_32F;}
};

template <>
class cutensor_type<complex<double>>
{
    static cutensorDataType_t type(){return CUTENSOR_C_64F;}
    static constexpr cutensorComputeDescriptor_t compute_type(){return CUTENSOR_COMPUTE_DESC_64F;}
};


cutensorTensorDescriptor_t create_tensor_descriptor(cutensorHandle_t handle, size_t N, const int64_t* extent, const int64_t* stride)
{
    cutensorTensorDescriptor_t desc;

    using ctt = cutensor_type<T>;

    size_t nmode = dims.size();
    call_and_handle(cutensor_safe_call(cutensorcreatetensordescriptor(handle, &desc, N, extent, stride, ctt::type(), 256)), "failed to create tensor descriptor");

    return desc;
}

void destroy_tensor_descriptor(cutensorTensorDescriptor_t desc)
{
    CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroyTensorDescriptor(desc)), "Failed to destroy cutensor descriptor");
}

void transpose(cutensorHandle_t handle, const T* A, const int64_t* extentA, const int64_t* strideA, const T* B, const int64_t* extentB, const int64_t* strideB, const std::vector<size_type>& inds)
{
    try
    {
        cutensorComputeDescriptor const descCompute = cutensor_type<T>::compute_type();
        //set up the cutensor tensor descriptors
        cutensorTensorDescriptor_t descA, descB;
        descA = create_tensor_descriptor(handle, extentA, strideA);
        descB = create_tensor_descriptor(handle, extentB, strideB);

        ASSERT(inds.size() < 26, "The cutensor wrapper currently only supports tensors with 26 or fewer modes.");

        //get the modes of A (the tensor we are not permuting
        std::vector<int> modeA(inds.size());
        for(size_t i = 0; i < inds.size(); ++i)
        {
            modesA[i] = i+97;
        }

        //set up the modes of B
        std::vector<int> modeB(inds.size());
        for(size_t i = 0; i < inds.size(); ++i)
        {
            modesB[i] = inds[i]+97;
        }

        cutensorOperationDescriptor_t perm;
        CALL_AND_HANDLE(cutensor_safe_call(cutensorCreatePermutation(handle, &perm, descA, modesA, CUTENSOR_OP_IDENTITY, descB, descCompute)), "Failed to construct permutation plan.");

        //TODO: ensure that the scalar type is correct

        //set the algorithm to use
        const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

        //create the plan for performing the permutation
        cutensorPlanPreference_t planPref;
        CALL_AND_HANDLE(cutensor_safe_call(cutensorCreatePlanPreference(handle, &planPref, algo, CUTENSOR_JIT_MODE_NONE)), "Failed to construct permutation plan preferences.");
        cutensorPlan_t plan;
        CALL_AND_HANDLE(cutensor_safe_call(cutensorCreatePlanPreference(handle, &plan, desc, planPref, 0)), "Failed to construct permutation plan.");

        T alpha(1.0);
        //now perform the permutation
        CALL_AND_HANDLE(cutensor_safe_call(cutensorPermute(handle, plan, &alpha, A, C, _environment.current_stream())), "Failed to construct permutation plan.");

        //destroy the tensor descriptors
        CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroyPlan(plan)), "Failed to destroy plan.");
        CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroyOperationDescription(desc)), "Failed to destroy plan.");
        CALL_AND_HANDLE(cutensor_safe_call(cutensorDestroyPlanPreference(planPref)), "Failed to destroy planPref.");
        destroy_tensor_descriptor(descA);
        destroy_tensor_descriptor(descB);
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


