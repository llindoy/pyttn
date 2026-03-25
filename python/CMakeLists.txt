include(CheckIPOSupported)

# Default CUDA architecture if not set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    # Safe defaults for Turing/ Ampere GPUs
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 CACHE STRING "CUDA architectures to build for" FORCE)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES was not set. Defaulting to: ${CMAKE_CUDA_ARCHITECTURES}")
endif()


if(NOT USE_CUDA)
    set(BINDING_SRC
        pyttn.cpp
        linalg/tensor.cpp linalg/backend.cpp linalg/sparseMatrix.cpp linalg/orthogonal_vector_gen.cpp
        utils/orthopol.cpp utils/discretisation.cpp
        ttns/sop/sSOP.cpp ttns/sop/SOP.cpp ttns/sop/system_information.cpp ttns/sop/models/models.cpp
        ttns/sop/operator_dictionary.cpp ttns/sop/liouville_space.cpp 
        #ttns/sop/state.cpp 
        ttns/sop/toDense.cpp
        ttns/ttn/ntree.cpp ttns/ttn/ttn.cpp ttns/ttn/ms_ttn.cpp
        ttns/observables/matrix_element.cpp ttns/observables/rdm.cpp 
        ttns/operators/siteOperators.cpp ttns/operators/product_operator.cpp ttns/operators/sop_operator.cpp ttns/operators/Op.cpp
        ttns/algorithms/dmrg_onesite_ttn.cpp  ttns/algorithms/dmrg_adapt_ttn.cpp  ttns/algorithms/dmrg_onesite_msttn.cpp
        ttns/algorithms/tdvp_onesite_ttn.cpp  ttns/algorithms/tdvp_adapt_ttn.cpp  ttns/algorithms/tdvp_onesite_msttn.cpp
    )
else()
    add_compile_definitions(PYTTN_BUILD_CUDA)
    enable_language(CUDA)

    #suppress warnings arising from spdlog.
    set(CUDA_WARNING_SUPPRESSION "-Xcudafe --diag_suppress=128 -Xcudafe --diag_suppress=2417")

    set(CMAKE_CUDA_FLAGS "${CUDA_WARNING_SUPPRESSION} -Xptxas -O3 --extended-lambda --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS} ")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(BINDING_SRC
        pyttn.cpp
        cuda/linalg/tensor.cu cuda/linalg/backend.cu linalg/backend.cpp cuda/linalg/sparseMatrix.cu cuda/linalg/orthogonal_vector_gen.cu 

        utils/orthopol.cpp utils/discretisation.cpp
        ttns/sop/sSOP.cpp ttns/sop/SOP.cpp ttns/sop/system_information.cpp ttns/sop/models/models.cpp
        ttns/sop/operator_dictionary.cpp cuda/ttns/sop/operator_dictionary.cu
        cuda/ttns/sop/liouville_space.cu
        #ttns/sop/state.cpp 
        ttns/sop/toDense.cpp
        ttns/ttn/ntree.cpp cuda/ttns/ttn/ttn.cu cuda/ttns/ttn/ms_ttn.cu
        cuda/ttns/observables/matrix_element.cu cuda/ttns/observables/rdm.cu
        cuda/ttns/operators/siteOperators.cu cuda/ttns/operators/product_operator.cu cuda/ttns/operators/sop_operator.cu cuda/ttns/operators/Op.cu
        cuda/ttns/algorithms/dmrg_onesite_ttn.cu  cuda/ttns/algorithms/dmrg_adapt_ttn.cu  cuda/ttns/algorithms/dmrg_onesite_msttn.cu
        cuda/ttns/algorithms/tdvp_onesite_ttn.cu  cuda/ttns/algorithms/tdvp_adapt_ttn.cu  cuda/ttns/algorithms/tdvp_onesite_msttn.cu

        ${CMAKE_SOURCE_DIR}/cuda/cuda_backend.cu ${CMAKE_SOURCE_DIR}/cuda/cuda_environment.cu

    )
    set(CUDA_LIBS
        #cuda_backend
        CUDA::cublas
        CUDA::cusparse
        CUDA::cusolver
        CUDA::curand
        CUDA::cutensor
        CUDA::cudart
    )
endif()

message(STATUS "pyttn src: ${BINDING_SRC}")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ttnpp SHARED ${BINDING_SRC})
set_target_properties(ttnpp PROPERTIES PREFIX "")

target_include_directories(ttnpp PUBLIC ${CMAKE_SOURCE_DIR}/ttnpp  ${PYBIND11_INCLUDE_DIRS} ${CEREAL_INCLUDE_DIRS}
        $<$<BOOL:${USE_SPDLOG}>:${SPDLOG_INCLUDE_DIRS}>)

# Include directories
target_link_libraries(ttnpp PUBLIC 
        pybind11::module
        $<$<BOOL:${USE_CUDA}>:${CUDA_LIBS}>
        ${LINALG_LIBS}
)


if(EVAL_BUT_DONT_APPLY)
  target_compile_definitions(ttnpp PUBLIC ALLOW_EVAL_BUT_DONT_APPLY=0)
endif()


# macOS fix
if(APPLE)
    set_target_properties(ttnpp PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()


install(
    TARGETS ttnpp
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)
