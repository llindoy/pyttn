if(NOT USE_CUDA)
    message(STATUS "Skipping CUDA library")
    return()
endif()

# Default CUDA architecture if not set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    # Safe defaults for Turing/ Ampere GPUs
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 CACHE STRING "CUDA architectures to build for" FORCE)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES was not set. Defaulting to: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

enable_language(CUDA)

# -------------------- CUDA source files --------------------
set(CUDA_COMMON_SRCS
    cuda_backend.cu 
    cuda_environment.cu
)

set(CUDA_SRCS ${CUDA_COMMON_SRCS})

# -------------------- Object library --------------------
# Build static CUDA library (avoids fatbin duplicate issues)
add_library(cuda_backend STATIC ${CUDA_SRCS})
set_target_properties(cuda_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

set(CMAKE_CUDA_FLAGS "-Xptxas -O3 --extended-lambda ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(cuda_backend PUBLIC
    ${CMAKE_SOURCE_DIR}/ttnpp
    ${cereal_SOURCE_DIR}/include
    $<$<BOOL:${USE_SPDLOG}>:${spdlog_SOURCE_DIR}/include>
)

# -------------------- Link CUDA libraries --------------------
target_link_libraries(cuda_backend
    PUBLIC
        ttnpp_core
        CUDA::cublas
        CUDA::cusparse
        CUDA::cusolver
        CUDA::curand
        CUDA::cutensor
        CUDA::cudart
)

# -------------------- CUDA compiler flags --------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(cuda_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -g -Xptxas -v --extended-lambda>)
elseif(MSVC)
    target_compile_options(cuda_backend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:/O2 /arch:sm_86>)
endif()