add_library(ttnpp_core INTERFACE)

target_include_directories(ttnpp_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${CEREAL_INCLUDE_DIRS}
        $<$<BOOL:${USE_SPDLOG}>:${SPDLOG_INCLUDE_DIRS}>
)

# Compile definitions
if(BUILD_REAL_TTN)
    target_compile_definitions(ttnpp_core INTERFACE BUILD_REAL_TTN)
endif()
if(DEBUG)
    target_compile_definitions(ttnpp_core INTERFACE DEBUG)
endif()
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_compile_definitions(ttnpp_core INTERFACE USE_OPENMP)
    target_link_libraries(ttnpp_core INTERFACE OpenMP::OpenMP_CXX)
endif()
if(USE_GEMM3M)
    target_compile_definitions(ttnpp_core INTERFACE USE_GEMM3M)
endif()

if(USE_CUDA)
    target_compile_definitions(ttnpp_core INTERFACE PYTTN_BUILD_CUDA)
endif()

# Link BLAS/LAPACK
target_link_libraries(ttnpp_core INTERFACE ${LINALG_LIBRARIES})

# Compiler-specific flags
if(MSVC)
    target_compile_options(ttnpp_core INTERFACE /W4 /O2)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ttnpp_core INTERFACE  -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    target_compile_options(ttnpp_core INTERFACE -O3 -xHost -qopt-report=3 -restrict)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    target_compile_options(ttnpp_core INTERFACE -O3 -stdlib=libc++)
endif()