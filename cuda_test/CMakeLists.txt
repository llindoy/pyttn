cmake_minimum_required(VERSION 3.12)
project(operator_test)

include(CheckIPOSupported)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


option(BUILD_SIAM "build siam" OFF)

if(USE_GEMM3M)
    add_compile_definitions(USE_GEMM3M)
endif()


if(BUILD_REAL_TTN)
  add_definitions(-DBUILD_REAL_TTN)
endif()

if(DEBUG)
    add_definitions(-DDEBUG)
endif()

if(TRACE_LOG)
    add_definitions(-DTRACE_LOG)
endif()

include(CheckLanguage)
check_language(CUDA)

#attempt to set up cuda if the user has requested it and cuda is installed

add_executable(operator_test.x  operator_test.cu
    ${CMAKE_SOURCE_DIR}/cuda/cuda_backend.cu ${CMAKE_SOURCE_DIR}/cuda/cuda_environment.cu

)

add_definitions(-DPYTTN_BUILD_CUDA)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(CUTENSOR REQUIRED)
message(STATUS ${CUTENSOR_LIBRARY})
message(STATUS ${CUTENSOR_INCLUDE_PATH})
#set_source_files_properties(${PYTHON_BINDINGS_SRC} PROPERTIES LANGUAGE CUDA)
 set(CUDA_WARNING_SUPPRESSION "-Xcudafe --diag_suppress=128 -Xcudafe --diag_suppress=2417")

set(CMAKE_CUDA_FLAGS "${CUDA_WARNING_SUPPRESSION} -g -Xptxas -O3 --extended-lambda --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LINALG_LIBS ${LINALG_LIBS} ${LLVM_LIBRARIES} ${CLANG_LIBS} ${CUDA_cusparse_LIBRARY})



target_include_directories(operator_test.x PUBLIC ${CMAKE_SOURCE_DIR}/ttnpp   ${CEREAL_INCLUDE_DIRS}
        $<$<BOOL:${USE_SPDLOG}>:${SPDLOG_INCLUDE_DIRS}>)
set_target_properties(operator_test.x PROPERTIES CUDA_ARCHITECTURES "86")
set_target_properties(operator_test.x PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(operator_test.x SYSTEM PUBLIC ${TTNS_INCLUDE_DIRECTORIES} ${cereal_INCLUDE_DIR} ${spdlog_INCLUDE_DIR})
#target_link_libraries(cudalib ${LINALG_LIBS} ${LIBS} ${OpenMP_LIBS} CUDA::cusparse CUDA::cublas CUDA::cusolver CUDA::curand CUDA::cutensor cUDA::cudart) 
target_link_libraries(operator_test.x PUBLIC ${LINALG_LIBS} ${LIBS} ${OpenMP_LIBS} CUDA::cusparse CUDA::cublas CUDA::cusolver CUDA::curand CUDA::cutensor CUDA::cudart ) 

check_ipo_supported(RESULT ipo)
if(ipo)
set_property(TARGET operator_test.x PROPERTY VISIBILITY_INLINES_HIDDEN TRUE)
endif()

install(TARGETS operator_test.x
        DESTINATION ${CMAKE_SOURCE_DIR}/bin)