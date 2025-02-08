cmake_minimum_required(VERSION 3.12)
project(ttnpp)

include(CheckIPOSupported)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

#compiler flags
option(USE_CUDA "Compile with cuda" OFF)
option(USE_OPENMP "Compile with openmp" OFF)
option(PARALLELISE_SET_VARIABLES "openmp parallelism over set variables" OFF)
option(PARALLELISE_HAMILTONIAN_SUM "openmp parallelism over hamiltonian sum" OFF)
option(BUILD_REAL_TTN "build wrapper for real valued tensor network objects." OFF)
option(USE_GEMM3M "Compile with openmp" OFF)
option(EVAL_BUT_DONT_APPLY "Compile with eval but don't apply option set" OFF)

if(PARALLELISE_SET_VARIABLES)
  add_definitions(-DPARALLELISE_SET_VARIABLES)
  set(USE_OPENMP ON)
endif()

if(PARALLELISE_HAMILTONIAN_SUM)
  add_definitions(-DPARALLELISE_HAMILTONIAN_SUM)
  set(USE_OPENMP ON)
endif()

if(USE_OPENMP)
  add_definitions(-DUSE_OPENMP)
endif()


if(BUILD_REAL_TTN)
  add_definitions(-DBUILD_REAL_TTN)
endif()


include(CheckLanguage)
check_language(CUDA)

#attempt to set up cuda if the user has requested it and cuda is installed
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    add_definitions(-DPYTTN_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(Thrust REQUIRED)
    thrust_create_target(Thrust)

    #TODO: Need to set up search for cutensor

    set(PYTHON_BINDINGS_SRC pyttn.cpp 
          linalg/tensor.cpp linalg/sparseMatrix.cpp linalg/backend.cpp linalg/orthogonal_vector_gen.cpp
          utils/orthopol.cpp utils/discretisation.cpp
          ttns/sop/sSOP.cpp ttns/sop/SOP.cpp ttns/sop/system_information.cpp ttns/sop/operator_dictionary.cpp ttns/sop/liouville_space.cpp ttns/sop/models/models.cpp
          ttns/ttn/ntree.cpp ttns/ttn/ttn.cpp ttns/ttn/ms_ttn.cpp
          ttns/observables/matrix_element.cpp 
          ttns/operators/siteOperators.cpp ttns/operators/product_operator.cpp ttns/operators/sop_operator.cpp
          ttns/algorithms/tdvp.cpp ttns/algorithms/dmrg.cpp
          )
    set_source_files_properties(${PYTHON_BINDINGS_SRC} PROPERTIES LANGUAGE CUDA)
    if(COMPILER_SET)
    else()
        set(CMAKE_CUDA_FLAGS " -x cu -Xptxas -O3 -g --extended-lambda ${CMAKE_CUDA_FLAGS}")

        if(USE_OPENMP)
            set(CMAKE_CUDA_FLAGS "-x cu ${CMAKE_CUDA_FLAGS} -fopenmp")
        endif()

        if(NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 17)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif()

        set(LINALG_LIBS ${LINALG_LIBS} ${LLVM_LIBRARIES} ${CLANG_LIBS} ${CLANG_CLANG_LIBS} ${CUDA_cusparse_LIBRARY})
    endif()
else()
    set(PYTHON_BINDINGS_SRC pyttn.cpp 
          utils/orthopol.cpp utils/discretisation.cpp
          linalg/sparseMatrix.cpp linalg/tensor.cpp linalg/backend.cpp
          ttns/observables/matrix_element.cpp 
          ttns/operators/siteOperators.cpp ttns/operators/product_operator.cpp ttns/operators/sop_operator.cpp
          ttns/sop/sSOP.cpp ttns/sop/SOP.cpp ttns/sop/system_information.cpp ttns/sop/operator_dictionary.cpp ttns/sop/liouville_space.cpp ttns/sop/models/models.cpp
          ttns/ttn/ntree.cpp ttns/ttn/ttn.cpp ttns/ttn/ms_ttn.cpp
          ttns/algorithms/tdvp.cpp ttns/algorithms/dmrg.cpp
          )


    if(COMPILER_SET)
    else()
        if(MSVC)
            message(FATAL_ERROR "Request MSVC compiler.  I don't know how this works so exiting.")
        else()
            set(CMAKE_CXX_FLAGS "-O3 -g -Wall -Wextra -Wshadow -pedantic -Wold-style-cast")
            option(WITH_WERROR "Compile with '-Werror' C++ compiler flag" ON)
            if(WITH_WERROR)
                #set(CMAKE_CXX_FLAGS "-Werror ${CMAKE_CXX_FLAGS}")
            endif(WITH_WERROR)

            #additional compilation flags for GNU compilers
            if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcx-fortran-rules -march=native -mtune=native")
                if(USE_OPENMP)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
                endif()
            #additional compilation flags for inteal compilers
            elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=3 -O3 -xHost -restrict")
                message(STATUS "using intel compiler.")
                if(USE_OPENMP)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
                endif()

            elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -qopt-report=3 ")
                if(USE_OPENMP)
                    message(STATUS "Using Openmp")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
                endif()

            elseif ( ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang") )
                find_package(LLVM REQUIRED)
                find_package(Clang REQUIRED)

                set(LINALG_LIBS ${LINALG_LIBS} ${LLVM_LIBRARIES} ${CLANG_LIBS} ${CLANG_CLANG_LIBS})
                

                #if we are using clang then find threads
                set(CMAKE_THREAD_LIBS_INIT "-lpthread")
                set(CMAKE_HAVE_THREADS_LIBRARY 1)
                set(CMAKE_USE_WIN32_THREADS_INIT 0)
                set(CMAKE_USE_PTHREADS_INIT 1)
                set(THREADS_PREFER_PTHREAD_FLAG ON)

                set(CMAKE_CXX_FLAGS "-stdlib=libc++ ${CMAKE_CXX_FLAGS}")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ ")


                if(USE_OPENMP)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libomp")
                endif()
            endif()
            

            if(CMAKE_VERSION VERSION_LESS 3.1)
                set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
            else()
              if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD STREQUAL "98")
                set(CMAKE_CXX_STANDARD 17)
              endif()

              if(CMAKE_CXX_STANDARD GREATER 17)
                cmake_minimum_required(VERSION 3.8)
              endif()

              set(CMAKE_CXX_STANDARD_REQUIRED ON)
            endif()
        endif()
    endif()
endif()

set(CMAKE_BUILD_PARALLEL_LEVEL 8)
set(PYBIND11_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/Pybind11)

include(FetchContent)
if(NOT EXISTS ${PYBIND11_ROOT_DIR})
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
        SOURCE_DIR ${PYBIND11_ROOT_DIR}
    )

else()
    message(STATUS "${PYBIND11_ROOT_DIR}")
    FetchContent_Declare(
        pybind11 SOURCE_DIR ${PYBIND11_ROOT_DIR}
    )
endif()

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
add_definitions(-DPYBIND11)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "${PYTHON_BINDINGS_SRC}")
pybind11_add_module(ttnpp ${PYTHON_BINDINGS_SRC})
target_include_directories(ttnpp PUBLIC ${TTNS_INCLUDE_DIRECTORIES})

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_link_libraries(ttnpp PUBLIC pybind11::module ${LINALG_LIBS} ${LIBS} CUDA::cusparse CUDA::cublas CUDA::cusolver CUDA::curand Thrust ) 
    set_target_properties(ttnpp PROPERTIES CUDA_ARCHITECTURES "86")
else()
    target_link_libraries(ttnpp PUBLIC pybind11::module ${LINALG_LIBS} ${LIBS}) 
endif()


add_library(restore_default_visibility INTERFACE)
target_compile_options(restore_default_visibility INTERFACE -fvisibility=default)
target_link_libraries(ttnpp PRIVATE restore_default_visibility)

#set_target_properties(ttnpp PROPERTIES
#        VISIBILITY_INLINES_HIDDEN ON)
check_ipo_supported(RESULT ipo)
if(ipo)
set_property(TARGET ttnpp PROPERTY VISIBILITY_INLINES_HIDDEN TRUE)
endif()
if(EVAL_BUT_DONT_APPLY)
  target_compile_definitions(ttnpp PUBLIC ALLOW_EVAL_BUT_DONT_APPLY=0)
endif()

install(
    TARGETS ttnpp
    DESTINATION ${CMAKE_INSTALL_PREFIX}
)




